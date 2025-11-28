import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input, group=None):
        ctx.save_for_backward(input)
        ctx.group = group
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size(group))]
        dist.all_gather(output, input.contiguous(), group=group)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank(ctx.group)]
        return grad_out, None


def get_pos_and_neg_mask(num_samples, batch_size, world_size):
    """
    Creates a selection mask for which samples are positive and which are negative.

    We only use samples from other devices (ddp) as negatives.
    This is because a batch of images stem from the same spatial crop

    For example, given num_samples=3, a batch size of 2 and a world size of 2 we would have:

    Positive mask (X=select)            Negative mask (#=select)
    . X X . . . . . . . . .             . . . . . . # # # # # #
    X . X . . . . . . . . .             . . . . . . # # # # # #
    X X . . . . . . . . . .             . . . . . . # # # # # #
    . . . . X X . . . . . .             . . . . . . # # # # # #
    . . . X . X . . . . . .             . . . . . . # # # # # #
    . . . X X . . . . . . .             . . . . . . # # # # # #
    . . . . . . . X X . . .             # # # # # # . . . . . .
    . . . . . . X . X . . .             # # # # # # . . . . . .
    . . . . . . X X . . . .             # # # # # # . . . . . .
    . . . . . . . . . . X X             # # # # # # . . . . . .
    . . . . . . . . . X . X             # # # # # # . . . . . .
    . . . . . . . . . X X .             # # # # # # . . . . . .

    """

    eye = torch.eye(num_samples, num_samples, dtype=torch.uint8)

    pos_mask = torch.block_diag(*[1 - eye] * batch_size * world_size)

    neg_mask = 1 - torch.block_diag(
        *[torch.ones((num_samples * batch_size, num_samples * batch_size), dtype=torch.uint8)] * world_size
    )

    return pos_mask.type(torch.bool), neg_mask.type(torch.bool)


class LandCoverGroupContrastLossV2(nn.Module):
    """
    Contrastive loss per group of products.

    Positive samples are averaged global tokens for a product from the same image,
    and negative samples are all other averaged global tokens from other devices.
    """

    def __init__(
        self,
        groups,
        num_samples=2,
        batch_size=8,
        world_size=1,
        tau=0.07,
        projection_input=768,
        projection_output=768,
        eps=1e-8,
        norm_type="l2",
        pixel_ratio=0.5,
        **kwargs,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.world_size = world_size
        self.groups = groups
        self.group_proj = {}
        for group_name, _group_members in self.groups.items():
            self.group_proj[group_name] = nn.Sequential(
                nn.LayerNorm(projection_input),
                nn.Linear(projection_input, int(4 * projection_input)),
                nn.GELU(),
                nn.Linear(int(4 * projection_input), projection_output),
            )
        self.group_proj = nn.ModuleDict(self.group_proj)

        self.eps = eps
        self.tau = tau
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
        pos_mask, neg_mask = get_pos_and_neg_mask(num_samples, batch_size, self.world_size)

        self.pixel_ratio = pixel_ratio
        self.norm_type = norm_type
        self.norm = None
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(projection_output)

        self.register_buffer("pos_mask", pos_mask, persistent=False)
        self.register_buffer("neg_mask", neg_mask, persistent=False)

    def random_paritioning_lc(
        self,
        group_features,
        partitioning,
        lc_classes,
        patch_size,
        group_name,
        dist_group=None,
    ):
        B, N, D = group_features.shape

        if partitioning is not None and partitioning.shape[1] > 0 and N >= self.num_samples * 2:
            # Generate averaged samples
            perm = torch.randperm(N, device=group_features.device)

            # array of indices to sample from [0, 0, 0, 1, 1, 1, ..., self.num_samples-1]
            sample_range = torch.arange(0, self.num_samples, self.num_samples / N, device=group_features.device).long()

            # Averaged patch samples
            group_pooled = torch.zeros((B, self.num_samples, D), device=group_features.device)
            group_pooled.index_reduce_(1, sample_range[perm], group_features, reduce="mean", include_self=False)
            group_pooled = group_pooled.reshape(-1, D)  # (B*num_samples, D)

            samples_per = torch.unique_consecutive(sample_range, return_counts=True)[1]  # (num_samples)

            # Average land cover histogram to N samples
            partitioning_pooled = torch.zeros(
                (B, self.num_samples, lc_classes), device=group_features.device, dtype=torch.float
            )
            partitioning_pooled.index_reduce_(
                1, sample_range[perm], partitioning.float(), reduce="mean", include_self=False
            )
            partitioning_pooled *= samples_per.unsqueeze(0).unsqueeze(-1)
            partitioning_pooled = partitioning_pooled.reshape(-1, lc_classes)  # (B*num_samples, lc_classes)
            partitioning_pooled_count = partitioning_pooled.sum(dim=-1, keepdim=True)

            # Average pixel count to N samples
            pixel_count_pooled = (samples_per.unsqueeze(0) * patch_size**2).expand(B, -1).reshape(-1)

            # Mask for our group
            soft_group_mask = partitioning_pooled_count.squeeze() > self.pixel_ratio * pixel_count_pooled

            group_mask = torch.ones((B * self.num_samples,), device=group_features.device, dtype=torch.bool)

            # Normalize land cover histogram
            partitioning_pooled = partitioning_pooled / partitioning_pooled_count.clamp(min=self.eps)

        else:
            # Create empty tensors, will be filtered out later
            group_pooled = torch.zeros(
                (B * self.num_samples, D), device=group_features.device, requires_grad=False, dtype=group_features.dtype
            )
            group_mask = torch.tensor([0] * B * self.num_samples, device=group_features.device, dtype=torch.bool)

            soft_group_mask = torch.tensor([0] * (B * self.num_samples), device=group_features.device, dtype=torch.bool)

            partitioning_pooled = torch.zeros(
                (B * self.num_samples, lc_classes),
                device=group_features.device,
                requires_grad=False,
                dtype=group_features.dtype,
            )

        # gather features from other GPUs
        all_group_features = group_pooled

        if dist.is_available() and dist.is_initialized():
            all_group_features = torch.cat(
                GatherLayer.apply(all_group_features, dist_group), dim=0
            )  # (B*num_samples*world_size, D)

            output_m = [torch.zeros_like(group_mask) for _ in range(dist.get_world_size(dist_group))]
            dist.all_gather(output_m, group_mask, group=dist_group)
            group_mask = torch.cat(output_m, dim=0)

            output_m = [torch.zeros_like(soft_group_mask) for _ in range(dist.get_world_size(dist_group))]
            dist.all_gather(output_m, soft_group_mask, group=dist_group)
            soft_group_mask = torch.cat(output_m, dim=0)

            output_p = [torch.zeros_like(partitioning_pooled) for _ in range(dist.get_world_size(dist_group))]
            dist.all_gather(output_p, partitioning_pooled, group=dist_group)
            partitioning_pooled = torch.cat(output_p, dim=0)

        soft_group_mask = soft_group_mask[group_mask == 1]

        # Filter out empty groups from partitioning
        partitioning_pooled = partitioning_pooled[group_mask == 1]

        # cosine similarity ranges from 0 to 1 for non negative count vectors
        soft_mask = F.cosine_similarity(partitioning_pooled.unsqueeze(1), partitioning_pooled.unsqueeze(0), dim=-1)

        soft_mask[~soft_group_mask, :][:, ~soft_group_mask] = 0.0  # Set soft mask to 0 for empty groups

        # Filter out empty groups from features
        all_group_features = all_group_features[group_mask == 1]

        # Filter out empty groups
        updated_pos_mask = self.pos_mask[group_mask == 1][:, group_mask == 1]
        updated_neg_mask = self.neg_mask[group_mask == 1][:, group_mask == 1]

        effective_world_size = (group_mask.sum() / (B * self.num_samples)).long().item()

        return all_group_features, soft_mask, updated_pos_mask, updated_neg_mask, effective_world_size

    def forward(
        self,
        features: dict[str, torch.FloatTensor],  # {"group[i]": (B, N, D), ...}
        partitioning: dict[str, torch.LongTensor],  # {"group[i]": (B, N), ...}
        lc_classes: dict[str, int],  # {"group[i]": num_classes, ...}
        patch_sizes: dict[str, int],  # {"group[i]": patch_size, ...}
        dist_group=None,
    ) -> tuple[dict[str, torch.FloatTensor], torch.FloatTensor]:
        losses = {}
        tot_loss = 0
        num_pos = 0
        debug_masks = {}

        rank = dist.get_rank(dist_group) if dist.is_available() and dist.is_initialized() else 0

        for group, group_features in features.items():
            B, _N, _D = group_features.shape
            if (
                group not in lc_classes
                or group not in patch_sizes
                or lc_classes[group] is None
                or patch_sizes[group] is None
            ):
                logger.debug(f"[rank: {rank}] Group: {group}, missing lc_classes or patch_sizes, skipping")
                continue
            all_group_features, soft_mask, updated_pos_mask, updated_neg_mask, effective_world_size = (
                self.random_paritioning_lc(
                    group_features,
                    partitioning.get(group, None),
                    lc_classes.get(group, None),
                    patch_size=patch_sizes.get(group, None),
                    group_name=group,
                    dist_group=dist_group,
                )
            )

            if all_group_features.numel() == 0:
                logger.debug(f"[rank: {rank}] Group: {group}, Global empty group!")
                continue

            # # Linear projection of group representations
            all_group_features = self.group_proj[group](all_group_features)

            if self.norm_type == "layernorm":
                all_group_features = self.norm(all_group_features)
            elif self.norm_type == "l2":
                # L2 normalize
                all_group_features = F.normalize(all_group_features, dim=-1, p=2)

            if updated_pos_mask.sum() == 0 or updated_neg_mask.sum() == 0:
                rank = dist.get_rank(dist_group) if dist.is_available() and dist.is_initialized() else 0
                logger.debug(f"Group: {group}, rank: {rank}, Empty group!!!")
                continue

            # dot product to get logits
            # (B*num_samples*world_size, B*num_samples*world_size)
            sim_mat = all_group_features @ all_group_features.t()

            temperature = self.logit_scale.exp()
            sim_mat = sim_mat * temperature

            sim_pos = sim_mat.masked_select(updated_pos_mask).view(
                B * self.num_samples * effective_world_size, self.num_samples - 1
            )
            sim_pos = sim_pos * soft_mask.masked_select(updated_pos_mask).view(
                B * self.num_samples * effective_world_size, self.num_samples - 1
            )

            sim_neg = sim_mat.masked_select(updated_neg_mask).view(B * self.num_samples * effective_world_size, -1)

            # Compute loss in log space for numerical stability
            # similar to https://lilianweng.github.io/posts/2021-05-31-contrastive/#soft-nearest-neighbors-loss
            log_sum_pos = torch.logsumexp(sim_pos, dim=-1)
            log_sum_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=-1), dim=-1)
            loss = -(log_sum_pos - log_sum_all).sum()
            num_pos += sim_pos.shape[0]

            debug_masks[group] = {
                "modulate": soft_mask,
                "pred": sim_mat,
                "target": updated_pos_mask,
                "sim_modulated": sim_mat * soft_mask,
            }

            tot_loss += loss
            losses[group] = loss / sim_pos.shape[0]

        if num_pos > 0:
            tot_loss /= num_pos

        return losses, tot_loss, self.logit_scale.item(), debug_masks


# DEBUG ##############################################################
def example(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42 + rank)

    gpus_per_node = 2
    node_rank = rank // gpus_per_node
    local_node_ranks = list(range(node_rank * gpus_per_node, (node_rank + 1) * gpus_per_node))

    # dist_group = dist.new_group(
    #    ranks=local_node_ranks,
    #    use_local_synchronization=True,
    # )
    dist_group = None
    local_world_size = dist.get_world_size(dist_group)
    world_size = dist.get_world_size()
    local_rank = dist.get_rank(dist_group)
    print(
        f"[rank: {rank}] Local rank {local_rank}, Node rank: {node_rank}, Local node ranks: {local_node_ranks}, Local world size: {local_world_size}, world size: {world_size}",
        flush=True,
    )

    ns = (10,)  # 20,
    patch_size = (4,)  # 2,
    B = 8
    N = sum(ns)
    D = 256
    num_samples = 2
    num_classes = 9

    # random input
    sample_input = {f"group{i}": torch.randn(B, ns[i], D).to(rank) for i in range(len(ns))}
    groups = {f"group{i}": None for i in range(len(ns))}
    lc_classes = {f"group{i}": num_classes for i in range(len(ns))}
    patch_sizes = {f"group{i}": patch_size[i] for i in range(len(ns))}

    sample_landcover = {
        f"group{i}": F.one_hot(torch.randint(0, num_classes, (B, ns[i], patch_size[i] ** 2)), num_classes).to(rank)
        for i in range(len(ns))
    }

    sample_partitioning = {f"group{i}": sample_landcover[f"group{i}"].sum(dim=-2) for i in range(len(ns))}

    for g in sample_partitioning.keys():
        sample_partitioning[g][:, :, 0] = 0
    if sample_input["group0"].numel() == 0:
        print(f"[rank: {rank}] Empty input")
    else:
        print(f"[rank: {rank}] Input: {sample_input['group0'].min()}, {sample_input['group0'].max()}")

    for i in range(len(ns)):
        print(
            f"[rank: {rank}] Group {i}, Input: {sample_input[f'group{i}'].shape}, "
            f"lc_classes: {lc_classes[f'group{i}']}, partition: {sample_partitioning[f'group{i}'].shape}, "
            f"landcover: {sample_landcover[f'group{i}'].shape},"
        )

    # for i in range(len(ns)):
    #     print(f"[rank: {rank}] Group {i}, Partitioning: {sample_partitioning[f'group{i}']}")

    class DummyModel(nn.Module):
        def __init__(self, i_channels, o_channels, group_contrast_loss, dist_group):
            super().__init__()
            self.group_contrast_loss = group_contrast_loss
            self.dist_group = dist_group
            self.layer = nn.Linear(i_channels, o_channels)
            self.layer.weight.data.fill_(1.0)
            self.layer.bias.data.fill_(0.0)

        def forward(self, x: dict[str, torch.FloatTensor]):
            x = {k: self.layer(v) for k, v in x.items()}
            return self.group_contrast_loss(
                x,
                partitioning=sample_partitioning,
                lc_classes=lc_classes,
                patch_sizes=patch_sizes,
                dist_group=self.dist_group,
            )

    constrastive_loss = LandCoverGroupContrastLossV2(
        groups=groups,
        num_samples=num_samples,
        batch_size=B,
        world_size=local_world_size,
        tau=0.07,
        eps=1e-8,
        projection_input=D,
        projection_output=D,
    ).to(rank)
    model = DummyModel(D, D, constrastive_loss, dist_group).to(rank)
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
    )

    losses, loss, logit_scale, masks = model(sample_input)

    if rank == 0:
        import matplotlib.pyplot as plt

        for group in masks:
            fig, axs = plt.subplots(1, 2, dpi=400)

            mask = masks[group]["target"].cpu().numpy()
            axs[0].imshow(mask)
            axs[0].set_title("mask")

            sim_mat = masks[group]["sim_modulated"].detach().cpu().numpy()
            axs[1].imshow(sim_mat)
            axs[1].set_title("Sim matrix")

            plt.savefig(f"mask_{group}.png")

    print(f"[rank: {rank}] Losses: {losses}, Total loss: {loss.item()}", flush=True)

    loss.backward()
    print(f"[rank: {rank}] Loss: {loss.item()}")

    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29511"
    main()
