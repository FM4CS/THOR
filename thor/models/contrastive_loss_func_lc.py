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


def latlon_similarity_matrix(coords_degrees):
    """
    Calculates a similarity matrix for N (lat, lon) coordinates.
    Similarity is 1 for identical points, 0 for antipodal points.

    Args:
        coords_degrees (torch.Tensor): A tensor of shape [N, 2]
                                       with (latitude, longitude) in degrees.

    Returns:
        torch.Tensor: An [N, N] similarity matrix.
    """
    N = coords_degrees.shape[0]
    if N == 0:
        return torch.empty((0, 0), dtype=coords_degrees.dtype, device=coords_degrees.device)

    # Convert degrees to radians
    coords_rad = torch.deg2rad(coords_degrees)  # Shape: [N, 2]

    # Extract latitudes and longitudes
    lat_rad = coords_rad[:, 0]  # Shape: [N]
    lon_rad = coords_rad[:, 1]  # Shape: [N]

    # Expand for broadcasting:
    # lat1, lon1 will be column vectors [N, 1]
    # lat2, lon2 will be row vectors [1, N]
    lat1 = lat_rad.unsqueeze(1)  # Shape: [N, 1]
    lon1 = lon_rad.unsqueeze(1)  # Shape: [N, 1]
    lat2 = lat_rad.unsqueeze(0)  # Shape: [1, N]
    lon2 = lon_rad.unsqueeze(0)  # Shape: [1, N]

    # Calculate differences
    delta_lat = lat1 - lat2  # Shape: [N, N]
    delta_lon = lon1 - lon2  # Shape: [N, N]

    # Haversine formula components
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # Δφ = delta_lat, Δλ = delta_lon
    # φ1 = lat1, φ2 = lat2
    a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2

    # Central angle c = 2 ⋅ atan2( √a, √(1−a) )
    # Clamp 'a' to [0, 1] to avoid numerical issues with sqrt
    # (though 'a' should theoretically be in [0,1] as it's hav(central_angle))
    a_clamped = torch.clamp(a, 0.0, 1.0)
    central_angle = 2 * torch.atan2(torch.sqrt(a_clamped), torch.sqrt(1 - a_clamped))  # Shape: [N, N]
    # central_angle is in radians, ranging from 0 to pi

    # Normalize angular distance to be [0, 1]
    # Max angular distance is pi (antipodal points)
    normalized_angular_distance = central_angle / torch.pi

    # Similarity: 1 for identical (dist=0), 0 for antipodal (dist=1)
    similarity = 1.0 - normalized_angular_distance

    return similarity


def month_similarity_matrix(months):
    if months.numel() == 0:
        return torch.empty((0, 0), dtype=months.dtype, device=months.device)
    # Expand to a column vector (N, 1)
    col_months = months.unsqueeze(1)

    # Expand to a row vector (1, N)
    row_months = months.unsqueeze(0)

    # Calculate absolute difference
    abs_diff_matrix = torch.abs(col_months - row_months)

    # Calculate shortest distance considering cyclical nature (0 to 6)
    # For example, diff(Jan, Dec) = min(abs(0-11), 12-abs(0-11)) = min(11, 1) = 1
    cyclic_distance = torch.minimum(abs_diff_matrix, 12 - abs_diff_matrix)

    # Normalize the distance (0 to 1), max distance is 6
    normalized_distance = cyclic_distance / 6.0

    # Similarity score: 1 - normalized_distance
    similarity_matrix = 1.0 - normalized_distance

    return similarity_matrix


class LandCoverGroupContrastLoss(nn.Module):
    """
    Contrastive loss per group of products.

    Positive samples are averaged global tokens for a product from the same image,
    and negative samples are all other averaged global tokens from other devices.
    """

    def __init__(
        self,
        groups,
        num_samples=2,
        tau=0.07,
        projection_input=768,
        projection_output=768,
        eps=1e-8,
        smoothing=0.1,
        modulate=True,
        norm_type="l2",
        pixel_ratio=0.5,
        **kwargs,
    ):
        super().__init__()

        self.num_samples = num_samples
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
        self.smoothing = smoothing
        self.modulate = modulate
        self.pixel_ratio = pixel_ratio
        self.norm_type = norm_type
        self.norm = None
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(projection_output)

    def random_paritioning_lc(
        self,
        group_features,
        partitioning,
        lc_classes,
        patch_size,
        group_name,
        latlon_batch=None,
        month_batch=None,
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
            group_mask = partitioning_pooled_count.squeeze() > self.pixel_ratio * pixel_count_pooled

            # Normalize land cover histogram
            partitioning_pooled = partitioning_pooled / partitioning_pooled_count.clamp(min=self.eps)

        else:
            # Create empty tensors, will be filtered out later
            group_pooled = torch.zeros(
                (B * self.num_samples, D), device=group_features.device, requires_grad=False, dtype=group_features.dtype
            )
            group_mask = torch.tensor([0] * B * self.num_samples, device=group_features.device, dtype=torch.bool)

            partitioning_pooled = torch.zeros(
                (B * self.num_samples, lc_classes),
                device=group_features.device,
                requires_grad=False,
                dtype=group_features.dtype,
            )

        # gather features from other GPUs
        all_group_features = group_pooled
        all_latlon_pooled = None
        all_month_pooled = None

        # Create lat/lon and month data if provided
        if latlon_batch is not None:
            # Expand (B, 2) to (B*num_samples, 2)
            all_latlon_pooled = (
                latlon_batch.unsqueeze(1)
                .expand(-1, self.num_samples, 2)
                .reshape(B * self.num_samples, 2)
                .to(group_features.device, non_blocking=True)
            )
        else:
            all_latlon_pooled = torch.zeros(
                (B * self.num_samples, 2),
                device=group_features.device,
                requires_grad=False,
                dtype=group_features.dtype,
            )

        if month_batch is not None:
            # Expand (B) to (B*num_samples)
            all_month_pooled = (
                month_batch.expand(-1, self.num_samples)
                .reshape(B * self.num_samples)
                .to(group_features.device, non_blocking=True)
            )
        else:
            all_month_pooled = torch.zeros(
                (B * self.num_samples),
                device=group_features.device,
                requires_grad=False,
                dtype=group_features.dtype,
            )

        if dist.is_available() and dist.is_initialized():
            all_group_features = torch.cat(
                GatherLayer.apply(all_group_features, dist_group), dim=0
            )  # (B*num_samples*world_size, D)

            output_m = [torch.zeros_like(group_mask) for _ in range(dist.get_world_size(dist_group))]
            dist.all_gather(output_m, group_mask, group=dist_group)
            group_mask = torch.cat(output_m, dim=0)

            output_p = [torch.zeros_like(partitioning_pooled) for _ in range(dist.get_world_size(dist_group))]
            dist.all_gather(output_p, partitioning_pooled, group=dist_group)
            partitioning_pooled = torch.cat(output_p, dim=0)

            all_latlon_pooled = torch.cat(GatherLayer.apply(all_latlon_pooled, dist_group), dim=0)
            all_month_pooled = torch.cat(GatherLayer.apply(all_month_pooled, dist_group), dim=0)

        # Filter out empty groups from partitioning
        partitioning_pooled = partitioning_pooled[group_mask == 1]

        # cosine similarity ranges from 0 to 1 for non negative count vectors
        soft_mask = F.cosine_similarity(partitioning_pooled.unsqueeze(1), partitioning_pooled.unsqueeze(0), dim=-1)

        # Filter out empty groups from features
        all_group_features = all_group_features[group_mask == 1]

        # Calculate modulation factor based on lat/lon and month similarity
        modulation_factor = torch.ones_like(soft_mask)  # Default to 1 (no modulation)

        if all_group_features.shape[0] > 0:  # If we have valid samples
            filtered_latlon = all_latlon_pooled[group_mask == 1]

            filtered_month = all_month_pooled[group_mask == 1]

            latlon_sim = latlon_similarity_matrix(filtered_latlon)

            month_sim = month_similarity_matrix(filtered_month)

            combined_sim = (latlon_sim + month_sim) / 2.0

            modulation_factor = 1.0 - combined_sim

        return all_group_features, soft_mask, modulation_factor

    def forward(
        self,
        features: dict[str, torch.FloatTensor],  # {"group[i]": (B, N, D), ...}
        partitioning: dict[str, torch.LongTensor],  # {"group[i]": (B, N), ...}
        lc_classes: dict[str, int],  # {"group[i]": num_classes, ...}
        patch_sizes: dict[str, int],  # {"group[i]": patch_size, ...}
        latlon=None,
        month=None,
        dist_group=None,
    ) -> tuple[dict[str, torch.FloatTensor], torch.FloatTensor]:
        losses = {}
        tot_loss = 0
        num_pos = 0
        debug_masks = {}

        rank = dist.get_rank(dist_group) if dist.is_available() and dist.is_initialized() else 0

        for group, group_features in features.items():
            if (
                group not in lc_classes
                or group not in patch_sizes
                or lc_classes[group] is None
                or patch_sizes[group] is None
            ):
                logger.debug(f"[rank: {rank}] Group: {group}, missing lc_classes or patch_sizes, skipping")
                continue
            all_group_features, soft_mask, modulation_factor = self.random_paritioning_lc(
                group_features,
                partitioning.get(group, None),
                lc_classes[group],
                patch_size=patch_sizes.get(group, None),
                group_name=group,
                latlon_batch=latlon,
                month_batch=month,
                dist_group=dist_group,
            )

            if all_group_features.numel() == 0:
                logger.debug(f"[rank: {rank}] Group: {group}, Global empty group!")
                continue

            # # Linear projection of group representations
            all_group_features = self.group_proj[group.split(":")[0]](all_group_features)

            if self.norm_type == "layernorm":
                all_group_features = self.norm(all_group_features)
            elif self.norm_type == "l2":
                # L2 normalize
                all_group_features = F.normalize(all_group_features, dim=-1, p=2)

            # dot product to get logits
            # (B*num_samples*world_size, B*num_samples*world_size)
            sim_mat = all_group_features @ all_group_features.t()
            sim_mat = sim_mat * self.logit_scale

            # Apply label smoothing to targets with modulation
            if self.modulate:
                effective_smoothing = self.smoothing * modulation_factor
            else:
                effective_smoothing = self.smoothing
            smoothed_targets = soft_mask * (1.0 - effective_smoothing) + (1.0 - soft_mask) * effective_smoothing

            # lets make the diagonal 1 again
            smoothed_targets.fill_diagonal_(1.0)

            loss = F.binary_cross_entropy_with_logits(sim_mat, smoothed_targets, reduction="sum")

            debug_masks[group] = {
                "target": soft_mask,
                "pred": F.sigmoid(sim_mat),
                "modulate": modulation_factor,
            }

            tot_loss += loss
            n = soft_mask.numel()
            if n > 0:
                losses[group] = loss / n
            num_pos += n

        if num_pos > 0:
            tot_loss /= num_pos

        return losses, tot_loss, self.logit_scale.item(), debug_masks


# DEBUG ##############################################################
def example(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

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
    # N = sum(ns)
    D = 256
    num_samples = 2
    num_classes = 9

    # random input
    sample_input = {f"group{i}": torch.randn(B, ns[i], D).to(rank) for i in range(len(ns))}
    groups = {f"group{i}": None for i in range(len(ns))}
    lc_classes = {f"group{i}": num_classes for i in range(len(ns))}
    patch_sizes = {f"group{i}": patch_size[i] for i in range(len(ns))}

    # sample_landcover = {
    #     f"group{i}": F.one_hot(torch.randint(0, num_classes, (B, ns[i], patch_size[i])), num_classes).to(rank)
    #     for i in range(len(ns))
    # }
    if rank == 0:
        sample_landcover = {
            f"group{i}": F.one_hot(
                torch.randint(low=4, high=num_classes - 1, size=(B, ns[i], patch_size[i] ** 2), dtype=torch.long),
                num_classes,
            ).to(rank)
            for i in range(len(ns))
        }
    elif rank == 1:
        sample_landcover = {
            f"group{i}": F.one_hot(
                torch.randint(low=0, high=3, size=(B, ns[i], patch_size[i] ** 2), dtype=torch.long),
                num_classes,
            ).to(rank)
            for i in range(len(ns))
        }

    sample_partitioning = {f"group{i}": sample_landcover[f"group{i}"].sum(dim=-2) for i in range(len(ns))}

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

    constrastive_loss = LandCoverGroupContrastLoss(
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

    losses, loss, _logit_scale, masks = model(sample_input)

    if rank == 0:
        import matplotlib.pyplot as plt

        for group in masks:
            _fig, axs = plt.subplots(1, 2, dpi=400)

            target_mask = masks[group]["target"].cpu().numpy()
            axs[0].imshow(target_mask[:, :])
            axs[0].set_title("target mask")

            pred_mask = masks[group]["pred"].detach().cpu().numpy()
            axs[1].imshow(pred_mask)
            axs[1].set_title("pred mask")

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
