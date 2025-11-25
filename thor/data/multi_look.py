import torch
from torch import nn


class MulitLook(nn.Module):
    def __init__(
        self,
        gsd: int,
        multilook_factors: int | list[int] = 2,
    ) -> None:
        super().__init__()
        self.gdf = gsd
        if isinstance(multilook_factors, int):
            multilook_factors = [multilook_factors]
        self.multilooked_gsd = [gsd // factor for factor in multilook_factors]
        proj_dict = {}
        for factor in multilook_factors:
            proj_dict[f"factor_{factor}"] = nn.Conv2d(
                2, 2, kernel_size=factor, stride=factor, padding=0, bias=False, groups=2
            )
            proj_dict[f"factor_{factor}"].weight.requires_grad = False

        self.patch_embed = nn.ModuleDict(proj_dict)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for _, p_embed in self.patch_embed.items():
            w = p_embed.weight.data
            s = w.shape
            w.fill_(1.0 / (s[-1] * s[-2]))

    def forward(
        self, x: torch.Tensor, multilook_gsd: int | None = None, mulitlook_factor: int | None = None, pad: bool = True
    ) -> torch.Tensor:
        if multilook_gsd is not None:
            mulitlook_factor = self.gdf // multilook_gsd
        if mulitlook_factor is None or mulitlook_factor == 1:
            return x

        if f"factor_{mulitlook_factor}" not in self.patch_embed:
            msg = f"mulitlook_factor {mulitlook_factor} not supported, only support {list(self.patch_embed.keys())}"
            raise ValueError(msg)
        if not pad and x.shape[-1] % mulitlook_factor != 0:
            msg = f"input shape {x.shape} not divisible by mulitlook_factor {mulitlook_factor}"
            raise ValueError(msg)

        return self.patch_embed[f"factor_{mulitlook_factor}"](x)


if __name__ == "__main__":
    gsd = 10
    multilook_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    multi_look = MulitLook(gsd, multilook_factors)

    x = torch.randn(1, 1, 100, 100)
    for factor in multilook_factors:
        try:
            print(multi_look(x, mulitlook_factor=factor, pad=False).shape)
        except Exception as e:
            print(e)

    print(multi_look(x, multilook_gsd=10).shape)
