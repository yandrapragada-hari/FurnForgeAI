from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchmcubes import marching_cubes as _torch_marching_cubes
except ImportError:
    _torch_marching_cubes = None


def _skimage_marching_cubes(level_tensor: torch.Tensor, threshold: float = 0.0):
    from skimage.measure import marching_cubes as sk_marching_cubes

    vol = level_tensor.detach().cpu().numpy()
    verts, faces, _normals, _values = sk_marching_cubes(
        vol, level=float(threshold), gradient_direction="ascent"
    )
    v_pos = torch.from_numpy(np.ascontiguousarray(verts, dtype=np.float32))
    t_pos_idx = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64))
    return v_pos, t_pos_idx


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _torch_marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        if self.mc_func is not None:
            try:
                v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
            except AttributeError:
                v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
            v_pos = v_pos[..., [2, 1, 0]]
        else:
            v_pos, t_pos_idx = _skimage_marching_cubes(level, 0.0)
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
