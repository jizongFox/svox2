from typing import Tuple

import torch
from torch import autograd

from . import utils
from .defs import BASIS_TYPE_MLP, BASIS_TYPE_3D_TEXTURE

_C = utils.get_c_extension()

assert _C is not None, _C


class _SampleGridAutogradFunction(autograd.Function):
    @staticmethod
    def forward(
            ctx=None,
            data_density: torch.Tensor = None,
            data_sh: torch.Tensor = None,
            grid=None,
            points: torch.Tensor = None,
            want_colors: bool = None,
    ):
        assert not points.requires_grad, "Point gradient not supported"
        out_density, out_sh = _C.sample_grid(grid, points, want_colors)
        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        return out_density, out_sh

    @staticmethod
    def backward(ctx, grad_out_density=None, grad_out_sh=None):
        (points,) = ctx.saved_tensors
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        _C.sample_grid_backward(
            ctx.grid,
            points,
            grad_out_density.contiguous(),
            grad_out_sh.contiguous(),
            grad_density_grid,
            grad_sh_grid,
            ctx.want_colors,
        )
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None

        return grad_density_grid, grad_sh_grid, None, None, None


class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(
            ctx=None,
            data_density: torch.Tensor = None,
            data_sh: torch.Tensor = None,
            data_basis: torch.Tensor = None,
            data_background: torch.Tensor = None,
            grid=None,
            rays=None,
            opt=None,
            backend: str = None,
    ):
        cu_fn = _C.__dict__[f"volume_render_{backend}"]
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend
        ctx.basis_data = data_basis
        return color

    @staticmethod
    def backward(ctx=None, grad_out=None):
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        if ctx.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = torch.zeros_like(ctx.basis_data)
        elif ctx.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_sh_out = grad_sh_grid
        if ctx.needs_input_grad[2]:
            grad_holder.grad_basis_out = grad_basis
        if ctx.grid.background_data is not None and ctx.needs_input_grad[3]:
            grad_holder.grad_background_out = grad_background
        cu_fn(
            ctx.grid, ctx.rays, ctx.opt, grad_out.contiguous(), color_cache, grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None
        if not ctx.needs_input_grad[3]:
            grad_background = None
        ctx.basis_data = None

        return (
            grad_density_grid,
            grad_sh_grid,
            grad_basis,
            grad_background,
            None,
            None,
            None,
            None,
        )


class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(
            ctx=None,
            data: torch.Tensor = None,
            links: torch.Tensor = None,
            start_dim: int = None,
            end_dim: int = None,
            use_logalpha: bool = None,
            logalpha_delta: float = None,
            ignore_edge: bool = None,
            ndc_coeffs: Tuple[float, float] = None,
    ):
        tv = _C.tv(
            links,
            data,
            start_dim,
            end_dim,
            use_logalpha,
            logalpha_delta,
            ignore_edge,
            ndc_coeffs[0],
            ndc_coeffs[1],
        )
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_logalpha = use_logalpha
        ctx.logalpha_delta = logalpha_delta
        ctx.ignore_edge = ignore_edge
        ctx.ndc_coeffs = ndc_coeffs
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(
            links,
            data,
            ctx.start_dim,
            ctx.end_dim,
            1.0,
            ctx.use_logalpha,
            ctx.logalpha_delta,
            ctx.ignore_edge,
            ctx.ndc_coeffs[0],
            ctx.ndc_coeffs[1],
            grad_grid,
        )
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None, None, None, None, None
