"""Pure spatial / spectral operations.

This package depends only on numpy and torch — never on `data/`, `models/`,
or `engine/`. Anything physics-related (cropping, periodicity, FFT
spectra, CIC density deposit) lives here so it can be unit-tested in
isolation and reused by both the dataloader and the evaluator.
"""

from .density import disp_to_density
from .geometry import (
    apply_periodic_bc,
    edge_buffer_mask,
    inner_crop,
    minimum_image,
    outside_mask_for_crop,
    overlap_crop_starts,
    point_in_inner_mask,
    points_to_voxel,
    sample_lagrangian_points,
    trilinear_devoxelize,
    voxel_to_points_indices,
)
from .spectrum import coherence, cross_power, power_spectrum, transfer_function

__all__ = [
    "apply_periodic_bc",
    "edge_buffer_mask",
    "inner_crop",
    "minimum_image",
    "outside_mask_for_crop",
    "overlap_crop_starts",
    "point_in_inner_mask",
    "points_to_voxel",
    "sample_lagrangian_points",
    "trilinear_devoxelize",
    "voxel_to_points_indices",
    "disp_to_density",
    "power_spectrum",
    "cross_power",
    "transfer_function",
    "coherence",
]
