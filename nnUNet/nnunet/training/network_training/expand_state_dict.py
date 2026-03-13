"""
expand_state_dict  -  Net2Net-style weight expansion for Generic_UNet.

Given a trained state_dict with `old_base` features, produce a new state_dict
whose tensors are compatible with a Generic_UNet built with `new_base` features.

Strategy: build two throwaway Generic_UNet instances (old_base and new_base),
compare tensor shapes key-by-key, and tile along any dimension that grew.
Tiled weights are scaled by 1/factor to preserve activation magnitudes.

For non-integer expansion (e.g. 128 -> 320 due to max_features clamping),
we repeat-and-truncate: tile enough full copies, slice to target size,
and scale by (old_size / new_size) to maintain the mean activation level.
"""

from collections import OrderedDict
import torch


def _tile_dim(tensor, dim, old_size, new_size):
    """
    Expand `tensor` along `dim` from `old_size` to `new_size`.

    If new_size is a clean multiple of old_size, this is pure tiling.
    Otherwise, we repeat enough copies and slice to target, then scale
    to preserve activation magnitude.
    """
    if old_size == new_size:
        return tensor

    repeats = (new_size + old_size - 1) // old_size  # ceil division
    # Build repeat spec: repeat 'repeats' times along 'dim'
    rep = [1] * tensor.ndim
    rep[dim] = repeats
    expanded = tensor.repeat(*rep)

    # Slice to exact target size
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, new_size)
    result = expanded[tuple(slices)]

    # Scale to preserve expected activation magnitude
    # Each output position is a copy of one of old_size original channels,
    # so we scale by old_size/new_size
    scale = old_size / new_size
    return result * scale


def expand_state_dict(old_sd, old_net, new_net):
    """
    Expand a Generic_UNet state_dict to match a wider network.

    Parameters
    ----------
    old_sd : dict
        Source state_dict (will NOT be mutated).
    old_net : Generic_UNet
        Network instance with the OLD (narrow) architecture.
        Only used to read .state_dict() shapes; not modified.
    new_net : Generic_UNet
        Network instance with the NEW (wide) architecture.
        Only used to read .state_dict() shapes; not modified.

    Returns
    -------
    OrderedDict - new state_dict loadable into new_net.
    """
    old_shapes = {k: v.shape for k, v in old_net.state_dict().items()}
    new_shapes = {k: v.shape for k, v in new_net.state_dict().items()}

    # Sanity: keys must match
    assert set(old_shapes.keys()) == set(new_shapes.keys()), (
        f"Key mismatch between old and new network.\n"
        f"  Only in old: {set(old_shapes) - set(new_shapes)}\n"
        f"  Only in new: {set(new_shapes) - set(old_shapes)}"
    )

    new_sd = OrderedDict()

    for key in old_shapes:
        old_shape = old_shapes[key]
        new_shape = new_shapes[key]
        param = old_sd[key].clone()

        if old_shape == new_shape:
            new_sd[key] = param
            continue

        # Expand along each dimension that changed
        for dim in range(len(old_shape)):
            if new_shape[dim] != old_shape[dim]:
                param = _tile_dim(param, dim, old_shape[dim], new_shape[dim])

        assert param.shape == new_shape, (
            f"Shape mismatch after expansion {key}: got {param.shape}, "
            f"expected {new_shape}"
        )
        new_sd[key] = param

    return new_sd
