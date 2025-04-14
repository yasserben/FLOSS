# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .typing_utils import SampleList


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value

    return outputs


def stack_batch(
    inputs: List[torch.Tensor],
    data_samples: Optional[SampleList] = None,
    size: Optional[tuple] = None,
    size_divisor: Optional[int] = None,
    pad_val: Union[int, float] = 0,
    seg_pad_val: Union[int, float] = 255,
) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(
        inputs, list
    ), f"Expected input type to be list, but got {type(inputs)}"
    assert len({tensor.ndim for tensor in inputs}) == 1, (
        f"Expected the dimensions of all inputs must be the same, "
        f"but got {[tensor.ndim for tensor in inputs]}"
    )
    assert inputs[0].ndim == 3, (
        f"Expected tensor dimension to be 3, " f"but got {inputs[0].ndim}"
    )
    assert len({tensor.shape[0] for tensor in inputs}) == 1, (
        f"Expected the channels of all inputs must be the same, "
        f"but got {[tensor.shape[0] for tensor in inputs]}"
    )

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (
        size_divisor is not None
    ), "only one of size and size_divisor should be valid"

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if "gt_sem_seg" in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val
                )
                pad_shape = data_sample.gt_sem_seg.shape
            if "tube_tensor" in data_sample:
                tube_tensor = data_sample.tube_tensor.data
                del data_sample.tube_tensor.data
                data_sample.tube_tensor.data = F.pad(tube_tensor, padding_size, value=0)
                pad_shape = data_sample.tube_tensor.shape
            if "gt_edge_map" in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val
                )
                pad_shape = data_sample.gt_edge_map.shape
            if "gt_depth_map" in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val
                )
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo(
                {
                    "img_shape": tensor.shape[-2:],
                    "pad_shape": pad_shape,
                    "padding_size": padding_size,
                }
            )
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(img_padding_size=padding_size, pad_shape=pad_img.shape[-2:])
            )

    return torch.stack(padded_inputs, dim=0), padded_samples


@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def downscale_label_ratio(gt, scale_factor, min_ratio, n_classes, ignore_index=255):
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out
