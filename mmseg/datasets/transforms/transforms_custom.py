import cv2
import mmcv
import numpy as np
import torch.nn.functional as F
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DeterministicPhotoMetricDistortion(BaseTransform):
    """Apply deterministic photometric distortion to image.
    Unlike PhotoMetricDistortion, this class applies transformations in a fixed order
    with deterministic parameters.

    Required Keys:
    - img

    Modified Keys:
    - img

    Args:
        brightness_shift (float): The shift value for brightness adjustment.
            Positive values increase brightness, negative values decrease it.
        contrast_factor (float): The contrast adjustment factor.
            Values > 1 increase contrast, values < 1 decrease it.
        saturation_factor (float): The saturation adjustment factor.
            Values > 1 increase saturation, values < 1 decrease it.
        hue_shift (int): The hue shift value in degrees (-180 to 180).
        order (tuple): Order of operations. Default is ('brightness', 'contrast',
            'saturation', 'hue'). Each operation can appear at most once.
        activated (bool): Whether to apply the transformations or not.
            If False, the image will be returned unchanged. Default: True.
    """

    def __init__(
        self,
        brightness_shift: float = 0.0,
        contrast_factor: float = 1.0,
        saturation_factor: float = 1.0,
        hue_shift: int = 0,
        order: tuple = ("brightness", "contrast", "saturation", "hue"),
        activated: bool = False,
    ):
        super().__init__()
        assert -255.0 <= brightness_shift <= 255.0
        assert contrast_factor > 0
        assert saturation_factor > 0
        assert -180 <= hue_shift <= 180
        assert all(
            op in ["brightness", "contrast", "saturation", "hue"] for op in order
        )
        assert len(set(order)) == len(order), "Each operation can only appear once"

        self.brightness_shift = brightness_shift
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_shift = hue_shift
        self.order = order
        self.activated = activated

    def convert(
        self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0
    ) -> np.ndarray:
        """Multiple with alpha and add beta with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness shift."""
        return self.convert(img, beta=self.brightness_shift)

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply contrast adjustment."""
        return self.convert(img, alpha=self.contrast_factor)

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Apply saturation adjustment."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 1] = self.convert(img[:, :, 1], alpha=self.saturation_factor)
        img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Apply hue shift."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + self.hue_shift) % 180
        img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Apply the transformations in the specified order if activated.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images transformed if activated,
                 otherwise returns the original image.
        """
        img = results["img"]

        if self.activated:
            # Apply transformations in the specified order
            for operation in self.order:
                transform_func = getattr(self, operation)
                img = transform_func(img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_shift={self.brightness_shift}, "
            f"contrast_factor={self.contrast_factor}, "
            f"saturation_factor={self.saturation_factor}, "
            f"hue_shift={self.hue_shift}, "
            f"order={self.order}, "
            f"activated={self.activated})"
        )
        return repr_str
