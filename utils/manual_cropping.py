import copy
from typing import Dict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import Transform
from monai.transforms import InvertibleTransform, MapTransform, TraceableTransform
from monai.utils import TraceKeys
from scipy.special import expit


def box_filter_size(s):
    size = int(np.floor(s * 3 * np.sqrt(2 * np.pi) / 4 + 0.5))
    if size % 2 == 1:
        size_list = [size] * 3
        anchor_list = [size // 2] * 3
    else:
        size_list = [size] * 2 + [size + 1]
        anchor_list = [max(size // 2 - 1, 0), size // 2] + [(size + 1) // 2]
    return list(zip(size_list, anchor_list))


def approx_gaussian_blur(image, width, height):
    width_list = box_filter_size(width)
    height_list = box_filter_size(height)

    for w, h in zip(width_list, height_list):
        if 0 <= w[1] < w[0] and 0 <= h[1] < h[0]:
            image = cv.blur(image, ksize=(w[0], h[0]), anchor=(w[1], h[1]))
    return image


def louis_window(dcm_array):
    return np.clip(dcm_array + 70.0 / 255.0, 0.0, 1.0)


def sigmoid_window(dcm):
    return expit(((dcm * 255.0 + 29.0) / 179.0 - 0.5) * 4)


def binarize(img):
    _, bin_img = cv.threshold(img, 0, 1, cv.THRESH_BINARY)
    return bin_img


def find_contour(img, sigma):
    blur = cv.GaussianBlur(img, (sigma, sigma), 0)
    filtered = binarize(blur)
    contours, hierarchy = cv.findContours(
        filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv.contourArea)


def curvature(contour):
    area = cv.contourArea(contour) / np.pi
    perimeter = cv.arcLength(contour, True) / (2 * np.pi)
    return area / perimeter**2


def find_optimal_contour_fast(img, mins=5, maxs=25, threshold=0.5):
    maxc, maxcontour = float("-inf"), None
    for sigma in range(mins, maxs, 2):
        contour = find_contour(img, sigma)
        c = curvature(contour)
        if c > threshold:
            return contour
        elif c > maxc:
            maxc = c
            maxcontour = contour
    return maxcontour


def make_malo_crop_bounding_box(dcm_array, pixel_width, pixel_height):
    dcm_array = binarize(dcm_array)
    kwidth, kheight = 5.0 / np.array([pixel_width, pixel_height])
    dcm_array = approx_gaussian_blur(dcm_array, kwidth, kheight)
    contours, _ = cv.findContours(
        np.round(dcm_array * 255.0).astype("uint8"),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv.contourArea)
    return cv.boundingRect(contour)


def crop_and_resize(
    img,
    pixel_width,
    pixel_height,
    square_pixel=True,
    keep_aspect_ratio=True,
    target_size=512,
):
    bounding_box = make_malo_crop_bounding_box(img, pixel_width, pixel_height)
    bounding_box = np.array(bounding_box).astype(np.float64)
    anchor = bounding_box[:2]
    size = bounding_box[2:]
    if square_pixel:
        input_scale = np.diag([pixel_width, pixel_height])
    else:
        input_scale = np.eye(2)
    input_scale = input_scale.astype(np.float64)
    if keep_aspect_ratio:
        target_size_scale = np.eye(2, dtype=np.float64) * np.min(
            target_size / input_scale.dot(size)
        )
    else:
        target_size_scale = np.diag(target_size / input_scale.dot(size))
    scale = target_size_scale.dot(input_scale)
    bias = 0.5 * (target_size - scale.dot(size)) - scale.dot(anchor)
    transform = np.round(
        np.concatenate([scale, np.expand_dims(bias, 1)], axis=1), decimals=6
    ).astype(np.float32)
    return transform


def make_crop_and_resize_transform(
    dcm_array,
    pixel_width,
    pixel_height,
    target_size=512,
    keep_aspect_ratio=True,
    square_pixel=True,
):
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_size = np.array(target_size).astype(np.float64)

    bounding_box = make_malo_crop_bounding_box(dcm_array, pixel_width, pixel_height)

    bounding_box = np.array(bounding_box).astype(np.float64)
    anchor = bounding_box[:2]
    size = bounding_box[2:]

    if square_pixel:
        input_scale = np.diag([pixel_width, pixel_height])
    else:
        input_scale = np.eye(2)
    input_scale = input_scale.astype(np.float64)
    if keep_aspect_ratio:
        target_size_scale = np.eye(2, dtype=np.float64) * np.min(
            target_size / input_scale.dot(size)
        )
    else:
        target_size_scale = np.diag(target_size / input_scale.dot(size))
    scale = target_size_scale.dot(input_scale)
    bias = 0.5 * (target_size - scale.dot(size)) - scale.dot(anchor)
    transform = np.round(
        np.concatenate([scale, np.expand_dims(bias, 1)], axis=1), decimals=6
    )
    return transform.astype(np.float32)


class Cropd(MapTransform, InvertibleTransform, TraceableTransform):
    """
    Dictionary-based transform that detects the roi, crops around it and resizes the image to desired shape.
    """

    def __init__(
        self,
        keys,
        target_size=512,
        keep_aspect_ratio=True,
        square_pixel=True,
        allow_missing_keys=False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            target_size: expected shape of spatial dimensions after resize operation.
                If some components of the spatial_size are non-positive values, the transform will use the corresponding components of img size.
                For example, spatial_size=(32, -1) will be adapted to (32, 64) if the second spatial dimension size of img is 64.
            keep_aspect_ratio: whether or not to keep the ration between pixels dimensions.
            square_pixel: whether or not to transform pixels into square pixels if they are not already square.
        """
        super().__init__(keys, allow_missing_keys)
        self.target_size = np.array((target_size, target_size)).astype(np.float64)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.square_pixel = square_pixel

    def crop_and_resize(self, img, pixel_width, pixel_height):
        bounding_box = make_malo_crop_bounding_box(img, pixel_width, pixel_height)
        bounding_box = np.array(bounding_box).astype(np.float64)
        anchor = bounding_box[:2]
        size = bounding_box[2:]
        if self.square_pixel:
            input_scale = np.diag([pixel_width, pixel_height])
        else:
            input_scale = np.eye(2)
        input_scale = input_scale.astype(np.float64)
        if self.keep_aspect_ratio:
            target_size_scale = np.eye(2, dtype=np.float64) * np.min(
                self.target_size / input_scale.dot(size)
            )
        else:
            target_size_scale = np.diag(self.target_size / input_scale.dot(size))
        scale = target_size_scale.dot(input_scale)
        bias = 0.5 * (self.target_size - scale.dot(size)) - scale.dot(anchor)
        transform = np.round(
            np.concatenate([scale, np.expand_dims(bias, 1)], axis=1), decimals=6
        ).astype(np.float32)
        return transform

    def __call__(self, data) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(
                d,
                key,
            )
            pixdim = data["input_dict"]["Pixel Spacing"]
            crop = self.crop_and_resize(
                d["input"], pixel_width=pixdim[0], pixel_height=pixdim[1]
            )
            d["input"] = cv.warpAffine(
                d["input"],
                M=crop,
                dsize=(512, 512),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=-1000.0,
            )
            d["label"] = cv.warpAffine(
                d["label"],
                M=crop,
                dsize=(512, 512),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=0.0,
            )
        return d

    def inverse(self, data) -> Dict:
        d = copy.deepcopy(dict(data))
        print(data)
        transforms = data["label_transforms"]
        pixdim = data["input_dict"]["Pixel Spacing"]
        for transform in transforms:
            if transform["class"] == "Cropd":
                orig_size = transform["orig_size"][0]

        for key in self.key_iterator(d):
            transform = self.crop_and_resize(
                d[key], pixel_width=pixdim[0], pixel_height=pixdim[1]
            )
            untransform = cv.invertAffineTransform(transform)
            d[key] = cv.warpAffine(
                d[key],
                M=untransform,
                dsize=(orig_size, orig_size),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=-1000.0,
            )
            self.pop_transform(d, key)
        return d


class Crop_3Dd(MapTransform, InvertibleTransform, TraceableTransform):
    """
    3D-wrapper of Cropd. Allows to crop 3 slices at a time and choose the best contour that will fit all three.
    """

    def __init__(
        self,
        keys,
        target_size=512,
        keep_aspect_ratio=True,
        square_pixel=True,
        allow_missing_keys=False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_size = np.array((target_size, target_size)).astype(np.float64)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.square_pixel = square_pixel

    def crop_and_resize(self, img, pixel_width, pixel_height):
        bounding_box = make_malo_crop_bounding_box(img, pixel_width, pixel_height)
        bounding_box = np.array(bounding_box).astype(np.float64)
        anchor = bounding_box[:2]
        size = bounding_box[2:]
        if self.square_pixel:
            input_scale = np.diag([pixel_width, pixel_height])
        else:
            input_scale = np.eye(2)
        input_scale = input_scale.astype(np.float64)
        if self.keep_aspect_ratio:
            target_size_scale = np.eye(2, dtype=np.float64) * np.min(
                self.target_size / input_scale.dot(size)
            )
        else:
            target_size_scale = np.diag(self.target_size / input_scale.dot(size))
        scale = target_size_scale.dot(input_scale)
        bias = 0.5 * (self.target_size - scale.dot(size)) - scale.dot(anchor)
        transform = np.round(
            np.concatenate([scale, np.expand_dims(bias, 1)], axis=1), decimals=6
        ).astype(np.float32)
        return transform

    def multi_channel_crop(self, data, pixel_width, pixel_height):
        """
        This function compares the sizes of the crop windows for each channel and returns the matric to use for the crop which is the larger window among all channels
        """
        number_channels = 1 if len(data.shape) == 2 else data.shape[0]
        size = [0, 0]
        max_channel = -1
        for channel in range(number_channels):
            bounding_box = make_malo_crop_bounding_box(
                (torch.unbind(torch.tensor(data))[channel]).numpy(),
                pixel_width=pixel_width,
                pixel_height=pixel_height,
            )
            bounding_box = np.array(bounding_box).astype(np.float64)
            if size[0] < bounding_box[2:][0]:
                size = bounding_box[2:]
                max_channel = channel
        transform = self.crop_and_resize(
            img=(torch.unbind(torch.tensor(data))[max_channel]).numpy(),
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
        return transform

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(
                d,
                key,
            )
            pixdim = data["input_dict"]["Pixel Spacing"]
            crop = self.multi_channel_crop(
                d["input"], pixel_width=pixdim[0], pixel_height=pixdim[1]
            )
            number_channels = data[key].shape[
                0
            ]  # admitted the input has the shape: C x W w H

            inputs_3d = [
                np.squeeze(d["input"][channel]) for channel in range(number_channels)
            ]

            images = [
                cv.warpAffine(
                    new_input,
                    M=crop,
                    dsize=(512, 512),
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=-1000.0,
                )
                for new_input in inputs_3d
            ]

            labels_3d = [
                np.squeeze(d["label"])[channel] for channel in range(number_channels)
            ]
            labels = [
                cv.warpAffine(
                    new_label,
                    M=crop,
                    dsize=(512, 512),
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                for new_label in labels_3d
            ]
            d["input"] = np.stack(images)
            d["label"] = np.stack(labels)
        return d

    def inverse(self, data):
        d = copy.deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = data[TraceKeys.ORIG_SIZE]
            print(orig_size)
            untransform = cv.invertAffineTransform(transform)
            print(untransform)
            d[key] = cv.warpAffine(
                d[key],
                M=untransform,
                dsize=(1000, 1000),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=-1000.0,
            )
            self.pop_transform(d, key)
        return d
