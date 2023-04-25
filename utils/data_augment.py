import albumentations as A
import cv2 as cv
import numpy as np
import torch
from albumentations import Blur, Flip, GridDistortion
from monai.data import IterableDataset
from monai.transforms import (
    InvertibleTransform,
    MapTransform,
    RandRotate90,
    Transform,
    apply_transform,
)
from torch.utils.data import Dataset, get_worker_info


class Augmentd(MapTransform, InvertibleTransform):
    """

    This transform is a dictionary based transform that performs some classical data augmentations: flips and elastic transforms.
    Ensure the input data to be a PyTorch Tensor or numpy array.
    It is based on monai.transforms.
    """

    def __init__(
        self,
        keys,
        proba=0.2,
        interpolation=cv.INTER_LINEAR,
        border_mode=cv.BORDER_CONSTANT,
        num_steps=5,
        distort_limit=0.4,
        blur_limit=(3, 7),
        do_flips=True,
        do_elastic_transforms=True,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
            proba: probability to apply the transform on the data.
            interpolation: interpolation mode for the Grid Distortion. Default is linear interpolation.
            distort_limit: distortion limit dor the grid distortion.
            blur_limit: blur limit for the blur transform.
            num_steps: count on grid cells on each side of the grid distortion.
            do_flips: whether or not to perform geometrical transforms which are rotations and horizontal/vertical flips.
            do_elastic_transforms: whether ot not to perform elastic transforms which are blur and grid distortion.
        """
        super().__init__(keys)
        self.proba = proba
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.distort_limit = distort_limit
        self.blur_limit = blur_limit
        self.num_steps = num_steps
        self.do_flips = do_flips
        self.do_elastic_transforms = do_elastic_transforms

    def __call__(self, data):
        d = dict(data)
        transform = []
        if self.do_flips:
            transform.append(Flip(p=self.proba))
        if self.do_elastic_transforms:
            transform.append(
                GridDistortion(
                    num_steps=self.num_steps,
                    distort_limit=self.distort_limit,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    p=self.proba,
                )
            )
            transform.append(
                Blur(blur_limit=7, p=self.proba),
            )
        rotator = RandRotate90(prob=self.proba, spatial_axes=[0, 1])
        transform = A.Compose(transforms=transform)
        for key in self.key_iterator(d):
            self.push_transform(
                d,
                key,
            )
            transformed = transform(image=d[key])
            d[key] = transformed["image"]
            if self.do_flips:
                d[key] = rotator(d[key])
        return d


class ProductDataset(Dataset):
    """
    Creates a new dataset from two datasets. _get_item_ returns a dictionary of two elements each one from a different dataset with a corresponding key.
    It is useful for mixup where we need a dataset of pairs to mix them together.
    """

    def __init__(self, dataset1, dataset2, transform=None):
        super().__init__
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transform

    def __len__(self):
        return len(self.dataset1) * len(self.dataset2)

    def __getitem__(self, index):
        index_1 = index // len(self.dataset1)
        index_2 = index % len(self.dataset2)
        element = {"gauche": self.dataset1[index_1], "droite": self.dataset2[index_2]}
        if self.transform is not None:
            element = self.transform(element)
        return element


class Mixupd(InvertibleTransform, MapTransform):
    """
    This transform is a dictionary based transform that performs Mixup on a ProductDataset.
    """

    def __init__(self, keys, alpha) -> None:
        super().__init__(keys)
        self.alpha = alpha
        self.keys = keys

    def __call__(self, data):
        gauche = data["gauche"]
        droite = data["droite"]
        d = dict(droite)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        for key in MapTransform.key_iterator(self, d):
            self.push_transform(d, key)
            d[key] = np.multiply(gauche[key], lam) + np.multiply(droite[key], (1 - lam))
        return d


# from Laura's transform


class ToBinary(Transform):
    """
    Will convert input into binary by marking anythin larger than threshold as 1.

    Args:
        threshold: Anything >= threshold will be labeled 1 o.w 0
    """

    def __init__(self, threshold: float = 0):
        super().__init__()
        self.threshold = threshold

    def __call__(self, img):
        img = torch.where(img > self.threshold, 1, 0)
        return img


class ToBinaryd(MapTransform):
    """
    Dictionary wrapper of ToBinary
    """

    backend = ToBinary.backend

    def __init__(self, keys, threshold, allow_missing_keys):
        super().__init__(keys, allow_missing_keys)
        self.binarizer = ToBinary(threshold=threshold)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.binarizer(d[key])
        return d
