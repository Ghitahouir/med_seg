import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import torch
from monai.metrics import CumulativeIterationMetric, DiceMetric
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.transforms import MapTransform, ShiftIntensity
from monai.utils import ensure_tuple, ensure_tuple_rep
from torch.nn import Module
from torchmetrics import (
    Accuracy,
    BinnedAveragePrecision,
    BinnedPrecisionRecallCurve,
    ConfusionMatrix,
    JaccardIndex,
)
from torchmetrics.functional.classification.accuracy import (
    _accuracy_update,
    _check_subset_validity,
    _mode,
    _subset_accuracy_update,
)
from torchmetrics.functional.classification.average_precision import (
    _average_precision_compute_with_precision_recall,
)
from torchmetrics.functional.classification.confusion_matrix import (
    _confusion_matrix_update,
)
from torchmetrics.functional.classification.jaccard import _jaccard_from_confmat
from torchmetrics.utilities.data import to_onehot

#########
# This script gathers some miscellaneous functions and classes that are used for the training and the inference.
#########

# converting dicoms to niis


def convert_dicom_to_nii(
    path_to_dcm, output_path, compress, keep_orientation, affine=np.eye(4)
):
    """This function converts a dicom file to a nii file.
    Args :
        - path_to_dcm : str or path of the dcm file
        - output_path : where to store the nii created
        - compress whether to add ".gz" in the extension
        - keep_orientation : whether to keep the original orientation or change it to the Nibabel normal one
        - affine : affine transformation if not identity, default=np.eye(4)
    """
    input_dicom = pydicom.dcmread(Path(path_to_dcm))
    slope = float(input_dicom["RescaleSlope"].value)
    inter = float(input_dicom["RescaleIntercept"].value)
    rescaled = input_dicom.pixel_array * slope + inter
    if keep_orientation:
        output_nifti = nib.Nifti1Image(rescaled, np.eye(4))
    else:
        output_nifti = nib.Nifti1Image(rescaled, affine)
    if not compress:
        nib.save(output_nifti, f"{output_path}.nii")
    else:
        nib.save(output_nifti, f"{output_path}.nii.gz")


# saving multiple slices nifti file to multiple single slice nifiti files


def slicer(input_path, dest_path):
    """This function takes a nifti file containing multiple slices and saves
    each slice separatly in the corresponding dest_path as one slice nifti files"""
    input_nifti = nib.load(Path(input_path))
    slices = input_nifti.get_fdata()
    if len(slices.shape) == 3:
        for k in range(slices.shape[2]):
            slice = slices[:, :, k]
            nifti_slice = nib.Nifti1Image(slice, np.eye(4))
            nib.save(
                img=nifti_slice,
                filename=f"{dest_path}/slice%s.nii.gz" % k,
            )
    if len(slices.shape) == 2:
        print("********************")
        print("Careful, the input only contains a single slice. It's a 2D input.")
        print("********************")
        shutil.copy(input_path, f"{dest_path}/slice0.nii.gz")
    else:
        print("The input doesn't have the right shape. The shape is :", slices.shape)


# computing some metrics with changed inputs format


class MyBinnedAPRecallCurve(BinnedPrecisionRecallCurve):
    def update(self, preds, target) -> None:  # type: ignore
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        # we have to rewrite the metric to flatten the data before using it to compute the AP.

        preds = torch.flatten(preds)
        target = torch.flatten(target)
        if len(preds.shape) == len(target.shape) == 1:
            preds = preds.reshape(-1, 1)
            target = target.reshape(-1, 1)

        if len(preds.shape) == len(target.shape) + 1:
            target = to_onehot(target, num_classes=self.num_classes)

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            self.TPs[:, i] += (target & predictions).sum(dim=0)
            self.FPs[:, i] += ((~target) & (predictions)).sum(dim=0)
            self.FNs[:, i] += ((target) & (~predictions)).sum(dim=0)


class MyBinnedAP(MyBinnedAPRecallCurve):
    def compute(self):  # type: ignore
        precisions, recalls, _ = super().compute()
        return _average_precision_compute_with_precision_recall(
            precisions, recalls, self.num_classes, average=None
        )


class MyAccuracy(Accuracy):
    def update(self, preds, target) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.

        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        """ returns the mode of the data (binary, multi label, multi class, multi-dim multi class) """

        target = target.type(torch.int)
        preds = preds
        mode = _mode(
            preds, target, self.threshold, self.top_k, self.num_classes, self.multiclass
        )

        if not self.mode:
            self.mode = mode
        elif self.mode != mode:
            raise ValueError(f"You can not use {mode} inputs with {self.mode} inputs.")

        if self.subset_accuracy and not _check_subset_validity(self.mode):
            self.subset_accuracy = False

        if self.subset_accuracy:
            correct, total = _subset_accuracy_update(
                preds,
                target,
                threshold=self.threshold,
                top_k=self.top_k,
            )
            self.correct += correct
            self.total += total
        else:
            if not self.mode:
                raise RuntimeError("You have to have determined mode.")
            tp, fp, tn, fn = _accuracy_update(
                preds,
                target,
                reduce=self.reduce,
                mdmc_reduce=self.mdmc_reduce,
                threshold=self.threshold,
                num_classes=self.num_classes,
                top_k=self.top_k,
                multiclass=self.multiclass,
                ignore_index=self.ignore_index,
                mode=self.mode,
            )

            # Update states
            if self.reduce != "samples" and self.mdmc_reduce != "samplewise":
                self.tp += tp
                self.fp += fp
                self.tn += tn
                self.fn += fn
            else:
                self.tp.append(tp)
                self.fp.append(fp)
                self.tn.append(tn)
                self.fn.append(fn)


class MyDice(DiceMetric):
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute dice (BxC) for each channel for each batch
        return compute_meandice(
            y_pred=y_pred, y=y, include_background=self.include_background
        )


def compute_meandice(y_pred, y, include_background=True) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, num_classes]).

    Raises:

        ValueError: when `y_pred` and `y` have different shapes.

    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = abs(torch.sum(y * y_pred, dim=reduce_axis))

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = abs(y_o + y_pred_o)

    return torch.where(
        y_o > 0,
        (2.0 * intersection) / denominator,
        torch.tensor(float("nan"), device=y_o.device),
    )


class MyJaccardIndex(ConfusionMatrix):
    def update(self, preds, target) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        target = target.int()
        confmat = _confusion_matrix_update(
            preds, target, self.num_classes, self.threshold, self.multilabel
        )
        self.confmat += confmat


class MyIoU(MyJaccardIndex):
    is_differentiable = False
    higher_is_better = True

    def __init__(
        self,
        num_classes: int,
        ignore_index=None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        reduction: str = "elementwise_mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group=None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self):
        """Computes intersection over union (IoU)"""
        return _jaccard_from_confmat(
            self.confmat,
            self.num_classes,
            self.ignore_index,
            self.absent_score,
            self.reduction,
        )


# trying to write a transform that converts pixel value to HU


class PixeltoHUd(MapTransform):
    """
    Dictionary-based transform that transforms Dicoms' pixel intensity to Housefields Units`.
    Eventually discovered that it cannot work for now as we cannot overwrite the Rescale Slope/Intercept of a Dicom, which is what I wanted to do here.
    Maybe it will be possible in incoming versions of pydicom.
    """

    backend = ShiftIntensity.backend

    def __init__(
        self,
        keys,
        meta_keys,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                used to extract the factor value is `factor_key` is not None.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                used to extract the factor value is `factor_key` is not None.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)

        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        scl_inter = data.header["scl_inter"]
        scl_slope = data.header["scl_slope"]

        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            d[key] = d[key] * scl_slope + scl_inter
        return d


def empty_checkpoint_clear(ckpt_dirpath):
    """
    Deletes empty checkpoint directories that where created for bugged runs.
    """
    ckpt_paths = [ckpts for ckpts in ckpt_dirpath.glob("*")]
    for path in ckpt_paths:
        inside = os.listdir(path)
        if len(inside) == 0:
            os.rmdir(path)
