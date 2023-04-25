import os
import shutil
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import pydicom
import pytorch_lightning as pl
from monai.data.image_reader import NibabelReader
from monai.transforms import (
    AddChanneld,
    Compose,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Rotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    SqueezeDimd,
)
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from utils.io import LoadDicomd
from utils.manual_cropping import Crop_3Dd, Cropd
from utils.misc import slicer


# defining the pl Data Module
class SarcoInfDataModule(pl.LightningDataModule):
    def __init__(self, infer_path=None, save_dir=None):
        super().__init__()
        self.infer_path = Path(infer_path)
        self.save_dir = Path(save_dir)
        self.pre_predict_transforms = Compose(
            [
                LoadDicomd(
                    keys="input",
                    dtype="float32",
                    image_only=False,
                    meta_keys="input_dict",
                ),
                LoadDicomd(
                    keys="label",
                    dtype="float32",
                    image_only=True,
                ),
                Cropd(keys=["input", "label"]),
                AddChanneld(keys=["input", "label"]),
                ScaleIntensityRanged(
                    keys=["input", "label"],
                    a_min=-3000,
                    a_max=3000,
                    b_min=-1,
                    b_max=1,
                    clip=True,
                ),
                EnsureTyped(keys=["input", "label"], data_type="tensor"),
            ]
        )
        self.post_predict_transforms = Compose(
            [
                EnsureTyped(
                    keys=["input", "label"], data_type="tensor", dtype="float32"
                ),
                #Invertd(
                #    keys=["label"],
                #    transform=self.pre_predict_transforms,
                #    orig_keys=["input"],
                #    orig_meta_keys="input_dict",
                #    nearest_interp=False,
                #),
                SqueezeDimd(keys="label", dim=0),
                SqueezeDimd(keys="label", dim=0),
                EnsureTyped(
                    keys=["input", "label"], data_type="numpy", dtype="float32"
                ),
            ]
        )

    @staticmethod
    def _dicom_index(filepath):
        dcm = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        return dcm.SliceLocation

    def setup(self, stage=None):
        if stage == "predict":
            Path(f"{self.save_dir}/inputdata").mkdir(parents=True, exist_ok=True)
            input_path = Path(f"{self.save_dir}/inputdata/{self.infer_path.name}")
            shutil.copytree(
                self.infer_path,
                input_path,
                dirs_exist_ok=True,
            )
            to_infer_dict = []
            files = sorted(
                [file for file in self.infer_path.glob("*")], key=self._dicom_index
            )
            for slice in sorted(files, key=self._dicom_index):
                to_infer_dict.append(
                    {
                        "input": input_path / slice.name,
                        "label": Path(input_path / slice.name),
                    }
                ),

            self.pred_ds = monai.data.Dataset(
                data=to_infer_dict, transform=self.pre_predict_transforms
            )
            print(list(self.pred_ds)[0])

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_ds, batch_size=1, num_workers=4, shuffle=False
        )

    def post_transform_dataset(self, x):
        return self.post_predict_transforms(x)


class SarcoInfDataModule_3D(pl.LightningDataModule):
    def __init__(self, infer_path=None, save_dir=None):
        super().__init__()
        self.infer_path = Path(infer_path)
        self.save_dir = Path(save_dir)
        self.pre_predict_transforms = Compose(
            [
                LoadDicomd(
                    keys="input",
                    dtype="float32",
                    image_only=False,
                    meta_keys="input_dict",
                ),
                LoadDicomd(
                    keys="label",
                    dtype="float32",
                    image_only=True,
                ),
                Crop_3Dd(keys=["input", "label"]),
                ScaleIntensityRanged(
                    keys=["input", "label"],
                    a_min=-3000,
                    a_max=3000,
                    b_min=-1,
                    b_max=1,
                    clip=True,
                ),
                EnsureTyped(keys=["input", "label"], data_type="tensor"),
            ]
        )
        self.post_predict_transforms = Compose(
            [
                EnsureTyped(keys=["label"], data_type="tensor", dtype="float32"),
                # Invertd(
                #    keys=["pred"],
                #    transform=self.pre_predict_transforms,
                #    orig_keys=["input_dict"],
                # ),
                SqueezeDimd(keys="label", dim=0),
                SaveImaged(
                    keys=["label"],
                    output_dir=Path(f"{self.save_dir}/predictions/slices"),
                    output_ext=".nii.gz",
                    separate_folder=False,
                    meta_key_postfix=None,
                    squeeze_end_dims=False,
                ),
            ]
        )

    def setup(self, stage=None):
        if stage == "predict":
            Path(f"{self.save_dir}/inputdata").mkdir(parents=True, exist_ok=True)
            input_path = Path(f"{self.save_dir}/inputdata/{self.infer_path.name}")
            shutil.copytree(
                self.infer_path,
                input_path,
                dirs_exist_ok=True,
            )
            to_infer_dict = []
            slices = list(self.infer_path.glob("*"))
            number_files = len(slices)
            for k in range(0, number_files // 3, 3):
                to_infer_dict.append(
                    {
                        "input": [
                            Path(input_path / slices[k].name),
                            Path(input_path / slices[k + 1].name),
                            Path(input_path / slices[k + 2].name),
                        ],
                        "label": [
                            Path(input_path / slices[k].name),
                            Path(input_path / slices[k + 1].name),
                            Path(input_path / slices[k + 2].name),
                        ],
                    }
                ),

            self.pred_ds = monai.data.Dataset(
                data=to_infer_dict, transform=self.pre_predict_transforms
            )

    def predict_dataloader(self):
        return DataLoader(dataset=self.pred_ds, batch_size=1, num_workers=4)

    def post_transform_dataset(self, x):
        return self.post_predict_transforms(x)
