from pathlib import Path

import monai
import nibabel as nib
import pydicom
import pytorch_lightning as pl
import torch
import torch._tensor
from monai.data.utils import get_valid_patch_size
from monai.transforms import (
    AddChanneld,
    Compose,
    EnsureTyped,
    Flipd,
    LoadImaged,
    Rotate90d,
    ScaleIntensityRanged,
    SqueezeDimd,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from utils.data_augment import Augmentd, Mixupd, ProductDataset, ToBinaryd
from utils.io import LoadDicomd, Renamed, PopKeyd
from utils.manual_cropping import Crop_3Dd, Cropd


# defining the pl Data Module
class SarcoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        do_mixup,
        batch_size,
        aug_prob,
        do_flips,
        do_elastic_transforms,
        add_predicted_data,
        path_to_prediction,
        mixup_alpha,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.do_mixup = do_mixup
        self.aug_prob = aug_prob
        self.do_flips = do_flips
        self.do_elastic_transforms = do_elastic_transforms
        self.add_predicted_data = add_predicted_data
        self.path_to_prediction = path_to_prediction
        self.mixup_alpha = mixup_alpha
        self.train_transforms = self.common_transforms(step="train")
        self.val_transforms = self.common_transforms(step="val")

    def common_transforms(self, step=None):
        transforms = [
            LoadDicomd(keys=["input"], meta_keys="input_dict", allow_missing_keys=True, dtype='float32'),
            LoadImaged(
                keys=["label"],
                reader="nibabelreader",
                dtype="float32",
                image_only=False,
                meta_keys="label_dict",
                allow_missing_keys=True,
            ),
            Flipd(keys="label", spatial_axis=0, allow_missing_keys=True),
            SqueezeDimd(keys=["label"], dim=2, allow_missing_keys=True),
            AddChanneld(keys=["input", "label"], allow_missing_keys=True),
            Rotate90d(keys="label", k=1, spatial_axes=[0, 1], allow_missing_keys=True),
            SqueezeDimd(keys=["input", "label"], dim=0, allow_missing_keys=True),
            Cropd(
                keys=["input", "label"],
                allow_missing_keys=True,
            ),
            AddChanneld(keys=["input", "label"], allow_missing_keys=True),
            ScaleIntensityRanged(
                keys="input",
                a_min=-3000,
                a_max=3000,
                b_min=-1,
                b_max=1,
                clip=True,
                allow_missing_keys=True,
            ),
        ]
        if self.add_predicted_data:
            print("adding predicted data transforms")
            transforms.append(
                LoadDicomd(
                    keys=["input_pred"],
                    meta_keys="input_pred_dict",
                    allow_missing_keys=True,
                    dtype='float32'
                )
            ),
            transforms.append(
                LoadImaged(
                    keys=["label_pred"],
                    image_only=False,
                    dtype="float32",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                ScaleIntensityRanged(
                    keys="input_pred",
                    a_min=-1000,
                    a_max=60000,
                    b_min=-1,
                    b_max=1,
                    clip=True,
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                AddChanneld(keys=["label_pred", "input_pred"], allow_missing_keys=True)
            )
            transforms.append(
                Renamed(
                    keys="input_pred",
                    src_key="input_pred",
                    dst_key="input",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="label_pred",
                    src_key="label_pred",
                    dst_key="label",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="input_pred_dict",
                    src_key="input_pred_dict",
                    dst_key="input_dict",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="label_pred_meta_dict",
                    src_key="label_pred_meta_dict",
                    dst_key="label_dict",
                    allow_missing_keys=True,
                )
            )
            transforms.append(PopKeyd(keys='input_transforms', src_key='input_transforms', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='label_transforms', src_key='label_transforms', allow_missing_keys=True)
            )
            transforms.append(PopKeyd(keys='label_pred_transforms', src_key='label_pred_transforms', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='input_dict', src_key='input_dict', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='label_dict', src_key='label_dict', allow_missing_keys=True)
            )
        if step == "train":
            return Compose(transforms)
        if step == "val":
            transforms.append(
                EnsureTyped(
                    keys=["input"],
                    data_type="tensor",
                    dtype=torch.float,
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                EnsureTyped(
                    keys=["input", "label"],
                    data_type="tensor",
                    allow_missing_keys=True,
                )
            )
            return Compose(transforms)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            files = sorted(
                [
                    filename
                    for filename in self.data_dir.glob("*")
                    if filename != Path(f"{self.data_dir}/HyuyQN0Tca.nii.gz")
                    and filename != Path(f"{self.data_dir}/HyuyQN0Tca.dcm")
                ]
            )
            data = []

            for k in range(0, len(sorted(files)), 2):
                input_dict = files[k]
                label_dict = files[k + 1]
                data.append(
                    {
                        "input": input_dict,
                        "label": label_dict,
                    }
                )
            self.nb_train_files=len(data)

            if self.add_predicted_data:
                print("******************")
                print("The length of the dataset was ", len(data))
                path = self.path_to_prediction
                predict_input = sorted([
                    inp for inp in Path(f"{path}/inputdata/dicom_preprocessed").glob("*")
                ])
                print(len(predict_input))
                predict_output = sorted([
                    out for out in Path(f"{path}/predictions/slices").glob("*")
                ])
                for k in range(len(predict_input) // 5):
                    input_dict = predict_input[k]
                    output_dict = predict_output[k]
                    data.append({"input_pred": input_dict, "label_pred": output_dict})
                print("After adding the predicted files, it's now ", len(data))
                print("*******************")

            # Experiences

            nbo=int(self.nb_train_files*0.8 // 1)

            self.train_files = data[0:nbo]
            self.val_files = data[nbo:]


            if self.do_mixup:
                print("*********************************")
                print("Performing Mixup on the train dataset")
                print("*********************************")
                train_ds = monai.data.Dataset(
                    data=self.train_files,
                    transform=self.common_transforms(step="train"),
                )
                self.mixup_train_ds = ProductDataset(
                    dataset1=train_ds,
                    dataset2=train_ds,
                    transform=Compose(
                        [
                            Mixupd(keys=["input", "label"], alpha=0.3),
                            Augmentd(
                                keys=["input", "label"],
                                proba=self.aug_prob,
                                do_flips=self.do_flips,
                                do_elastic_transforms=self.do_elastic_transforms,
                            ),
                            EnsureTyped(
                                keys=["input"],
                                data_type="tensor",
                                dtype=torch.float,
                            ),
                            EnsureTyped(
                                keys=["label"],
                                data_type="tensor",
                                dtype=torch.float,
                            ),
                        ]
                    ),
                )

            else:
                print("*********************************")
                print("Performing classical data augmentations on the dataset")
                print("*********************************")
                self.num_steps = int(
                    ((300 * self.batch_size / len(self.train_files)) // 1)
                )
                self.nomixup_train_ds = monai.data.IterableDataset(
                    data=self.num_steps * self.train_files,
                    transform=Compose(
                        [
                            self.common_transforms(step="train"),
                            Augmentd(
                                keys=["input", "label"],
                                proba=self.aug_prob,
                                do_flips=self.do_flips,
                                do_elastic_transforms=self.do_elastic_transforms,
                            ),
                            EnsureTyped(
                                keys=["input"], data_type="tensor", dtype=torch.float
                            ),
                            EnsureTyped(keys=["label"], data_type="tensor"),
                        ]
                    ),
                )

            self.val_ds = monai.data.Dataset(
                data=self.val_files, transform=self.common_transforms(step="val")
            )
            self.num_steps = int(((300 * self.batch_size / len(self.train_files)) // 1))
            k=0
            #for element in list(self.nomixup_train_ds):
            #    if len(element)!=6:
            #        print('list does not have len of 6')
            #        k+=1
            #    if 'input_pred_dict' in element:
            #        print('element still has input_pred_dict')
            #        k+=1
            #    if element['input'].shape != torch.Size((1, 512, 512)):
            #        print(element['input'].shape)
            #        k+=1
            #print('there are ', k, ' problematic files')

    def train_dataloader(self):
        if self.do_mixup:

            sampler = RandomSampler(
                data_source=self.mixup_train_ds,
                num_samples=len(self.train_files) * self.num_steps,
                replacement=False,
            )
            print("*********************************")
            print(
                "The length of the mixuped train dataset is: ",
                self.num_steps * len(self.train_files),
            )
            print("*********************************")
            return DataLoader(
                self.mixup_train_ds,
                batch_size=self.batch_size,
                num_workers=4,
                drop_last=True,
                sampler=sampler,
            )

        else:
            print("*********************************")
            print(
                "The length of the non-mixuped train dataset is: ",
                len(self.train_files),
            )
            print("*********************************")
            return DataLoader(
                self.nomixup_train_ds,
                batch_size=self.batch_size,
                num_workers=4,
                drop_last=True,
            )

    def val_dataloader(self):
        print("*********************************")
        print(
            "The length of the val dataset is: ",
            len(self.val_ds),
        )
        print("*********************************")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=False,
        )


class SarcoDataModule_3D(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        do_mixup,
        batch_size,
        aug_prob,
        do_flips,
        do_elastic_transforms,
        add_predicted_data,
        path_to_prediction,
        mixup_alpha=0.3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.do_mixup = do_mixup
        self.aug_prob = aug_prob
        self.mixup_alpha = mixup_alpha
        self.do_flips = do_flips
        self.do_elastic_transforms = do_elastic_transforms
        self.add_predicted_data = add_predicted_data
        self.path_to_prediction = path_to_prediction
        self.train_transforms = self.common_transforms(step="train")
        self.val_transforms = self.common_transforms(step="val")

    def common_transforms(self, step=None):
        transforms = [
            LoadDicomd(keys=["input"], meta_keys="input_dict", allow_missing_keys=True),
            LoadImaged(
                keys=["label"],
                reader="nibabelreader",
                dtype="float32",
                image_only=False,
                meta_keys="label_dict",
                allow_missing_keys=True,
            ),
            Flipd(keys="label", spatial_axis=0, allow_missing_keys=True),
            Rotate90d(keys="label", k=3, spatial_axes=[0, 1], allow_missing_keys=True),
            Crop_3Dd(
                keys=["input", "label"],
                allow_missing_keys=True,
            ),
            ScaleIntensityRanged(
                keys="input",
                a_min=-3000,
                a_max=3000,
                b_min=-1,
                b_max=1,
                clip=True,
                allow_missing_keys=True,
            ),
        ]
        if self.add_predicted_data:
            print("adding predicted data transforms")
            transforms.append(
                LoadDicomd(
                    keys=["input_pred"],
                    meta_keys="input_pred_dict",
                    allow_missing_keys=True,
                    dtype='float32'
                )
            ),
            transforms.append(
                LoadImaged(
                    keys=["label_pred"],
                    image_only=False,
                    dtype="float32",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                ScaleIntensityRanged(
                    keys="input_pred",
                    a_min=-1000,
                    a_max=60000,
                    b_min=-1,
                    b_max=1,
                    clip=True,
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="input_pred",
                    src_key="input_pred",
                    dst_key="input",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="label_pred",
                    src_key="label_pred",
                    dst_key="label",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="input_pred_dict",
                    src_key="input_pred_dict",
                    dst_key="input_dict",
                    allow_missing_keys=True,
                )
            )
            transforms.append(
                Renamed(
                    keys="label_pred_meta_dict",
                    src_key="label_pred_meta_dict",
                    dst_key="label_dict",
                    allow_missing_keys=True,
                )
            )
            transforms.append(PopKeyd(keys='input_transforms', src_key='input_transforms', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='label_transforms', src_key='label_transforms', allow_missing_keys=True)
            )
            transforms.append(PopKeyd(keys='label_pred_transforms', src_key='label_pred_transforms', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='input_dict', src_key='input_dict', allow_missing_keys=True))
            transforms.append(PopKeyd(keys='label_dict', src_key='label_dict', allow_missing_keys=True)
            )
        if step == "train":
            return Compose(transforms)
        if step == "val":
            transforms.append(
                EnsureTyped(
                    keys=["input", "label"], data_type="tensor", dtype=torch.float
                )
            )
            return Compose(transforms)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            files = sorted(
                [
                    filename
                    for filename in self.data_dir.glob("*")
                    if filename != Path(f"{self.data_dir}/HyuyQN0Tca.nii.gz")
                    and filename != Path(f"{self.data_dir}/HyuyQN0Tca.dcm")
                ]
            )
            data = []

            for k in range(0, len(sorted(files)), 2):
                input_dict = [files[k], files[k], files[k]]
                label_dict = [files[k + 1], files[k + 1], files[k + 1]]
                data.append(
                    {
                        "input": input_dict,
                        "label": label_dict,
                    }
                )

            if self.add_predicted_data:
                print("******************")
                print("The length of the dataset was ", len(data))
                path = self.path_to_prediction
                predict_input = sorted([
                    inp for inp in Path(f"{path}/inputdata/dicom_preprocessed").glob("*")
                ])
                predict_output = sorted([
                    out for out in Path(f"{path}/predictions/slices").glob("*")
                ])
                for k in range(0, len(predict_input) // 3, 3):
                    input_dict = [
                        predict_input[k],
                        predict_input[k + 1],
                        predict_input[k + 2],
                    ]
                    label_dict = [
                        predict_output[k],
                        predict_output[k + 1],
                        predict_output[k + 2],
                    ]

                    data.append(
                        {
                            "input_pred": input_dict,
                            "label_pred": label_dict,
                        }
                    )
                print("After adding the predicted files, it's now ", len(data))
                print("*******************")

            self.train_files, val_files = train_test_split(data, test_size=0.2, shuffle=False)

            if self.do_mixup:
                print("*********************************")
                print("Performing Mixup on the train dataset")
                print("*********************************")
                train_ds = monai.data.Dataset(
                    data=self.train_files,
                    transform=self.common_transforms(step="train"),
                )
                self.mixup_train_ds = ProductDataset(
                    dataset1=train_ds,
                    dataset2=train_ds,
                    transform=Compose(
                        [
                            Mixupd(keys=["input", "label"], alpha=self.mixup_alpha),
                            Augmentd(
                                keys=["input", "label"],
                                proba=self.aug_prob,
                                do_flips=self.do_flips,
                                do_elastic_transforms=self.do_elastic_transforms,
                            ),
                            EnsureTyped(
                                keys=["input"],
                                data_type="tensor",
                                dtype=torch.float,
                            ),
                            EnsureTyped(
                                keys=["label"],
                                data_type="tensor",
                                dtype=torch.float,
                            ),
                        ]
                    ),
                )

            else:
                print("*********************************")
                print("Performing classical data augmentations on the dataset")
                print("*********************************")

                self.nomixup_train_ds = monai.data.Dataset(
                    data=self.train_files
                    * int(((500 * self.batch_size / len(self.train_files)) // 1)),
                    transform=Compose(
                        [
                            self.train_transforms,
                            Augmentd(
                                keys=["input", "label"],
                                proba=self.aug_prob,
                                do_flips=self.do_flips,
                                do_elastic_transforms=self.do_elastic_transforms,
                            ),
                            EnsureTyped(
                                keys=["input"], data_type="tensor", dtype=torch.float
                            ),
                            EnsureTyped(keys=["label"], data_type="tensor"),
                        ]
                    ),
                )

            self.val_ds = monai.data.Dataset(
                data=val_files, transform=self.common_transforms(step="val")
            )

    def train_dataloader(self):
        if self.do_mixup:
            num_samples = 270 * int(
                ((500 * self.batch_size / len(self.train_files)) // 1)
            )

            sampler = RandomSampler(
                data_source=self.mixup_train_ds,
                num_samples=num_samples,
                replacement=False,
            )
            print("*********************************")
            print(
                "The length of the mixuped train dataset is: ",
                270 * int(((500 * self.batch_size / len(self.train_files)) // 1)),
            )
            print("*********************************")
            return DataLoader(
                self.mixup_train_ds,
                batch_size=self.batch_size,
                num_workers=4,
                drop_last=True,
                sampler=sampler,
            )
        else:
            print("*********************************")
            print(
                "The length of the non-mixuped train dataset is: ",
                len(self.train_files),
            )
            print("*********************************")
            return DataLoader(
                self.nomixup_train_ds,
                batch_size=self.batch_size,
                num_workers=4,
                drop_last=True,
                pin_memory=False,
            )

    def val_dataloader(self):
        print("*********************************")
        print(
            "The length of the val dataset is: ",
            len(self.val_ds),
        )
        print("*********************************")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=False,
            pin_memory=False,
        )
