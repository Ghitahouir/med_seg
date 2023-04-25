import argparse
import glob
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from models.segmentation_model import *
from monai.transforms import SavitzkyGolaySmooth
from pytorch_lightning.callbacks import ModelCheckpoint
from sarco_infdatamodule import *
from torch.utils.data import Dataset
from utils.data_augment import ToBinary
from utils.io import Write_dicom

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_choice",
    choices=["best", "last"],
    type=str,
    help="choose best to load from the best checkpoint in terms of val "
    "loss and choose last to load from last checkpoint",
    default="best",
)
parser.add_argument(
    "--infer_path",
    type=str,
    help="path to the file to make the prediction from",
    default="/home/ghita/milvuetap/ghita_exploration/data/clean_CT_scans/scan_ABDO_SS_IV_PACS_CD_of_2.16.840.1.113669.632.20.1532476995.537038478.10000378356",
)

parser.add_argument(
    "--dirpath",
    type=str,
    help="the dirpath where the checkpoints are saved",
    default="/home/ghita/milvuetap/ghita_exploration/checkpoints",
)
parser.add_argument(
    "--run_id", type=str, help="the id of the run you want to load the checkpoints from"
)

parser.add_argument(
    "--dest_dir",
    type=str,
    help="the dirpath you wish to save the inference in",
    default="/home/ghita/milvuetap/ghita_exploration/data",
)
parser.add_argument("--gpu", type=int, help="choose which gpu to use", default=0)
parser.add_argument(
    "--log_wandb_project",
    type=str,
    help="name of the wandb project to log into",
    default="Predictions",
)

parser.add_argument(
    "--ckpt_wandb_project",
    type=str,
    help="name of the wandb project the run used for the inference is from",
    default="Predictions",
)

parser.add_argument(
    "--wandb", type=str, help="whether or not to enable wandb syncing", default="yes"
)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(value).lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        print(f"The value {value} was not set.")


def _dicom_index(filepath):
    dcm = pydicom.dcmread(str(filepath), stop_before_pixels=True)
    return dcm.SliceLocation


args = parser.parse_args()
infer_path = Path(args.infer_path)
dirpath = Path(args.dirpath)
dest_dir = Path(args.dest_dir)
gpu = args.gpu
run_id = args.run_id
ckpt_choice = args.ckpt_choice
log_wandb_project = args.log_wandb_project
ckpt_wandb_project = args.ckpt_wandb_project
wandb_enable = normalize_bool(args.wandb)

if __name__ == "__main__":

    save_dir = Path(dest_dir / f"preds_from_{run_id}_of_{infer_path.name}")

    os.mkdir(save_dir)

    # choosing the checkpoints

    train_2d = os.path.isdir(Path(dirpath / f"{run_id}_{ckpt_wandb_project}"))

    if train_2d:

        common_checkpoint_path = Path(dirpath / f"{run_id}_{ckpt_wandb_project}")

    else:
        common_checkpoint_path = Path(dirpath / f"{run_id}_{ckpt_wandb_project}_3d")

    checkpoints = sorted([ckpt for ckpt in common_checkpoint_path.glob("*")])
    if len(checkpoints) == 0:
        raise argparse.ArgumentError(
            argument=None,
            message="There are no checkpoints saved for this run, please choose another run.",
        )
    checkpoint_path = checkpoints[0] if ckpt_choice == "best" else checkpoints[1]

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("*********************************")
    print("You are using the following device :", device)
    print("*********************************")
    model = pl_UNet.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)

    if wandb_enable:
        run = wandb.init(
            project=f"{log_wandb_project}",
            settings=wandb.Settings(start_method="fork", config=args),
        )
    args = wandb.config

    logger = WandbLogger(project=f"{log_wandb_project}")

    if train_2d:

        datamodule_inference = SarcoInfDataModule(
            infer_path=infer_path, save_dir=save_dir
        )

    if not train_2d:
        datamodule_inference = SarcoInfDataModule_3D(
            infer_path=infer_path, save_dir=save_dir
        )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gpus=[gpu] if torch.cuda.is_available() else None,
        logger=False,
    )
    print("*********************************")
    print("The trainer is ready, about to predict")
    print("*********************************")
    preds = trainer.predict(model, datamodule=datamodule_inference)
    print("*********************************")
    print("Predicted, about to apply post transforms and save")
    print("*********************************")

    predictions = datamodule_inference.post_transform_dataset(x=preds)

    writer = Write_dicom(save=True)

    k = 0

    os.mkdir(
        f"{save_dir}/predictions",
    )
    os.mkdir(
        f"{save_dir}/predictions/slices",
    )

    os.mkdir(
        f"{save_dir}/inputdata/nii_slices",
    )

    os.mkdir(f"{save_dir}/inputdata/dicom_preprocessed")

    for element in predictions:
        nib.save(
            nib.Nifti1Image(element["input"], affine=None),
            f"{save_dir}/inputdata/nii_slices/%s_input.nii.gz" % k,
        )
        nib.save(
            nib.Nifti1Image(element["label"], affine=None),
            f"{save_dir}/predictions/slices/%s_mask.nii.gz" % k,
        )

        dicom = sorted(
            list(Path(f"{save_dir}/inputdata/{infer_path.name}").glob("*")),
            key=_dicom_index,
        )[k]

        writer(
            np.squeeze(element["input"]),
            dicom,
            f"{save_dir}/inputdata/dicom_preprocessed/%s_input.dcm" % k,
        )

        k += 1

    preds_files = sorted(
        [
            Path(f"{save_dir}/predictions/slices/%1s_mask.nii.gz" % k)
            for k in range(len(predictions))
        ]
    )

    input_files = sorted(
        [
            Path(f"{save_dir}/inputdata/nii_slices/%1s_input.nii.gz" % k)
            for k in range(len(predictions))
        ]
    )

    stacked_layers = numpy.array(
        [np.squeeze(nib.load(layer).get_fdata()) for layer in sorted(preds_files)]
    )

    input_stacked_layers = numpy.array(
        [np.squeeze(nib.load(layer).get_fdata()) for layer in sorted(input_files)]
    )

    bin = ToBinary()

    binarized_stacked_layers = numpy.array(
        bin(
            torch.tensor(
                [
                    np.squeeze(nib.load(layer).get_fdata())
                    for layer in sorted(preds_files)
                ],
                dtype=torch.float32,
            )
        ),
        dtype=np.float_,
    )
    nii_pred = nib.Nifti1Image(
        stacked_layers,
        affine=nib.load(
            list(Path(f"{save_dir}/predictions/slices").glob("*"))[0]
        ).affine,
    )

    nii_inp = nib.Nifti1Image(
        input_stacked_layers,
        affine=nib.load(
            list(Path(f"{save_dir}/inputdata/nii_slices").glob("*"))[0]
        ).affine,
    )
    bin_nii_pred = nib.Nifti1Image(
        binarized_stacked_layers,
        affine=nib.load(
            list(Path(f"{save_dir}/predictions/slices").glob("*"))[0]
        ).affine,
    )
    os.mkdir(f"{save_dir}/inputdata/stacked")

    os.mkdir(f"{save_dir}/predictions/stacked")
    os.mkdir(f"{save_dir}/predictions/bin_stacked")

    nib.save(nii_pred, Path(f"{save_dir}/predictions/stacked/nii_pred"))
    nib.save(nii_inp, Path(f"{save_dir}/inputdata/stacked/nii_inp"))

    nib.save(bin_nii_pred, Path(f"{save_dir}/predictions/bin_stacked/bin_nii_pred"))
