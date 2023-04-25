import argparse
import os
from pathlib import Path
from sqlite3 import Date

import pytorch_lightning as pl
import torch
from models.segmentation_model import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sarco_datamodule import *

import wandb

# parsing all the arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="the dirpath where the training/valid/test data is stored",
    default="/home/ghita/milvuetap/ghita_exploration/data/sarco dataset 1 & 2 improved",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    help="enter the learning rate",
    dest="lr",
    default=1e-4,
)
parser.add_argument(
    "--max_epochs",
    type=int,
    help="enter the max number of epochs for the training",
    default=30,
)
parser.add_argument("--batch_size", type=int, help="choose a batch_size", default=16)
parser.add_argument(
    "--aug_prob",
    type=float,
    help="probability to perform augmentations",
    default=0.2,
)
parser.add_argument(
    "--dirpath",
    type=str,
    help="the dirpath where to save the checkpoints from the training",
    default="/home/ghita/milvuetap/ghita_exploration/checkpoints",
)

parser.add_argument(
    "--mixup",
    help="whether or not to perform mixup data augmentation as described in : https://arxiv.org/pdf/1710.09412.pdf "
    "and https://arxiv.org/abs/1806.05236 ."
    "Else classical data augmentation will be performed",
    type=str,
    choices=["no", "input", "manifold"],
    default="no",
)
parser.add_argument(
    "--mixup_alpha",
    help="value of alpha to perform Input Mixup or Manifold Mixup",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--mixed_layer",
    help="layer chosen to perform Manifold Mixup on",
    type=int,
    default=0,
)
parser.add_argument(
    "--model_choice",
    help="Model choice",
    type=str,
    choices=["unet", "segresnet"],
    default="unet",
)
parser.add_argument(
    "--backbone",
    type=str,
    help="choose the backbone for the encoder",
    choices=[
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ],
    default="efficientnet-b0",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    help="name of the wandb project to log into",
    default="comparison_mixup",
)
parser.add_argument("--gpu", type=int, help="choose which gpu to use", default=0)
parser.add_argument(
    "--wandb", type=str, help="whether or not to enable wandb syncing", default="yes"
)
parser.add_argument(
    "--channel_3d",
    type=str,
    help="whether or not to have 3D channeled input",
    default="no",
)
parser.add_argument(
    "--do_elastic_transforms",
    help="whether to perform relastic transforms",
    type=str,
    default="no",
)
parser.add_argument(
    "--do_flips",
    help="whether to perform flips (randrot, vert, horiz)",
    type=str,
    default="no",
)
parser.add_argument(
    "--resume_training",
    help="whether to resume previous training",
    type=str,
    default="no",
)
parser.add_argument(
    "--ckpt_path_to_resume_from",
    help="Checkpoint path to resume from training if resume_trainig is True",
    type=str,
    default=None,
)
parser.add_argument(
    "--early_stopping",
    help="Whether to perform Early Stopping during training",
    type=str,
    default="yes",
)
parser.add_argument(
    "--patience",
    help="When using Early Stopping, specify the patience before stopping training",
    type=int,
    default=7,
)
parser.add_argument(
    "--add_predicted_data",
    help="whether to feed the model with predicted data from previous inference",
    type=str,
    default="no",
)
parser.add_argument(
    "--path_to_prediction",
    help="the path to the directory where the inputs and predictions are saved",
    type=str,
)


# starting a train

if __name__ == "__main__":

    def normalize_bool(value):
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif str(value).lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            print("*******/!\*******")
            print("The value %s was not set." % value)

    args = parser.parse_args()

    # saving the arguments
    data_dir = Path(args.data_dir)
    dirpath = Path(args.dirpath)
    learning_rate = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    model_choice = args.model_choice
    gpu = args.gpu
    backbone = args.backbone
    aug_prob = args.aug_prob
    do_flips = normalize_bool(args.do_flips)
    do_elastic_transforms = normalize_bool(args.do_elastic_transforms)
    wandb_project = args.wandb_project
    wandb_enable = normalize_bool(args.wandb)
    channel_3d = normalize_bool(args.channel_3d)
    resume_training = normalize_bool(args.resume_training)
    early_stopping = normalize_bool(args.early_stopping)
    patience = args.patience
    add_predicted_data = normalize_bool(args.add_predicted_data)
    if resume_training:
        ckpt_path_to_resume_from = Path(args.ckpt_path_to_resume_from)
    path_to_prediction = args.path_to_prediction
    if args.mixup == "no":
        do_manifold_mixup = False
        do_mixup = False
        print("*************")
        print("not performing Mixup")
    if args.mixup == "input":
        do_manifold_mixup = False
        do_mixup = True
        print("*************")
        print("performing Input Mixup")
    if args.mixup == "manifold":
        do_manifold_mixup = True
        do_mixup = False
        print("*************")
        print("performing Manifold Mixup")

    mixup_alpha = args.mixup_alpha
    mixed_layer = args.mixed_layer

    # if channel_3d and not add_predicted_data:
    #    raise argparse.ArgumentError(
    #        argument=None,
    #        message="It's not useful to stack inputs in 3D if you don't add predicted data",
    #    )

    # defining the model

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("*********************************")
    print("You are using the following device :", device)
    print("*********************************")
    if resume_training:
        if ckpt_path_to_resume_from is None:
            raise argparse.ArgumentError(
                argument=None,
                message="You chose to resume training but haven't provided from which checkpoint to resume from",
            )
        print("*********************************")
        print("Resuming training from :", ckpt_path_to_resume_from.name)
        print("*********************************")
        ckpt_used = list(ckpt_path_to_resume_from.glob("*"))[1]
        print("ckpt used", ckpt_used)
        model = pl_UNet.load_from_checkpoint(checkpoint_path=ckpt_used).to(device)

    else:

        model = pl_UNet(
            learning_rate=learning_rate,
            model_choice=model_choice,
            backbone=backbone,
            channel_3d=channel_3d,
            project_name=wandb_project,
            first_lr_decay=int((max_epochs * 0.8) // 1),
            second_lr_decay=int((max_epochs * 0.9) // 1),
            do_manifold_mixup=do_manifold_mixup,
            batch_size=batch_size,
            mixup_alpha=mixup_alpha,
            mixed_layer=mixed_layer,
        ).to(device)
    if wandb_enable:

        if resume_training:
            run = wandb.restore(ckpt_used)
            logger = WandbLogger(
                project=f"{wandb_project}",
                resume=resume_training,
                id=(ckpt_path_to_resume_from.name).split("_", 1)[0],
            )

        else:

            run = wandb.init(
                project=f"{wandb_project}",
                settings=wandb.Settings(start_method="fork", config=args),
            )
            logger = WandbLogger(project=f"{wandb_project}")
        args = wandb.config
        run_id = wandb.run.id

        if channel_3d:
            save_ckpt_dir = dirpath / f"{run_id}_{wandb_project}_3d"
        else:
            save_ckpt_dir = dirpath / f"{run_id}_{wandb_project}"

    checkpoint_callback_best = ModelCheckpoint(
        dirpath=save_ckpt_dir,
        filename="best_checkpoint-{epoch}-{step}",
        monitor="best_val_average_precision",
        save_top_k=1,
        mode="max",
    )

    checkpoint_callback_last = ModelCheckpoint(
        dirpath=save_ckpt_dir,
        filename="last_checkpoint-{epoch}-{step}",
        monitor="step",
        save_top_k=1,
        mode="max",
    )

    callbacks = [checkpoint_callback_best, checkpoint_callback_last]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="best_val_average_precision", mode="max", patience=patience
            )
        )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        precision=16,
        callbacks=callbacks,
        enable_checkpointing=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gpus=[gpu] if torch.cuda.is_available() else None,
    )

    if channel_3d:

        datamodule = SarcoDataModule_3D(
            data_dir,
            do_mixup=do_mixup,
            batch_size=batch_size,
            aug_prob=aug_prob,
            do_flips=do_flips,
            do_elastic_transforms=do_elastic_transforms,
            add_predicted_data=add_predicted_data,
            path_to_prediction=path_to_prediction,
            mixup_alpha=mixup_alpha,
        )

    else:

        datamodule = SarcoDataModule(
            data_dir,
            do_mixup=do_mixup,
            batch_size=batch_size,
            aug_prob=aug_prob,
            do_flips=do_flips,
            do_elastic_transforms=do_elastic_transforms,
            add_predicted_data=add_predicted_data,
            path_to_prediction=path_to_prediction,
            mixup_alpha=mixup_alpha,
        )

    if resume_training:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_used)
    else:
        trainer.fit(model, datamodule=datamodule)

    best_checkpoint_path = checkpoint_callback_best.best_model_path
    last_checkpoint_path = checkpoint_callback_last.best_model_path

    wandb.finish(1)
