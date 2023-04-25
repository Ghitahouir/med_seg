import re

import matplotlib
import matplotlib.pyplot as plt
import monai
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch._tensor
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.inferers import SimpleInferer
from monai.networks.nets import SegResNet
from monai.transforms import ScaleIntensityRange
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import binarize
from timm.scheduler import TanhLRScheduler
from torchmetrics.wrappers.minmax import MinMaxMetric
from utils.data_augment import ToBinary
from utils.misc import MyAccuracy, MyBinnedAP, MyDice, MyIoU

import wandb

from .mixup_unet import MixupUNet


# defining the pl module
class pl_UNet(pl.LightningModule):
    def __init__(
        self,
        model_choice="unet",
        learning_rate=1e-4,
        backbone="efficientnet-b0",
        project_name=None,
        # defining two steps on which the lr will be divided : first and second lr decay
        first_lr_decay=None,
        second_lr_decay=None,
        channel_3d=None,
        # to perform Manifold Mixup we need the following arguments
        do_manifold_mixup=False,
        batch_size=16,
        mixup_alpha=1.0,
        mixed_layer=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.channel_3d = channel_3d
        self.model_choice = model_choice
        self.project_name = project_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.do_manifold_mixup = do_manifold_mixup
        self.automatic_optimization = False
        self.first_lr_decay = first_lr_decay
        self.second_lr_decay = second_lr_decay
        self.train_metrics = self.common_metrics_builder("_train")
        self.val_metrics = self.common_metrics_builder("_val")
        self.best_val_average_precision = MinMaxMetric(
            base_metric=self.val_metrics["AP_val"], compute_on_step=False
        )

        self.dice_metric = MyDice()
        self.inferer = SimpleInferer()
        self.mixup_alpha = mixup_alpha
        self.mixed_layer = mixed_layer
        self.wandblogger = WandbLogger(project=self.project_name)
        if self.channel_3d:
            self.model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
            )

        if self.do_manifold_mixup:
            self.model = MixupUNet(
                alpha=self.mixup_alpha,
                encoder_name=self.backbone,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
            )
        else:
            if self.model_choice == "unet":
                self.model = smp.Unet(
                    encoder_name=self.backbone,
                    encoder_weights="imagenet",
                    in_channels=1,
                    classes=1,
                )
            if self.model_choice == "segresnet":
                self.model = SegResNet(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                )

        # Uncomment the following line to change the loss used

        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = monai.losses.DiceLoss(sigmoid=True)

    def common_metrics_builder(self, suffix=""):
        """This function builds the metrics dict for train and val steps"""
        return torch.nn.ModuleDict(
            {
                f"AP{suffix}": MyBinnedAP(num_classes=1, thresholds=1),
                f"iou{suffix}": MyIoU(num_classes=2),
                f"accuracy{suffix}": MyAccuracy(),
            }
        )

    def mixup(self, sample, alpha):
        """
        This function performs the Mixup interpolation and returns a mixed sample with same original size.
        Args :
            - sample : sample to be mixed
            - alpha : parameter of the Beta distribution to generate lambda, the interpolation factor.
            - index : sample[k] will be mixed with sample[k+index].
        """
        lam = np.random.beta(alpha, alpha)
        inv_idx = torch.arange(sample.size(0) - 1, -1, -1).long()
        inv_tensor = sample[inv_idx]
        mixed_x = lam * sample + (1 - lam) * inv_tensor
        return mixed_x

    def forward(self, x, step):
        if self.do_manifold_mixup:
            out = self.model(x, step)
        else:
            out = self.model(x)
        # hidden_states= self.encoder(out)
        # self.wandblogger.log_image(
        #        key="hidden_states",
        #        images=[(hid) for hid in hidden_states],
        #        step=50,
        #    )
        return out

    def training_step(self, train_batch, batch_nb):

        inputs = train_batch["input"]

        if self.do_manifold_mixup:
            labels = self.mixup(sample=train_batch["label"], alpha=self.mixup_alpha)
        else:
            labels = train_batch["label"]

        logits = self.forward(inputs, step="train")
        preds = torch.sigmoid(logits)

        # computing the metrics
        for _, to_compute_metric in self.train_metrics.items():
            to_compute_metric(logits, labels)

        self.log(
            "dice_metric_train",
            self.dice_metric(preds, labels).mean(),
            on_step=True,
            on_epoch=True,
        )

        optimizer = self.optimizers()
        #sch = self.lr_schedulers()
        optimizer.zero_grad()
        loss_train = self.loss(logits, labels)
        self.manual_backward(loss_train)
        optimizer.step()
        #sch.step()

        self.log("loss_train", loss_train, on_step=True, on_epoch=True)

        # logging other parameters

        #self.log("learning_rate", self.lr_schedulers().get_last_lr()[0], on_step=True)

        return loss_train

    def training_epoch_end(self, outputs) -> None:
        # logging the metrics and resetting them at the end of each epoch
        for metric_name, to_compute_metric in self.train_metrics.items():
            self.log(metric_name, to_compute_metric.compute())
            to_compute_metric.reset()

    def validation_step(self, val_batch, batch_nb):
        print(self.model)
        inputs = val_batch["input"]
        labels = val_batch["label"]
        logits = self.forward(inputs, step="val")
        preds = torch.sigmoid(logits)

        # computing metrics
        for _, to_compute_metric in self.val_metrics.items():
            to_compute_metric(logits, labels)

        # logging dice
        self.log(
            "dice_metric_val",
            self.dice_metric(preds, labels).mean(),
            on_step=True,
            on_epoch=True,
        )

        # logging preds
        if batch_nb % 50 == 0:
            self.wandblogger.log_image(
                key="prediction_val",
                images=[(predi) for predi in preds],
                step=50,
            )
            self.wandblogger.log_image(
                key="label_val",
                images=[lab for lab in labels],
                step=50,
            )

        # computing the monitor metric
        self.best_val_average_precision(logits, labels)

        # computing and logging the loss
        loss_val = self.loss(logits, labels)
        self.log("loss_val", loss_val, on_step=True, on_epoch=True)

        return loss_val

    def on_validation_epoch_end(self) -> None:
        # logging and resetting the metrics at the end of each epoch
        self.log(
            "best_val_average_precision",
            self.best_val_average_precision.compute()["max"],
            metric_attribute=self.best_val_average_precision,
        )
        for metric_name, to_compute_metric in self.val_metrics.items():
            self.log(
                metric_name,
                to_compute_metric.compute(),
            )
            to_compute_metric.reset()

        # manual lr decay at two steps of the training
        if self.current_epoch == self.first_lr_decay:
            self.learning_rate /= 10
        if self.current_epoch == self.second_lr_decay:
            self.learning_rate /= 5

    def predict_step(self, batch_data, batch_nb):
        d = dict(batch_data)
        d["input"] = batch_data["input"]
        d["label"] = torch.sigmoid(self.inferer(inputs=d["input"], network=self.model))

        self.wandblogger.log_image("to_predict", list(d["input"]))
        self.wandblogger.log_image("predicted", list(d["label"]))

        return d

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = LinearWarmupCosineAnnealingLR(
        #    optimizer, warmup_epochs=10, max_epochs=80
        #)
        #return [optimizer], [scheduler]
        return [optimizer]
