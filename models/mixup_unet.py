import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MixupUNet(smp.Unet):
    def __init__(
        self,
        alpha,
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ):
        super().__init__()
        self.alpha = (alpha,)
        self.in_channels = (in_channels,)
        self.encoder_name = (encoder_name,)
        self.encoder_weights = (encoder_weights,)
        self.classes = (classes,)

    def mixup_features(self, features: list):
        lam = torch.distributions.beta.Beta(
            self.alpha[0], self.alpha[0], validate_args=None
        )
        lam = lam.sample()
        mixed_features = []
        for feature in features:
            inv_idx = torch.arange(feature.size(0) - 1, -1, -1).long()
            inv_tensor = feature[inv_idx]
            new_feature = lam * feature + (1 - lam) * inv_tensor
            mixed_features.append(new_feature)
        return mixed_features

    def get_features(self, x):
        x = x.repeat(1, 3, 1, 1)
        features = self.encoder(x)
        mixed_features = self.mixup_features(features)
        return features, mixed_features

    def forward(self, x, step):

        x = x.repeat(1, 3, 1, 1)
        features = self.encoder(x)
        mixed_features = self.mixup_features(features)
        if step == 'train': 
            decoder_output = self.decoder(*mixed_features)
        if step == 'val': 
            decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            if step =='train':
                labels = self.classification_head(mixed_features[-1])
            if step == 'val': 
                labels = self.classification_head(features[-1])
            return masks, labels

        return masks
