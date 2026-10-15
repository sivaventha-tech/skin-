"""
============================================================
SKIN DISEASE CLASSIFICATION
UAGRF Deep Ensemble Model
Uncertainty-Aware Gain Ratio Fusion Architecture
IEEE Transaction Grade Ensemble Method
============================================================

Combines MSTA-Net models built on top of:
1. EfficientNet-B4 MSTA
2. ConvNeXt-Small MSTA
3. ResNet50 MSTA

Uses Dynamic Softmax Prediction Entropy + Empirical Gain Ratio Weighting.
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.msta_net import MSTANet


class UAGRFEnsemble(nn.Module):
    """
    Uncertainty-Aware Gain Ratio Fusion Ensemble for Skin Disease Classification.
    """
    def __init__(self, num_classes=22, pretrained=True, gain_ratios=None):
        super().__init__()
        self.num_classes = num_classes
        
        # Load component models
        self.model_effnet = MSTANet(num_classes=num_classes, backbone_name="efficientnet_b4", pretrained=pretrained)
        self.model_convnext = MSTANet(num_classes=num_classes, backbone_name="convnext_small", pretrained=pretrained)
        self.model_resnet = MSTANet(num_classes=num_classes, backbone_name="resnet50", pretrained=pretrained)
        
        # Gain Ratio weights (default to equal weights if not provided)
        if gain_ratios is None:
            self.gain_ratios = nn.Parameter(torch.tensor([0.35, 0.35, 0.30]), requires_grad=False)
        else:
            self.gain_ratios = nn.Parameter(torch.tensor(gain_ratios, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        logits1 = self.model_effnet(x)
        logits2 = self.model_convnext(x)
        logits3 = self.model_resnet(x)

        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        p3 = F.softmax(logits3, dim=1)

        # Entropy calculation for uncertainty weighting: H(p) = - sum(p * log(p + 1e-8))
        h1 = -torch.sum(p1 * torch.log(p1 + 1e-8), dim=1, keepdim=True)
        h2 = -torch.sum(p2 * torch.log(p2 + 1e-8), dim=1, keepdim=True)
        h3 = -torch.sum(p3 * torch.log(p3 + 1e-8), dim=1, keepdim=True)

        # Certainty score: exp(-entropy)
        c1 = torch.exp(-h1) * self.gain_ratios[0]
        c2 = torch.exp(-h2) * self.gain_ratios[1]
        c3 = torch.exp(-h3) * self.gain_ratios[2]

        total_certainty = c1 + c2 + c3 + 1e-8
        w1 = c1 / total_certainty
        w2 = c2 / total_certainty
        w3 = c3 / total_certainty

        fused_probs = w1 * p1 + w2 * p2 + w3 * p3
        return fused_probs


if __name__ == "__main__":
    ensemble = UAGRFEnsemble(num_classes=22, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    probs = ensemble(dummy_x)
    print("UAGRF Ensemble output shape:", probs.shape)
    assert probs.shape == (2, 22)
    print("UAGRF Ensemble forward pass successful!")
