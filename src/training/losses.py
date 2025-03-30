import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2)
        
        intersection = torch.sum(pred * target, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + (1 - self.dice_weight) * ce

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Example usage:
# loss_fn = CombinedLoss(dice_weight=0.5)
# loss = loss_fn(pred, target) 