import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(probs * targets_one_hot, dim=(2,3,4))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2,3,4))

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        # NaN 방지: 해당 클래스가 아예 없을 때는 dice_score=1로 처리
        dice_score = torch.where(cardinality == 0, torch.ones_like(dice_score), dice_score)

        return 1. - dice_score.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, ignore_index=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0,4,1,2,3).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        tp = torch.sum(probs * targets_one_hot, dim=(2,3,4))
        fp = torch.sum(probs * (1 - targets_one_hot), dim=(2,3,4))
        fn = torch.sum((1-probs) * targets_one_hot, dim=(2,3,4))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1. - tversky_index.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets.long(),
            reduction='none',
            ignore_index=self.ignore_index if self.ignore_index is not None else -100
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceCELoss(nn.Module):
    def __init__(self, w_dice=0.5, w_ce=0.5, ignore_index=None):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        ce_ignore = ignore_index if ignore_index is not None else -100
        self.ce = nn.CrossEntropyLoss(ignore_index=ce_ignore)
        self.w_dice = w_dice
        self.w_ce = w_ce

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets.long())
        return self.w_dice * dice_loss + self.w_ce * ce_loss


class DiceTverskyLoss(nn.Module):
    def __init__(self, w_dice=0.5, w_tversky=0.5, ignore_index=None):
        super(DiceTverskyLoss, self).__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.tversky = TverskyLoss(ignore_index=ignore_index)
        self.w_dice = w_dice
        self.w_tversky = w_tversky

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return self.w_dice * dice_loss + self.w_tversky * tversky_loss
