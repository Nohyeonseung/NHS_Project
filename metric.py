# metric
import torch
import numpy as np

def dice_score(preds, targets, num_classes, ignore_index=0):
    smooth = 1e-6
    dice_scores = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        denominator = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_scores.append(dice.item())

    return np.mean(dice_scores) if dice_scores else 0.0

def per_class_dice(preds, targets, num_classes):
    results = {}
    smooth = 1e-6

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        denominator = pred_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        results[f'class_{cls}'] = dice.item()

    return results
def calculate_metrics(preds, targets, num_classes, ignore_index=0):
    metrics = {
        'dice': [],
        'iou': [],
        'sensitivity': [],
        'specificity': []
    }

    pixel_accuracy = (preds == targets).float().mean().item()

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        # True Positives, False Positives, etc.
        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        tn = ((1 - pred_cls) * (1 - target_cls)).sum()
        
        smooth = 1e-6

        # Dice Score
        dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
        metrics['dice'].append(dice.item())
        
        # IoU (Jaccard Index)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        metrics['iou'].append(iou.item())

        # Sensitivity (Recall)
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        metrics['sensitivity'].append(sensitivity.item())

        # Specificity
        specificity = (tn + smooth) / (tn + fp + smooth)
        metrics['specificity'].append(specificity.item())

    mean_metrics = {key: np.mean(values) if values else 0.0 for key, values in metrics.items()}
    mean_metrics['accuracy'] = pixel_accuracy
    
    return mean_metrics
