import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(predictions, targets, num_classes=5, ignore_index=-1):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    ious = []
    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        ious.append(iou)

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    result = {'mean_iou': mean_iou}
    for i, iou in enumerate(ious):
        result[f'class_{i}_iou'] = iou if not np.isnan(iou) else 0.0

    return result


def compute_pixel_accuracy(predictions, targets, ignore_index=-1):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    valid_mask = targets != ignore_index
    correct = (predictions == targets) & valid_mask

    if valid_mask.sum() == 0:
        return 0.0

    return correct.sum() / valid_mask.sum()


def compute_confusion_matrix(predictions, targets, num_classes=5, ignore_index=-1):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    cm = confusion_matrix(targets, predictions,
                          labels=list(range(num_classes)))
    return cm


class MetricsTracker:
    def __init__(self, num_classes=5, ignore_index=-1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.total_iou = 0.0
        self.total_accuracy = 0.0
        self.num_batches = 0
        self.class_ious = [0.0] * self.num_classes

    def update(self, predictions, targets):
        iou_dict = compute_iou(predictions, targets,
                               self.num_classes, self.ignore_index)
        accuracy = compute_pixel_accuracy(
            predictions, targets, self.ignore_index)

        self.total_iou += iou_dict['mean_iou']
        self.total_accuracy += accuracy
        self.num_batches += 1

        for i in range(self.num_classes):
            self.class_ious[i] += iou_dict[f'class_{i}_iou']

    def get_metrics(self):
        if self.num_batches == 0:
            return {'mean_iou': 0.0, 'accuracy': 0.0}

        metrics = {
            'mean_iou': self.total_iou / self.num_batches,
            'accuracy': self.total_accuracy / self.num_batches
        }

        for i in range(self.num_classes):
            metrics[f'class_{i}_iou'] = self.class_ious[i] / self.num_batches

        return metrics
