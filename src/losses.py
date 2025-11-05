import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross Entropy Loss implemented from scratch.

    Computes cross-entropy loss only on valid (labeled) pixels,
    ignoring pixels marked with ignore_index.

    Mathematical formulation:
        L = -1/|V| * Σ_{i∈V} w[y_i] * log(softmax(z_i)[y_i])

    where:
        V = set of valid pixels (targets != ignore_index)
        |V| = number of valid pixels
        z_i = logits at pixel i
        y_i = ground truth class at pixel i
        w[y_i] = class weight for class y_i (optional)
    """

    def __init__(self, ignore_index=-1, reduction='mean', weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs (unnormalized)
            targets: (B, H, W) - ground truth labels, with ignore_index for unlabeled pixels

        Returns:
            loss: scalar tensor if reduction='mean' or 'sum', otherwise (B, H, W)
        """
        # Identify valid (labeled) pixels
        valid_mask = (targets != self.ignore_index)  # (B, H, W)
        num_valid = valid_mask.sum()

        # Handle edge case: no valid pixels
        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Get dimensions
        B, C, H, W = logits.shape

        # Reshape logits and targets for easier indexing
        # logits: (B, C, H, W) -> (B*H*W, C)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        valid_mask_flat = valid_mask.reshape(-1)  # (B*H*W,)

        # Filter only valid pixels
        valid_logits = logits_flat[valid_mask_flat]  # (num_valid, C)
        valid_targets = targets_flat[valid_mask_flat]  # (num_valid,)

        # Compute log softmax: log(softmax(z_i)) = log(exp(z_i) / Σ exp(z_j))
        #                                        = z_i - log(Σ exp(z_j))
        log_softmax = F.log_softmax(valid_logits, dim=1)  # (num_valid, C)

        # Gather the log probabilities for the correct classes
        # For each pixel i, we want log_softmax[i, y_i]
        nll_loss = - \
            log_softmax[torch.arange(
                num_valid, device=logits.device), valid_targets]
        # nll_loss: (num_valid,) - negative log likelihood for each valid pixel

        # Apply class weights if provided
        if self.weight is not None:
            # Get weights for each valid target
            weights = self.weight[valid_targets]  # (num_valid,)
            nll_loss = nll_loss * weights

        # Apply reduction
        if self.reduction == 'mean':
            return nll_loss.mean()  # Average over valid pixels
        elif self.reduction == 'sum':
            return nll_loss.sum()
        elif self.reduction == 'none':
            # Reshape back to original spatial dimensions
            loss_map = torch.zeros(B * H * W, device=logits.device)
            loss_map[valid_mask_flat] = nll_loss
            return loss_map.reshape(B, H, W)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
