import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EstimationLoss(nn.Module):
    def __init__(self, cfg):
        super(EstimationLoss, self).__init__()
        self.weights = torch.from_numpy(np.load(cfg.training_cfg.weights_dir)).cuda().float()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target, weight=self.weights)
        return total_loss


class ContrastiveLoss(nn.Module):
    def __init__(self):
        """
        Contrastive loss for training a model using CLIP-style loss.
        Args:
            cfg: Configuration object containing training parameters.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # Learnable temperature parameter

    def forward(self, similarity_logits):
        """
        Compute the contrastive loss.
        Args:
            similarity_logits (torch.Tensor): The similarity logits (B x N), 
                                              where B is the batch size and N is the number of classes or samples.
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Get batch size
        batch_size = similarity_logits.size(0)
        
        # Target labels (diagonal matches)
        target = torch.arange(batch_size, device=similarity_logits.device)

        # print(f'the shape of similarity_logits is {similarity_logits.shape}')
        # print(f'target is {target}')
        # print(f'the shape of target is {target.shape}')

        # raise ValueError("With Great Power Comes Great Responsibility")
        
        # Symmetric cross-entropy loss
        loss_image_to_text = F.cross_entropy(similarity_logits, target)
        loss_text_to_image = F.cross_entropy(similarity_logits.t(), target)
        
        # Final symmetric contrastive loss
        total_loss = (loss_image_to_text + loss_text_to_image) / 2
        
        return total_loss