import torch


def total_variation_loss(images: torch.Tensor) -> torch.Tensor:
    loss_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
    loss_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
    return loss_h + loss_w
