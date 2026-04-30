import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as TF

from src.models.expression_model import build_expression_model

class FrozenExpressionGuidance(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet18",
        num_classes: int = 7,
        input_size: int = 224,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.temperature = float(temperature)
        self.encoder = build_expression_model(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        real_logits = self.encoder(self._preprocess(real_images)).detach()
        fake_logits = self.encoder(self._preprocess(fake_images))

        target_distribution = F.softmax(real_logits / self.temperature, dim=1)
        fake_log_distribution = F.log_softmax(fake_logits / self.temperature, dim=1)
        loss = F.kl_div(fake_log_distribution, target_distribution, reduction="batchmean") * (self.temperature**2)

        real_pred = real_logits.argmax(dim=1)
        fake_pred = fake_logits.argmax(dim=1)
        metrics = {
            "expression_prediction_match": (real_pred == fake_pred).float().mean().detach().item(),
        }
        return loss, metrics

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = TF.resize(images, [self.input_size, self.input_size], antialias=True)
        images = (images * 0.5 + 0.5).clamp(0.0, 1.0)
        return (images - self.mean) / self.std
