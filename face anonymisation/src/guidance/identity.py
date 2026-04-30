import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as TF

from facenet_pytorch import InceptionResnetV1


class FaceNetIdentityGuidance(nn.Module):
    def __init__(
        self,
        pretrained: str = "vggface2",
        input_size: int = 160,
        target_similarity: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.target_similarity = float(target_similarity)
        self.encoder = InceptionResnetV1(pretrained=pretrained).eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        real_embeddings = self._encode(real_images).detach()
        fake_embeddings = self._encode(fake_images)

        cosine_similarity = F.cosine_similarity(real_embeddings, fake_embeddings, dim=1)
        identity_loss = F.relu(cosine_similarity - self.target_similarity).mean()

        metrics = {
            "identity_cosine_similarity": cosine_similarity.mean().detach().item(),
        }
        return identity_loss, metrics

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        embeddings = self.encoder(images)
        return F.normalize(embeddings, dim=1)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = TF.resize(images, [self.input_size, self.input_size], antialias=True)
        images = (images * 0.5 + 0.5).clamp(0.0, 1.0)
        images = images * 255.0
        return (images - 127.5) / 128.0
