import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass(frozen=True)
class Prediction:
    predicted_label: int | None
    predicted_name: str
    confidence: float
    is_unknown: bool


class ResNetIdentityModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            builder = models.resnet18
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            builder = models.resnet50
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        try:
            base_model = builder(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Failed to load pretrained {backbone} weights ({exc}). Falling back to random initialization."
            )
            base_model = builder(weights=None)

        self.feature_dim = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.backbone = base_model
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, images: torch.Tensor):
        embeddings = self.backbone(images)
        logits = self.classifier(embeddings)
        return logits, embeddings


class ResNet18Recognizer:
    model_name = "resnet18"
    checkpoint_filename = "resnet18_checkpoint.pt"

    def __init__(
        self,
        image_size: tuple[int, int],
        num_classes: int,
        threshold: float,
        backbone: str = "resnet18",
        pretrained: bool = True,
        distance_metric: str = "cosine",
        device: str = "cpu",
    ) -> None:
        self.image_size = tuple(int(v) for v in image_size)
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.backbone_name = backbone
        self.pretrained = bool(pretrained)
        self.distance_metric = distance_metric
        self.device = device
        self.label_to_name: dict[int, str] = {}
        self.network = ResNetIdentityModel(
            backbone=self.backbone_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        ).to(self.device)
        self.prototypes: torch.Tensor | None = None
        self.threshold_source = "manual"

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            _, embeddings = self.network(images.to(self.device))
            return F.normalize(embeddings, dim=1)

    def infer_batch(self, images: torch.Tensor) -> list[Prediction]:
        if self.prototypes is None:
            raise RuntimeError("Prototypes are missing. Please compute or load prototypes before inference.")

        embeddings = self.extract_embeddings(images)
        distances = self._compute_distances(embeddings, self.prototypes.to(self.device))
        nearest_distance, nearest_label = distances.min(dim=1)

        predictions = []
        for label_idx, distance in zip(nearest_label.cpu().tolist(), nearest_distance.cpu().tolist()):
            is_unknown = float(distance) > self.threshold
            if is_unknown:
                predictions.append(
                    Prediction(
                        predicted_label=None,
                        predicted_name="unknown",
                        confidence=float(distance),
                        is_unknown=True,
                    )
                )
            else:
                predictions.append(
                    Prediction(
                        predicted_label=int(label_idx),
                        predicted_name=self.label_to_name[int(label_idx)],
                        confidence=float(distance),
                        is_unknown=False,
                    )
                )
        return predictions

    def set_prototypes(self, prototypes: torch.Tensor, label_to_name: dict[int, str]) -> None:
        self.prototypes = F.normalize(prototypes.float(), dim=1).cpu()
        self.label_to_name = {int(key): str(value) for key, value in label_to_name.items()}

    def save(self, output_dir: str | Path, history: list[dict], label_to_name: dict[int, str]) -> None:
        if self.prototypes is None:
            raise RuntimeError("Prototypes are missing. Cannot save model without prototypes.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "network_state_dict": self.network.state_dict(),
            "prototypes": self.prototypes,
            "label_to_name": {int(key): str(value) for key, value in label_to_name.items()},
        }
        torch.save(payload, output_dir / self.checkpoint_filename)

        with (output_dir / "label_map.json").open("w", encoding="utf-8") as file:
            json.dump(label_to_name, file, indent=2, ensure_ascii=False)

        metadata = {
            "model_name": self.model_name,
            "backbone": self.backbone_name,
            "image_size": list(self.image_size),
            "num_classes": self.num_classes,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "distance_metric": self.distance_metric,
        }
        with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, model_dir: str | Path, device: str) -> "ResNet18Recognizer":
        model_dir = Path(model_dir)
        with (model_dir / "model_metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        recognizer = cls(
            image_size=tuple(metadata["image_size"]),
            num_classes=int(metadata["num_classes"]),
            threshold=float(metadata["threshold"]),
            backbone=metadata.get("backbone", "resnet18"),
            pretrained=False,
            distance_metric=metadata.get("distance_metric", "cosine"),
            device=device,
        )
        payload = torch.load(model_dir / cls.checkpoint_filename, map_location=device, weights_only=False)
        recognizer.network.load_state_dict(payload["network_state_dict"])
        recognizer.network.to(device)
        recognizer.set_prototypes(payload["prototypes"], payload["label_to_name"])
        recognizer.threshold_source = metadata.get("threshold_source", "manual")
        return recognizer

    def _compute_distances(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return 1.0 - embeddings @ prototypes.T
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
