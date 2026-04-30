from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.datasets import FaceFolderDataset
from src.data.torch_datasets import ResNetFaceDataset
from src.models.resnet_recognizer import ResNet18Recognizer
from src.preprocess.detect_and_crop import FacePreprocessor


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_train_epoch(recognizer: ResNet18Recognizer, loader, optimizer, device: str):
    recognizer.network.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = recognizer.network(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def _compute_prototypes(recognizer: ResNet18Recognizer, loader, device: str):
    feature_dim = recognizer.network.feature_dim
    prototype_sums = torch.zeros(recognizer.num_classes, feature_dim, dtype=torch.float32, device=device)
    counts = torch.zeros(recognizer.num_classes, dtype=torch.long, device=device)

    recognizer.network.eval()
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, embeddings = recognizer.network(images)
        embeddings = F.normalize(embeddings, dim=1)

        for label in labels.unique():
            mask = labels == label
            prototype_sums[label] += embeddings[mask].sum(dim=0)
            counts[label] += mask.sum()

    counts = counts.clamp_min(1).unsqueeze(1)
    prototypes = prototype_sums / counts
    return F.normalize(prototypes, dim=1).cpu()


@torch.no_grad()
def _collect_embedding_distances(recognizer: ResNet18Recognizer, loader, prototypes: torch.Tensor, device: str):
    genuine_distances = []
    nearest_impostor_distances = []

    recognizer.network.eval()
    prototypes = prototypes.to(device)

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, embeddings = recognizer.network(images)
        embeddings = F.normalize(embeddings, dim=1)
        distances = 1.0 - embeddings @ prototypes.T

        genuine = distances.gather(1, labels.unsqueeze(1)).squeeze(1)
        genuine_distances.extend(genuine.cpu().tolist())

        masked = distances.clone()
        masked.scatter_(1, labels.unsqueeze(1), float("inf"))
        nearest_impostor = masked.min(dim=1).values
        nearest_impostor_distances.extend(nearest_impostor.cpu().tolist())

    return {
        "genuine": genuine_distances,
        "nearest_impostor": nearest_impostor_distances,
    }


def _recommend_threshold(model_cfg: dict, distance_stats: dict):
    genuine = torch.tensor(distance_stats["genuine"], dtype=torch.float32)
    impostor = torch.tensor(distance_stats["nearest_impostor"], dtype=torch.float32)

    percentile = float(model_cfg.get("threshold_percentile", 95.0))
    threshold_min = float(model_cfg.get("threshold_min", 0.0))
    threshold_max = float(model_cfg.get("threshold_max", 2.0))

    genuine_percentile = float(torch.quantile(genuine, percentile / 100.0).item())
    impostor_percentile = float(torch.quantile(impostor, 0.05).item()) if len(impostor) else genuine_percentile

    if genuine_percentile < impostor_percentile:
        recommended = (genuine_percentile + impostor_percentile) / 2.0
        strategy = "midpoint_between_genuine_pctl_and_impostor_p05"
    else:
        recommended = genuine_percentile
        strategy = "genuine_percentile_fallback"

    recommended = max(threshold_min, min(threshold_max, recommended))

    return {
        "recommended_threshold": recommended,
        "strategy": strategy,
        "genuine_percentile": genuine_percentile,
        "impostor_p05": impostor_percentile,
        "percentile": percentile,
        "genuine_mean": float(genuine.mean().item()),
        "genuine_min": float(genuine.min().item()),
        "genuine_max": float(genuine.max().item()),
        "impostor_mean": float(impostor.mean().item()) if len(impostor) else None,
        "impostor_min": float(impostor.min().item()) if len(impostor) else None,
        "impostor_max": float(impostor.max().item()) if len(impostor) else None,
    }


def train_resnet18_from_config(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    folder_dataset = FaceFolderDataset(
        image_root=data_cfg["image_root"],
        split_dir=data_cfg["train_dir"],
    )
    if len(folder_dataset) == 0:
        raise ValueError("Training folder is empty. Please put images under data/train/<person_name>/ before training.")

    preprocessor = FacePreprocessor.from_config(cfg["preprocess"])
    tensor_dataset = ResNetFaceDataset(folder_dataset, preprocessor)

    train_loader = _build_loader(
        tensor_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    prototype_loader = _build_loader(
        tensor_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    recognizer = ResNet18Recognizer(
        image_size=tuple(cfg["preprocess"]["image_size"]),
        num_classes=len(folder_dataset.label_to_name),
        threshold=model_cfg["threshold"],
        backbone=model_cfg.get("backbone", "resnet18"),
        pretrained=model_cfg.get("pretrained", True),
        distance_metric=model_cfg.get("distance_metric", "cosine"),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        recognizer.network.parameters(),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    history = []
    epochs = int(train_cfg.get("epochs", 5))
    for epoch in range(1, epochs + 1):
        metrics = _run_train_epoch(recognizer, train_loader, optimizer, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}"
        )

    prototypes = _compute_prototypes(recognizer, prototype_loader, device)
    recognizer.set_prototypes(prototypes, folder_dataset.label_to_name)

    threshold_mode = model_cfg.get("threshold_mode", "manual").lower()
    threshold_summary = {
        "mode": threshold_mode,
        "configured_threshold": float(model_cfg["threshold"]),
    }
    if threshold_mode == "auto":
        distance_stats = _collect_embedding_distances(recognizer, prototype_loader, prototypes, device)
        recommendation = _recommend_threshold(model_cfg, distance_stats)
        recognizer.threshold = recommendation["recommended_threshold"]
        recognizer.threshold_source = "auto"
        threshold_summary.update(recommendation)
    else:
        recognizer.threshold = float(model_cfg["threshold"])
        recognizer.threshold_source = "manual"
        threshold_summary["recommended_threshold"] = recognizer.threshold

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    recognizer.save(output_dir, history=history, label_to_name=folder_dataset.label_to_name)

    summary = {
        "recognizer": recognizer.model_name,
        "backbone": recognizer.backbone_name,
        "device": device,
        "train_dir": str(folder_dataset.split_dir),
        "train_samples_seen": len(folder_dataset),
        "train_samples_used": len(folder_dataset),
        "train_samples_skipped": 0,
        "labels": recognizer.label_to_name,
        "threshold": recognizer.threshold,
        "threshold_source": recognizer.threshold_source,
        "threshold_summary": threshold_summary,
        "distance_metric": recognizer.distance_metric,
        "image_size": list(recognizer.image_size),
        "cascade_path": str(preprocessor.cascade_path) if preprocessor.cascade_path else "",
        "fallback_mode": preprocessor.fallback_mode,
        "history": history,
    }
    return output_dir, summary
