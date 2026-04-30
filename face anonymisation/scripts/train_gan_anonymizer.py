import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.data.datasets import UnlabeledFaceImageDataset
from src.data.transforms import build_image_transform
from src.guidance.expression import FrozenExpressionGuidance
from src.guidance.identity import FaceNetIdentityGuidance
from src.models.discriminator import PatchDiscriminator
from src.models.generator import UNetAnonymizer
from src.train.trainer import train_gan_model


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_data_path(image_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = image_root / path
    return path.resolve()


def maybe_load_checkpoint(module, checkpoint_value: str | None, repo_root: Path, label: str):
    if not checkpoint_value:
        return

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    module.load_state_dict(state_dict)
    print(f"Loaded {label} checkpoint: {checkpoint_path}")


def maybe_limit_dataset(dataset, limit: int | None, label: str):
    if limit is None:
        return dataset
    limit = int(limit)
    if limit <= 0 or limit >= len(dataset):
        return dataset
    print(f"Using a limited {label} subset: {limit}/{len(dataset)} samples")
    return Subset(dataset, range(limit))


def build_guidance_modules(cfg: dict, image_root: Path, repo_root: Path):
    guidance_cfg = cfg.get("guidance", {})

    identity_guidance = None
    identity_cfg = guidance_cfg.get("identity", {})
    if identity_cfg.get("enabled", False):
        identity_guidance = FaceNetIdentityGuidance(
            pretrained=identity_cfg.get("pretrained", "vggface2"),
            input_size=identity_cfg.get("input_size", 160),
            target_similarity=identity_cfg.get("target_similarity", 0.25),
        )

    expression_guidance = None
    expression_cfg = guidance_cfg.get("expression", {})
    if expression_cfg.get("enabled", False):
        checkpoint = expression_cfg.get("checkpoint", "")
        if not checkpoint:
            raise ValueError("guidance.expression.enabled is true but no checkpoint was provided.")

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expression guidance checkpoint was not found: {checkpoint_path}"
            )

        expression_guidance = FrozenExpressionGuidance(
            checkpoint_path=str(checkpoint_path),
            backbone=expression_cfg.get("backbone", "resnet18"),
            num_classes=expression_cfg.get("num_classes", 7),
            input_size=expression_cfg.get("input_size", 224),
            temperature=expression_cfg.get("temperature", 2.0),
        )

    return identity_guidance, expression_guidance


def main():
    parser = argparse.ArgumentParser(description="Train the GAN-style face anonymisation prototype")
    parser.add_argument("--config", type=str, default="face anonymisation/configs/gan_baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(resolve_repo_path(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pin_memory = device == "cuda"

    train_transform = build_image_transform(cfg["data"]["image_size"], is_train=True)
    val_transform = build_image_transform(cfg["data"]["image_size"], is_train=False)

    image_root = resolve_repo_path(cfg["data"].get("image_root", "."))
    train_dir = resolve_data_path(image_root, cfg["data"]["train_dir"])
    val_dir = resolve_data_path(image_root, cfg["data"]["val_dir"])

    train_dataset = UnlabeledFaceImageDataset(
        image_root=image_root,
        source_dir=train_dir,
        transform=train_transform,
        crop_cfg=cfg["data"].get("crop", {}),
        detection_cfg=cfg.get("detection", {}),
        is_train=True,
        supported_extensions=cfg["data"].get("supported_extensions", []),
    )
    val_dataset = UnlabeledFaceImageDataset(
        image_root=image_root,
        source_dir=val_dir,
        transform=val_transform,
        crop_cfg=cfg["data"].get("crop", {}),
        detection_cfg=cfg.get("detection", {}),
        is_train=False,
        supported_extensions=cfg["data"].get("supported_extensions", []),
    )
    train_dataset = maybe_limit_dataset(
        train_dataset,
        cfg["data"].get("max_train_images"),
        label="training",
    )
    val_dataset = maybe_limit_dataset(
        val_dataset,
        cfg["data"].get("max_val_images"),
        label="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )

    generator_cfg = cfg["model"]["generator"]
    discriminator_cfg = cfg["model"]["discriminator"]

    generator = UNetAnonymizer(
        in_channels=3,
        out_channels=3,
        base_channels=generator_cfg.get("base_channels", 64),
        noise_channels=generator_cfg.get("noise_channels", 1),
        bottleneck_dropout=generator_cfg.get("bottleneck_dropout", 0.2),
        bottleneck_blocks=generator_cfg.get("bottleneck_blocks", 0),
        residual_output=generator_cfg.get("residual_output", False),
        residual_scale=generator_cfg.get("residual_scale", 0.5),
    ).to(device)
    discriminator = PatchDiscriminator(
        in_channels=3,
        base_channels=discriminator_cfg.get("base_channels", 64),
    ).to(device)
    maybe_load_checkpoint(
        generator,
        cfg["train"].get("generator_checkpoint"),
        REPO_ROOT,
        label="generator",
    )
    maybe_load_checkpoint(
        discriminator,
        cfg["train"].get("discriminator_checkpoint"),
        REPO_ROOT,
        label="discriminator",
    )

    identity_guidance, expression_guidance = build_guidance_modules(cfg, image_root, REPO_ROOT)
    if identity_guidance is not None:
        identity_guidance = identity_guidance.to(device)
    if expression_guidance is not None:
        expression_guidance = expression_guidance.to(device)

    output_dir = resolve_repo_path(cfg["train"]["output_dir"])
    train_gan_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        identity_guidance=identity_guidance,
        expression_guidance=expression_guidance,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
