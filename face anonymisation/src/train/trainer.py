import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.losses import total_variation_loss


def _sample_noise(batch_size: int, noise_channels: int, image_size: int, noise_std: float, device: str):
    return torch.randn((batch_size, noise_channels, image_size, image_size), device=device) * noise_std


def _adversarial_targets_like(prediction: torch.Tensor, value: float):
    return torch.full_like(prediction, fill_value=value)


def _run_train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    cfg,
    identity_guidance,
    expression_guidance,
    device: str,
):
    generator.train()
    discriminator.train()

    adv_weight = float(cfg["train"].get("adv_weight", 1.0))
    recon_weight = float(cfg["train"].get("recon_weight", 10.0))
    tv_weight = float(cfg["train"].get("tv_weight", 0.0001))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.9))
    noise_std = float(cfg["train"].get("noise_std", 1.0))
    image_size = int(cfg["data"]["image_size"])
    noise_channels = int(cfg["model"]["generator"].get("noise_channels", 1))

    metrics = {
        "generator_loss": 0.0,
        "discriminator_loss": 0.0,
        "adv_loss": 0.0,
        "recon_loss": 0.0,
        "tv_loss": 0.0,
        "identity_loss": 0.0,
        "expression_loss": 0.0,
        "identity_cosine_similarity": 0.0,
        "expression_prediction_match": 0.0,
    }

    for images in tqdm(loader, leave=False):
        real_images = images.to(device)
        noise = _sample_noise(real_images.size(0), noise_channels, image_size, noise_std, device)

        fake_images = generator(real_images, noise)

        optimizer_d.zero_grad()
        pred_real = discriminator(real_images)
        pred_fake = discriminator(fake_images.detach())

        loss_d_real = F.binary_cross_entropy_with_logits(
            pred_real, _adversarial_targets_like(pred_real, label_smoothing)
        )
        loss_d_fake = F.binary_cross_entropy_with_logits(
            pred_fake, _adversarial_targets_like(pred_fake, 0.0)
        )
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        pred_fake_for_g = discriminator(fake_images)
        loss_g_adv = F.binary_cross_entropy_with_logits(
            pred_fake_for_g, _adversarial_targets_like(pred_fake_for_g, 1.0)
        )
        loss_recon = F.l1_loss(fake_images, real_images)
        loss_tv = total_variation_loss(fake_images)
        loss_identity = torch.zeros((), device=device)
        loss_expression = torch.zeros((), device=device)
        identity_metrics = {"identity_cosine_similarity": 0.0}
        expression_metrics = {"expression_prediction_match": 0.0}

        if identity_guidance is not None:
            loss_identity, identity_metrics = identity_guidance(real_images, fake_images)

        if expression_guidance is not None:
            loss_expression, expression_metrics = expression_guidance(real_images, fake_images)

        loss_g = (
            adv_weight * loss_g_adv
            + recon_weight * loss_recon
            + tv_weight * loss_tv
            + float(cfg.get("guidance", {}).get("identity", {}).get("weight", 0.0)) * loss_identity
            + float(cfg.get("guidance", {}).get("expression", {}).get("weight", 0.0)) * loss_expression
        )
        loss_g.backward()
        optimizer_g.step()

        batch_size = real_images.size(0)
        metrics["generator_loss"] += loss_g.item() * batch_size
        metrics["discriminator_loss"] += loss_d.item() * batch_size
        metrics["adv_loss"] += loss_g_adv.item() * batch_size
        metrics["recon_loss"] += loss_recon.item() * batch_size
        metrics["tv_loss"] += loss_tv.item() * batch_size
        metrics["identity_loss"] += loss_identity.item() * batch_size
        metrics["expression_loss"] += loss_expression.item() * batch_size
        metrics["identity_cosine_similarity"] += identity_metrics["identity_cosine_similarity"] * batch_size
        metrics["expression_prediction_match"] += expression_metrics["expression_prediction_match"] * batch_size

    return {key: value / max(len(loader.dataset), 1) for key, value in metrics.items()}


@torch.no_grad()
def _run_val_epoch(generator, loader, cfg, identity_guidance, expression_guidance, device: str):
    generator.eval()

    noise_std = float(cfg["train"].get("noise_std", 1.0))
    image_size = int(cfg["data"]["image_size"])
    noise_channels = int(cfg["model"]["generator"].get("noise_channels", 1))

    total_recon = 0.0
    total_identity_loss = 0.0
    total_expression_loss = 0.0
    total_identity_similarity = 0.0
    total_expression_match = 0.0
    preview_batch = None

    for images in tqdm(loader, leave=False):
        real_images = images.to(device)
        noise = _sample_noise(real_images.size(0), noise_channels, image_size, noise_std, device)
        fake_images = generator(real_images, noise)
        total_recon += F.l1_loss(fake_images, real_images, reduction="mean").item() * real_images.size(0)

        if identity_guidance is not None:
            identity_loss, identity_metrics = identity_guidance(real_images, fake_images)
            total_identity_loss += identity_loss.item() * real_images.size(0)
            total_identity_similarity += identity_metrics["identity_cosine_similarity"] * real_images.size(0)

        if expression_guidance is not None:
            expression_loss, expression_metrics = expression_guidance(real_images, fake_images)
            total_expression_loss += expression_loss.item() * real_images.size(0)
            total_expression_match += expression_metrics["expression_prediction_match"] * real_images.size(0)

        if preview_batch is None:
            preview_batch = (real_images[:], fake_images[:])

    avg_recon = total_recon / max(len(loader.dataset), 1)
    dataset_size = max(len(loader.dataset), 1)
    return {
        "recon_l1": avg_recon,
        "identity_loss": total_identity_loss / dataset_size,
        "expression_loss": total_expression_loss / dataset_size,
        "identity_cosine_similarity": total_identity_similarity / dataset_size,
        "expression_prediction_match": total_expression_match / dataset_size,
    }, preview_batch


def _save_preview(preview_batch, output_path: Path, max_samples: int):
    if preview_batch is None:
        return

    real_images, fake_images = preview_batch
    count = min(max_samples, real_images.size(0), fake_images.size(0))
    if count <= 0:
        return

    paired = []
    for idx in range(count):
        paired.extend([real_images[idx], fake_images[idx]])

    grid = make_grid(paired, nrow=2, normalize=True, value_range=(-1, 1))
    save_image(grid, output_path)


def train_gan_model(
    generator,
    discriminator,
    train_loader,
    val_loader,
    cfg,
    identity_guidance,
    expression_guidance,
    device: str,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    betas = tuple(float(value) for value in cfg["train"].get("betas", [0.5, 0.999]))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg["train"]["lr_generator"], betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["train"]["lr_discriminator"], betas=betas)

    history = []
    best_val_recon = float("inf")
    best_generator_path = output_dir / "best_generator.pt"
    last_generator_path = output_dir / "last_generator.pt"
    last_discriminator_path = output_dir / "last_discriminator.pt"
    preview_samples = int(cfg["train"].get("preview_samples", 6))

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = _run_train_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            cfg=cfg,
            identity_guidance=identity_guidance,
            expression_guidance=expression_guidance,
            device=device,
        )
        val_metrics, preview_batch = _run_val_epoch(
            generator=generator,
            loader=val_loader,
            cfg=cfg,
            identity_guidance=identity_guidance,
            expression_guidance=expression_guidance,
            device=device,
        )

        epoch_summary = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch}/{cfg['train']['epochs']} | "
            f"G={train_metrics['generator_loss']:.4f} "
            f"D={train_metrics['discriminator_loss']:.4f} "
            f"Recon={val_metrics['recon_l1']:.4f} "
            f"IDsim={val_metrics['identity_cosine_similarity']:.4f} "
            f"ExprMatch={val_metrics['expression_prediction_match']:.4f}"
        )

        torch.save(generator.state_dict(), last_generator_path)
        torch.save(discriminator.state_dict(), last_discriminator_path)

        preview_path = output_dir / f"preview_epoch_{epoch:03d}.png"
        _save_preview(preview_batch, preview_path, preview_samples)

        if val_metrics["recon_l1"] < best_val_recon:
            best_val_recon = val_metrics["recon_l1"]
            torch.save(generator.state_dict(), best_generator_path)

        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_val_recon_l1": best_val_recon,
                    "epochs_completed": epoch,
                    "history": history,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Best validation reconstruction L1: {best_val_recon:.4f}")
    print(f"Saved best generator to: {best_generator_path}")
