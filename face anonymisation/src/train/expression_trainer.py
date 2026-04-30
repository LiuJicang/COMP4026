import json
from pathlib import Path

import torch
from tqdm import tqdm


def _run_epoch(model, loader, criterion, optimizer, device: str, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_expression_teacher(
    model,
    train_loader,
    val_loader,
    class_to_idx: dict,
    cfg: dict,
    device: str,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    history = []
    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(summary)

        print(
            f"Epoch {epoch}/{cfg['train']['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_val_accuracy": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "history": history,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to: {best_path}")
