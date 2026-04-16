from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.metrics import classification_metrics


def _run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true, all_pred = [], []

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

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    metrics = classification_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    device,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best checkpoint to: {best_path}")
