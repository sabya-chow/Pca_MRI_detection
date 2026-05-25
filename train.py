"""Training loops with full checkpointing for SimCLR pre-training and CORAL fine-tuning."""
from __future__ import annotations
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import coral_loss, coral_predict
from src.simclr import NTXentLoss


def _ckpt_path(checkpoint_dir: str, name: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, name)


def train_simclr(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    save_path: str = "weights/simclr_encoder.pth",
    checkpoint_dir: str = "weights/checkpoints",
    resume: bool = True,
    log_every: int = 5,
) -> dict:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = NTXentLoss(temperature=0.07)
    model.to(device)

    history = {"loss": [], "lr": []}
    start_epoch = 1

    ckpt_file = _ckpt_path(checkpoint_dir, "simclr_ckpt.pth")
    if resume and os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"  Resumed SimCLR from epoch {ckpt['epoch']}  loss={history['loss'][-1]:.4f}")

    if start_epoch > num_epochs:
        print("SimCLR already completed — skipping.")
        return history

    print(f"SimCLR pre-training: epochs {start_epoch}-{num_epochs}  device={device}")
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for v1, v2 in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        current_lr = scheduler.get_last_lr()[0]
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)
        scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }, ckpt_file)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  ({time.time()-t0:.1f}s)")

    torch.save(model.encoder.state_dict(), save_path)
    print(f"Encoder saved -> {save_path}")
    return history


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    num_classes: int = 5,
    training: bool = True,
) -> tuple:
    model.train(training)
    total_loss = 0.0
    total_gnorm = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = coral_loss(out, y, num_classes)

            if training:
                optimizer.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                total_gnorm += gnorm
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(coral_predict(out.detach()).cpu())
            all_labels.append(y.cpu())

    n = max(len(loader), 1)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / n, total_gnorm / n if training else 0.0, preds, labels


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 40,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    save_path: str = "weights/classifier_best.pth",
    checkpoint_dir: str = "weights/checkpoints",
    resume: bool = True,
    num_classes: int = 5,
    log_every: int = 5,
    patience: int = 12,
) -> dict:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )
    model.to(device)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_mae":  [], "val_mae":  [],
        "lr": [], "grad_norm": [],
    }
    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 1

    ckpt_name = os.path.basename(save_path) + "_ckpt.pth"
    ckpt_file = _ckpt_path(checkpoint_dir, ckpt_name)
    if resume and os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        history       = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        no_improve    = ckpt.get("no_improve", 0)
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val={best_val_loss:.4f}")

    if start_epoch > num_epochs:
        print("Training already completed.")
        return history

    print(f"Fine-tuning: epochs {start_epoch}-{num_epochs}  lr={lr:.1e}  device={device}")
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_gn, tr_preds, tr_labels = _run_epoch(
            model, train_loader, optimizer, device, num_classes, training=True
        )
        vl_loss, _,    vl_preds, vl_labels = _run_epoch(
            model, val_loader, None, device, num_classes, training=False
        )

        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        tr_mae = float(np.abs(tr_preds - tr_labels).mean())
        vl_mae = float(np.abs(vl_preds - vl_labels).mean())
        tr_acc = float((tr_preds == tr_labels).mean())
        vl_acc = float((vl_preds == vl_labels).mean())

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)
        history["lr"].append(current_lr)
        history["grad_norm"].append(tr_gn)

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss = vl_loss
            no_improve    = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == num_epochs or improved:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "no_improve": no_improve,
            }, ckpt_file)

        if epoch % log_every == 0 or epoch == 1:
            mark = " *" if improved else ""
            print(
                f"  Ep {epoch:3d}/{num_epochs}  "
                f"tr_loss={tr_loss:.4f} acc={tr_acc:.3f} mae={tr_mae:.3f}  |  "
                f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} mae={vl_mae:.3f}  "
                f"lr={current_lr:.1e} gn={tr_gn:.2f}  ({time.time()-t0:.1f}s){mark}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"Best model -> {save_path}  (val_loss={best_val_loss:.4f})")
    return history
