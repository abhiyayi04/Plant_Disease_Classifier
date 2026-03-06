import os
import json
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score

import wandb

from src.data import DataConfig, build_dataloaders, seed_everything
from src.model import (
    build_efficientnet_b0,
    freeze_backbone,
    unfreeze_last_n_blocks,
    get_trainable_params,
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    running_correct = 0
    running_total = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

        running_correct += (preds == y).sum().item()
        running_total += y.size(0)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = running_correct / max(1, running_total)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return acc, macro_f1


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x, y in tqdm(train_loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == y).sum().item()
        running_total += y.size(0)

    avg_loss = running_loss / max(1, running_total)
    acc = running_correct / max(1, running_total)
    return avg_loss, acc


def save_checkpoint(path, model, optimizer, cfg, epoch, best_metric, metric_name="macro_f1", extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "extra": extra or {},
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data/plantvillage/color")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--epochs_head", type=int, default=2)
    parser.add_argument("--epochs_ft", type=int, default=5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_ft", type=float, default=1e-4)
    parser.add_argument("--unfreeze_blocks", type=int, default=2)

    parser.add_argument("--project", type=str, default="plant-disease-level3")
    args = parser.parse_args()

    cfg = DataConfig(
        base_dir=args.base_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    seed_everything(cfg.seed)
    device = get_device()
    print("device:", device)

    train_loader, val_loader, _, class_to_idx = build_dataloaders(cfg)
    num_classes = len(class_to_idx)

    os.makedirs("artifacts/reports", exist_ok=True)
    with open("artifacts/reports/class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    wandb.init(
        project=args.project,
        config={**asdict(cfg), **vars(args), "num_classes": num_classes, "device": device},
    )

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    models_dir = "artifacts/models"
    os.makedirs(models_dir, exist_ok=True)

    existing_versions = [
        int(f.split("_v")[-1].split(".pt")[0])
        for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".pt")
    ]

    next_version = max(existing_versions, default=0) + 1
    best_path = os.path.join(models_dir, f"model_v{next_version}.pt")
    global_epoch = 0

    # --------------------
    # Phase 1: Train head
    # --------------------
    freeze_backbone(model)
    print("Phase 1 trainable params:", sum(p.numel() for p in get_trainable_params(model)))

    optimizer = Adam(get_trainable_params(model), lr=args.lr_head)

    for epoch in range(1, args.epochs_head + 1):
        global_epoch += 1

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        wandb.log({
            "phase": "head",
            "epoch": global_epoch,
            "phase_epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"[HEAD] epoch {epoch} (global {global_epoch}) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=global_epoch,
                best_metric=best_f1,
                metric_name="macro_f1",
                extra={
                    "phase": "head",
                    "phase_epoch": epoch,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "num_classes": num_classes,
                },
            )
            print("Saved new best:", best_path)

    # -------------------------
    # Phase 2: Fine-tune last N
    # -------------------------
    unfreeze_last_n_blocks(model, n=args.unfreeze_blocks)
    print("Phase 2 trainable params:", sum(p.numel() for p in get_trainable_params(model)))

    optimizer = Adam(get_trainable_params(model), lr=args.lr_ft)

    for epoch in range(1, args.epochs_ft + 1):
        global_epoch += 1

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        wandb.log({
            "phase": "finetune",
            "epoch": global_epoch,
            "phase_epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        print(
            f"[FT] epoch {epoch} (global {global_epoch}) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=global_epoch,
                best_metric=best_f1,
                metric_name="macro_f1",
                extra={
                    "phase": "finetune",
                    "phase_epoch": epoch,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "num_classes": num_classes,
                },
            )
            print("Saved new best:", best_path)

    print("Training done. Best macro F1:", best_f1)
    wandb.finish()


if __name__ == "__main__":
    main()