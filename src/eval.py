import os
import json
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

from src.data import DataConfig, build_dataloaders
from src.model import build_efficientnet_b0
import seaborn as sns


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    return ckpt


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_targets, all_preds


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(16, 16))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def get_latest_model_path(models_dir="artifacts/models"):
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".pt")
    ]

    if not model_files:
        raise FileNotFoundError(f"No versioned model files found in: {models_dir}")

    latest_version = max(
        int(f.split("_v")[-1].split(".pt")[0])
        for f in model_files
    )

    return os.path.join(models_dir, f"model_v{latest_version}.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default="data/plantvillage/color")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = get_latest_model_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    class_map_path = "artifacts/reports/class_to_idx.json"
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(
            f"Missing {class_map_path}. Run training once or ensure this file exists."
        )

    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    cfg = DataConfig(
        base_dir=args.base_dir,
        img_size=224,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    _, _, test_loader, _ = build_dataloaders(cfg)

    model = build_efficientnet_b0(num_classes=len(class_to_idx), pretrained=False).to(device)

    ckpt = load_checkpoint(args.weights, device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred = run_eval(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("test accuracy:", acc)
    print("macro f1:", macro_f1)
    print("weighted f1:", weighted_f1)

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("artifacts/reports", exist_ok=True)

    with open("artifacts/reports/classification_report.txt", "w") as f:
        f.write(report_txt)

    plot_confusion_matrix(cm, class_names, "artifacts/reports/confusion_matrix.png")

    with open("artifacts/reports/metrics.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
            },
            f,
            indent=2,
        )

    print("Saved:")
    print("- artifacts/reports/classification_report.txt")
    print("- artifacts/reports/confusion_matrix.png")
    print("- artifacts/reports/metrics.json")


if __name__ == "__main__":
    main()