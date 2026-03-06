import os
import json
import argparse
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.model import build_efficientnet_b0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_inference_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_class_mapping(class_map_path: str) -> Tuple[dict, dict]:
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class map not found: {class_map_path}")

    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_model(
    weights_path: str,
    num_classes: int,
    device: str,
) -> torch.nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(img_size)
    x = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return x


@torch.no_grad()
def predict_topk(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    idx_to_class: dict,
    device: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, probs.shape[0])
    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        results.append((idx_to_class[idx], float(prob)))

    return results

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
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--class_map", type=str, default="artifacts/reports/class_to_idx.json")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    if args.weights is None:
        args.weights = get_latest_model_path()

    device = get_device()
    print("device:", device)

    class_to_idx, idx_to_class = load_class_mapping(args.class_map)
    model = load_model(
        weights_path=args.weights,
        num_classes=len(class_to_idx),
        device=device,
    )

    image_tensor = preprocess_image(args.image, img_size=args.img_size)
    predictions = predict_topk(
        model=model,
        image_tensor=image_tensor,
        idx_to_class=idx_to_class,
        device=device,
        top_k=args.top_k,
    )

    print(f"\nPredictions for: {args.image}")
    for rank, (label, prob) in enumerate(predictions, start=1):
        print(f"{rank}. {label} -> {prob:.4f}")

    print(f"\nTop-1 prediction: {predictions[0][0]}")


if __name__ == "__main__":
    main()