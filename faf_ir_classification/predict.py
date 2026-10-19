"""Predict the retinal image modality (FAF / IR / OTHER) of one or more images.

Usage:
    python predict.py IMAGE_OR_FOLDER [IMAGE_OR_FOLDER ...] [--checkpoint PATH] [--out predictions.csv]

Folders are searched recursively. Preprocessing (image size, circle mask) is read from the
checkpoint, so it always matches how the model was trained.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "model" / "final_model_pretrained.pt"
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ResizePad:
    """Keep the aspect ratio: scale the longer side to `size`, then pad to a black square."""
    def __init__(self, size, fill=0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = self.size / max(image.width, image.height)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


class CircleMask:
    """Black out everything outside the centred circle."""
    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def __call__(self, image):
        a = np.array(image)
        h, w = a.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        r = min(h, w) / 2 * self.scale
        a[(yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 > r ** 2] = 0
        return Image.fromarray(a)


def build_model(model_name, n_classes):
    model = getattr(models, model_name)(weights=None)
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif model_name.startswith("efficientnet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


class ModalityClassifier:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model_name = checkpoint["model_name"]
        self.classes = checkpoint["classes"]
        self.image_size = checkpoint["image_size"]
        self.circle_mask = checkpoint.get("circle_mask", False)

        self.model = build_model(self.model_name, len(self.classes))
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()})
        self.model.to(self.device).eval()

        # what the model "sees" (also useful for display)
        self.model_view = transforms.Compose(
            [ResizePad(self.image_size)]
            + ([CircleMask(checkpoint.get("circle_scale", 1.0))] if self.circle_mask else [])
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        with Image.open(image_path) as im:
            view = self.model_view(im.convert("RGB"))
        x = self.to_tensor(view).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(x), dim=1)[0].cpu()
        i = probabilities.argmax().item()
        return {
            "predicted_label": self.classes[i],
            "confidence": probabilities[i].item(),
            "probabilities": {c: probabilities[j].item() for j, c in enumerate(self.classes)},
            "image": view,
        }


def find_images(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {p}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Predict FAF / IR / OTHER for retinal images.")
    parser.add_argument("paths", nargs="+", help="image files and/or folders")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--out", default="predictions.csv", type=Path, help="CSV file for the results")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    clf = ModalityClassifier(args.checkpoint, args.device)
    print(f"Model: {clf.model_name} | classes: {clf.classes} | image size: {clf.image_size} "
          f"| circle mask: {clf.circle_mask} | device: {clf.device}")

    files = find_images(args.paths)
    if not files:
        raise SystemExit("No images found.")

    rows = []
    for f in files:
        r = clf.predict(f)
        rows.append({"image": str(f), "prediction": r["predicted_label"], "confidence": round(r["confidence"], 4),
                     **{f"p_{c}": round(p, 4) for c, p in r["probabilities"].items()}})
    df = pd.DataFrame(rows)

    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 60):
        print(df.to_string(index=False))
    print("\n" + df["prediction"].value_counts().to_string())
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
