import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


def build_model(arch: str, num_classes: int):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch in ["efficientnet_b0", "effnet_b0"]:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/split")
    parser.add_argument("--ckpt", type=str, default="models/model.pt")
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    num_classes = len(labels)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt["arch"]
    img_size = ckpt["img_size"]

    model = build_model(arch, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dir = Path(args.data) / "test"
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Test macro F1:  {macro_f1:.4f}")
    print(f"Arch: {arch} | img_size: {img_size}")

    # Report por clase (muy útil para README)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=False)
    print("\nClassification report:\n")
    print(report)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Guardar métricas
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "test_acc": acc,
                "test_macro_f1": macro_f1,
                "arch": arch,
                "img_size": img_size,
            },
            indent=2
        ),
        encoding="utf-8"
    )

    # Guardar report como txt
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(reports_dir / "confusion_matrix.png", bbox_inches="tight", dpi=200)

    print("\n✅ Saved:")
    print(" - reports/test_metrics.json")
    print(" - reports/classification_report.txt")
    print(" - reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
