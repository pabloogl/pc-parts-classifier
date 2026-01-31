import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # freeze backbone
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        trainable_params = model.fc.parameters()

    elif name in ["efficientnet_b0", "effnet_b0"]:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        trainable_params = model.classifier.parameters()

    else:
        raise ValueError("Modelo no soportado. Usa resnet18 o efficientnet_b0.")

    return model, trainable_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/split")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Num classes: {num_classes}")
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model, trainable_params = get_model(args.model, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # VAL
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"\nEpoch {epoch}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}\n")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "arch": args.model,
                "img_size": args.img_size,
                "state_dict": model.state_dict(),
            }, models_dir / "model.pt")
            (models_dir / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
            print("✅ Saved best model to models/model.pt")

    metrics = {
        "best_val_acc": best_val_acc,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "img_size": args.img_size,
        "history": history
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Done. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
