import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_resnet18(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # reemplazar capa final
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def set_trainable_layer4_and_fc(model: torch.nn.Module):
    # congelar todo
    for p in model.parameters():
        p.requires_grad = False
    # descongelar layer4 + fc
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/split")
    parser.add_argument("--ckpt", type=str, default="models/model.pt")
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--out", type=str, default="models/model_ft.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("Device:", device, "| AMP:", use_amp)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    num_classes = len(labels)

    # Datasets
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

    data_dir = Path(args.data)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tfms)

    # Modelo
    model = build_resnet18(num_classes)

    # Cargar pesos del baseline
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # Fine-tune SOLO layer4 + fc
    set_trainable_layer4_and_fc(model)

    # channels_last opcional
    if device == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # Optimizador SOLO params entrenables
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train-ft]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val-ft]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if device == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"\n[FT] Epoch {epoch}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "arch": "resnet18",
                "img_size": args.img_size,
                "state_dict": model.state_dict(),
                "finetuned": True,
                "trainable": "layer4+fc",
                "lr": args.lr
            }, out_path)
            print(f"✅ Saved best fine-tuned model to {out_path}")

    print("Done. Best fine-tuned val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
