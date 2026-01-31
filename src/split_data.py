import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # resto test

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def has_images(d: Path) -> bool:
    return any(p.suffix.lower() in IMG_EXTS for p in d.rglob("*") if p.is_file())

def find_class_root(raw_dir: Path) -> Path:
    """
    Encuentra la carpeta que contiene las carpetas de clase.
    Caso típico:
      data/raw/pc_parts/cpu, gpu, ...
    """
    # Si raw_dir ya contiene múltiples carpetas con imágenes, es root de clases
    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    img_subdirs = [d for d in subdirs if has_images(d)]
    if len(img_subdirs) >= 2:
        return raw_dir

    # Si solo hay 1 carpeta intermedia (ej: pc_parts), probar dentro
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_subdirs = [d for d in inner.iterdir() if d.is_dir()]
        inner_img_subdirs = [d for d in inner_subdirs if has_images(d)]
        if len(inner_img_subdirs) >= 2:
            return inner

    # Si no lo encuentra, falla con un mensaje útil
    raise RuntimeError(
        f"No encuentro carpetas de clases dentro de {raw_dir}. "
        f"Comprueba la estructura con: ls {raw_dir}"
    )

def main():
    random.seed(SEED)

    raw_dir = Path("data/raw")
    class_root = find_class_root(raw_dir)
    print("✅ Class root detectado:", class_root)

    out_dir = Path("data/split")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in class_root.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)

    for class_dir in class_dirs:
        images = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_dir = out_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy2(img_path, target_dir / img_path.name)

        print(f"{class_dir.name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    print("\n✅ Split creado en data/split/{train,val,test}")

if __name__ == "__main__":
    main()
