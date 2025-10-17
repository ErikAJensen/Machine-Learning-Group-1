import os, copy, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# === KONFIGURASJON ===
DATA_ROOT = r"C:\Users\erik1\OneDrive\Desktop\archive (3)\resized"
BATCH_SIZE   = 32
NUM_WORKERS  = 2   # Sett til 0 hvis du fortsatt får feilmeldinger
EPOCHS       = 12
LR           = 3e-4
PATIENCE     = 4
IMG_SIZE     = 224
mean, std    = [0.485,0.456,0.406], [0.229,0.224,0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === HOVEDFUNKSJON ===
def main():
    assert Path(DATA_ROOT).exists(), f"Fant ikke mappen: {DATA_ROOT}"
    print("Fant data i:", DATA_ROOT)
    print("Klassemapper:", [p.name for p in Path(DATA_ROOT).iterdir() if p.is_dir()])

    # === TRANSFORMS ===
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    # === DATASET & STRATIFIED SPLIT ===
    full_ds = datasets.ImageFolder(DATA_ROOT)
    targets = full_ds.targets
    indices = np.arange(len(full_ds))

    # 70/15/15 split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(indices, targets))
    y_temp = np.array(targets)[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(sss2.split(temp_idx, y_temp))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    train_ds = Subset(datasets.ImageFolder(DATA_ROOT, transform=train_tfms), train_idx)
    val_ds   = Subset(datasets.ImageFolder(DATA_ROOT, transform=eval_tfms),   val_idx)
    test_ds  = Subset(datasets.ImageFolder(DATA_ROOT, transform=eval_tfms),   test_idx)

    cls_to_idx = full_ds.class_to_idx
    idx_to_cls = {v:k for k,v in cls_to_idx.items()}
    print("Klasser:", [idx_to_cls[i] for i in range(len(idx_to_cls))])

    # === DATALOADERS ===
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # === MODELL (ResNet-18) ===
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, len(cls_to_idx))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === TRENING MED EARLY STOPPING ===
    best_w, best_val, patience = copy.deepcopy(model.state_dict()), float("inf"), PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_dl.dataset)

        # --- validering ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_dl.dataset)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_w = copy.deepcopy(model.state_dict())
            patience = PATIENCE
            torch.save(best_w, "best_model.pth")
            print(" Lagret ny beste modell (best_model.pth)")
        else:
            patience -= 1
            if patience == 0:
                print("  Early stopping!")
                break

    # last beste modell
    model.load_state_dict(best_w)

    # === TEST ===
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(DEVICE))
            preds.append(logits.argmax(1).cpu().numpy())
            gts.append(yb.numpy())
    y_true = np.concatenate(gts)
    y_pred = np.concatenate(preds)

    acc = (y_true == y_pred).mean()
    print(f"\n Test accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=[idx_to_cls[i] for i in range(len(idx_to_cls))]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


# === Windows-safe oppstart ===
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
