import torch, torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.utils import train_tf, val_tf


DATA = Path("data")
EPOCHS, BATCH, LR = 8, 32, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    train_ds = tv.datasets.ImageFolder(DATA/"train", transform=train_tf)
    val_ds   = tv.datasets.ImageFolder(DATA/"val",   transform=val_tf)

    print("class_to_idx:", train_ds.class_to_idx)  # should print {'mask':0,'no_mask':1}

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    model = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        for x,y in tqdm(train_dl, desc=f"epoch {ep}/{EPOCHS}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        acc = correct/total
        print(f"val_acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mask_cls_best.pt")

    print("best_val_acc:", best_acc)

if __name__ == "__main__":
    main()
