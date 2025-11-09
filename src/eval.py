import torch, torchvision as tv
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from src.utils import val_tf, LABELS
from torchvision import models
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ds = tv.datasets.ImageFolder(Path("data")/"test", transform=val_tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("mask_cls_best.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu().tolist()
            y_pred += pred; y_true += y.tolist()

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

if __name__ == "__main__":
    main()
