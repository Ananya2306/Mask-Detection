import cv2, torch, torchvision.transforms as T
from torchvision import models
from src.utils import IMG_SIZE, MEAN, STD, LABELS
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("mask_cls_best.pt", map_location=DEVICE))
model = model.to(DEVICE).eval()

tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW avoids some Windows camera quirks
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    label = LABELS[pred]  # 'mask' or 'no_mask'
    color = (0,255,0) if label=="mask" else (0,0,255)
    cv2.putText(frame, label.upper(), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Mask Classification (MobileNetV2)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break  # ESC

cap.release(); cv2.destroyAllWindows()
