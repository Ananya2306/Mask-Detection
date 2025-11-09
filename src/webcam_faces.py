import cv2, numpy as np, torch, torchvision.transforms as T
from torchvision import models
from PIL import Image
import mediapipe as mp

# === labels & transforms (must match training) ===
LABELS = ["mask", "no_mask"]   # ImageFolder: {'mask':0,'no_mask':1}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(MEAN, STD)])

# === load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("mask_cls_best.pt", map_location=device))
model = model.to(device).eval()

# === mediapipe face detector ===
mp_fd = mp.solutions.face_detection
detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def classify_face(crop_bgr):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

def expand_box(x, y, w, h, scale, W, H):
    cx, cy = x + w/2, y + h/2
    nw, nh = w*scale, h*scale
    nx, ny = int(max(0, cx - nw/2)), int(max(0, cy - nh/2))
    nx2, ny2 = int(min(W, cx + nw/2)), int(min(H, cy + nh/2))
    return nx, ny, nx2 - nx, ny2 - ny

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise SystemExit("No webcam found.")

# optional: lower resolution for FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok: break
    H, W = frame.shape[:2]

    # detect faces in RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    if not res.detections:
        cv2.putText(frame, "NO FACE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    else:
        for det in res.detections:
            bb = det.location_data.relative_bounding_box
            x, y, w, h = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
            # expand a bit to include mask area; clamp to image
            x, y, w, h = expand_box(x, y, w, h, 1.3, W, H)
            face = frame[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
            if face.size == 0: 
                continue
            label, conf = classify_face(face)
            color = (0, 200, 0) if label=="mask" else (0, 0, 255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label.upper()} {conf:.2f}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Mask Classification (per-face)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
