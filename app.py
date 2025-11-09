# app.py — Face Mask Classification (Streamlit Cloud ready)
import io, time, numpy as np, cv2, torch, torchvision.transforms as T
from torchvision import models
from PIL import Image
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Mask Detection", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:2rem;}
.big-title {font-size: 38px; font-weight: 800; letter-spacing:.5px;}
</style>
""", unsafe_allow_html=True)

TITLE  = "🛡️ Face Mask Classification — MobileNetV2"
LABELS = ["mask", "no_mask"]      # ImageFolder order during training: {'mask':0,'no_mask':1}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------------- HELPERS ----------------
@st.cache_resource
def load_model(weights_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(LABELS))
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, device

@st.cache_resource
def get_tf():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

def predict_pil(pil_img, model, device):
    x = get_tf()(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    i = int(np.argmax(probs))
    return LABELS[i], float(probs[i]), probs

# MediaPipe face detector (works with numpy 1.26.x)
mp_fd = mp.solutions.face_detection
@st.cache_resource
def get_detector():
    return mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def expand_box(x, y, w, h, scale, W, H):
    cx, cy = x + w/2, y + h/2
    nw, nh = w*scale, h*scale
    x1 = int(max(0, cx - nw/2)); y1 = int(max(0, cy - nh/2))
    x2 = int(min(W, cx + nw/2));  y2 = int(min(H, cy + nh/2))
    return x1, y1, x2, y2

def annotate_bgr(img_bgr, model, device, conf_thresh=0.6, per_face=True):
    """Return annotated BGR image and list of detections."""
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    results = []

    if not per_face:
        label, conf, _ = predict_pil(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), model, device)
        color = (0,200,0) if label=="mask" else (0,0,255)
        cv2.putText(out, f"{label.upper()} {conf:.2f}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        results.append({"bbox":[0,0,W,H],"label":label,"conf":conf})
        return out, results

    detector = get_detector()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    det = detector.process(rgb)

    if not det.detections:
        cv2.putText(out, "NO FACE", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3, cv2.LINE_AA)
        return out, results

    for d in det.detections:
        bb = d.location_data.relative_bounding_box
        x, y, w, h = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
        x1, y1, x2, y2 = expand_box(x, y, w, h, 1.25, W, H)
        crop = img_bgr[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
        if crop.size == 0: 
            continue
        label, conf, _ = predict_pil(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), model, device)
        if conf < conf_thresh: 
            continue
        color = (0,200,0) if label=="mask" else (0,0,255)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{label.upper()} {conf:.2f}", (x1, max(20,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        results.append({"bbox":[x1,y1,x2,y2], "label":label, "conf":conf})
    return out, results

def bgr_to_png_bytes(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0); return buf

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")
weights_path = st.sidebar.text_input("Model weights (.pt)", value="mask_cls_best.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.10, 0.99, 0.60, 0.01)
mode = st.sidebar.radio("Mode", ["Image", "Webcam"], horizontal=True)
per_face = st.sidebar.toggle("Per-face boxes (MediaPipe)", value=True, help="Detect faces, crop, then classify. Turn off to classify the full frame.")
if st.sidebar.button("Load / Reload Model"):
    st.session_state.pop("bundle", None)

if "bundle" not in st.session_state:
    try:
        st.session_state["bundle"] = load_model(weights_path)
        st.sidebar.success(f"Loaded on {'GPU' if st.session_state['bundle'][1]=='cuda' else 'CPU'}")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")
        st.stop()

model, device = st.session_state["bundle"]

# ---------------- HEADER ----------------
st.markdown(f"<div class='big-title'>{TITLE}</div>", unsafe_allow_html=True)
st.caption("Per-face boxes • Confidence bars • FPS • Works with browser camera on Streamlit Cloud")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📷 Image", "🎥 Webcam"])

# ==== IMAGE TAB ====
with tab1:
    st.subheader("Image Inference")
    c_img, c_info = st.columns([3, 2])
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        out, dets = annotate_bgr(bgr, model, device, conf_thresh=conf_thresh, per_face=per_face)

        with c_img:
            st.image(out, caption="Detections", use_container_width=True)
            st.download_button("⬇️ Download annotated image", data=bgr_to_png_bytes(out),
                               file_name="mask_detection.png", mime="image/png")
        with c_info:
            st.markdown("**Results**")
            if not dets:
                st.write("No confident detections.")
            else:
                for d in dets:
                    st.write(f"- **{d['label']}** — {d['conf']:.2f}")
                    st.progress(int(d["conf"]*100))

# ==== WEBCAM TAB (browser camera via WebRTC) ====
class FaceMaskTransformer(VideoTransformerBase):
    def __init__(self):
        # reuse already loaded globals
        self.model, self.device = model, device
        # values will be read live each frame from st.session_state for responsiveness
    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        # read latest sidebar controls live
        conf = float(st.session_state.get("conf_thresh_state", conf_thresh))
        pface = bool(st.session_state.get("per_face_state", per_face))
        out, _ = annotate_bgr(img_bgr, self.model, self.device,
                              conf_thresh=conf, per_face=pface)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

with tab2:
    st.subheader("Webcam (browser)")
    st.info("Runs in your browser using WebRTC. Works on Streamlit Community Cloud / Hugging Face Spaces.")
    # sync sidebar values into session_state so the transformer can read them
    st.session_state["conf_thresh_state"] = conf_thresh
    st.session_state["per_face_state"] = per_face

    webrtc_streamer(
        key="mask-webrtc",
        video_transformer_factory=FaceMaskTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
