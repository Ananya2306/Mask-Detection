<h1 align="center">😷 Face Mask Detection using Deep Learning</h1>

<p align="center">
  <em>AI that keeps you safe — one pixel at a time.</em><br>
  Built with ❤️ using <strong>PyTorch</strong> + <strong>Streamlit</strong> + <strong>Computer Vision</strong>
</p>

---

## 🌍 Overview

This project is a real-time **Face Mask Detection System** — a blend of AI, vision, and a dash of magic ✨.  
It uses a fine-tuned **MobileNetV2** model trained on the **Kaggle Face Mask Dataset** to classify faces as:

- 😷 **Mask**
- 😐 **No Mask**

You can upload an image and instantly see whether the model detects a mask — all within a slick **Streamlit UI**.

---

## 🧠 Tech Stack

| Tool | Role |
|------|------|
| **Python 3.12** | Core language |
| **PyTorch** | Model training |
| **Torchvision** | Pretrained MobileNetV2 backbone |
| **Streamlit** | Interactive web interface |
| **OpenCV** | Image processing |
| **Pillow** | Image handling |
| **NumPy** | Math wizardry 🧮 |

---

## 🗂️ Folder Architecture
```
Mask-Detection/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── train.py
│   ├── eval.py
│   └── utils.py
│
├── app.py
├── mask_cls_best.pt
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```


---

## 📦 Dataset

The dataset was sourced from [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset).

It contains:
```
data/
├── train/
│ ├── WithMask/
│ └── WithoutMask/
├── val/
│ ├── WithMask/
│ └── WithoutMask/
└── test/
├── WithMask/
└── WithoutMask/
```
Data was split into **train**, **validation**, and **test** sets for robust evaluation.

---

## ⚙️ Setup & Run Locally

### 🧩 1. Clone the repo
```bash
git clone https://github.com/Ananya2306/Mask-Detection.git
cd Mask-Detection
```

### 🧪 2. Create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 🚀 4. Run the Streamlit app
```
streamlit run app.py
```
App will open at 👉 http://localhost:8501

---

## 🧩 Model Summary

| Property| Value |
|---------|-------|
|Architecture	| MobileNetV2 |
|Framework |	PyTorch |
|Loss Function |	CrossEntropy |
|Optimizer |	Adam |
|Epochs |	8 |
|Validation Accuracy |	99.9% 🎯|

---
## 🌐 Deployment
```
🌩️ Streamlit Cloud (Image Upload Version)

🤖 Coming Soon: Hugging Face Space (Webcam + Realtime Mode)
```

---

## ✨ Results
| Metric |	Value |
|--------|--------|
|Train Accuracy |	99.8% |
|Validation Accuracy |	99.9% |
|Test Accuracy |	99.9% |

📊 The model basically never misses a masked face.

---

## 💬 Author

👩‍💻 Ananya

B.Tech CSE (AI & ML) | IILM University, Greater Noida

📍 India

🔗 [LinkedIn](https://www.linkedin.com/in/ananya-61314128b/)

 • [GitHub](https://github.com/Ananya2306)

---

## 🧡 Credits

Dataset: Kaggle

Frameworks: PyTorch, Streamlit

UI Inspiration: Modern ML Demos

Mentor: Google & IBM AI Ecosystem (self-driven journey 🚀)

--- 

<h3 align="center">"When code meets compassion, AI becomes care." 💫</h3> ```
