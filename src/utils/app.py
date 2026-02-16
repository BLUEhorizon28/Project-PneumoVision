import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #080c10;
    --surface:   #0d1117;
    --panel:     #111820;
    --border:    #1e2d3d;
    --accent:    #00d9ff;
    --accent2:   #0066ff;
    --danger:    #ff4d6d;
    --safe:      #00e5a0;
    --warn:      #ffb547;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
    --body:      'Inter', sans-serif;
}

/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1400px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Header Banner ── */
.header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 2rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #0d1117 0%, #111820 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}
.header-wrap::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,217,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,102,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.logo-area { display: flex; align-items: center; gap: 1rem; }
.logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 24px rgba(0,217,255,0.25);
}
.logo-text h1 {
    font-family: var(--display) !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    background: linear-gradient(90deg, #ffffff 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important; padding: 0 !important;
    line-height: 1 !important;
}
.logo-text p {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.15em !important;
    margin: 4px 0 0 0 !important;
    text-transform: uppercase;
}
.status-pill {
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(0, 229, 160, 0.1);
    border: 1px solid rgba(0, 229, 160, 0.3);
    color: var(--safe);
    letter-spacing: 0.1em;
}

/* ── Upload Zone ── */
.upload-section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
}
.section-label {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    margin-bottom: 1rem !important;
    display: block;
}

[data-testid="stFileUploader"] {
    background: rgba(0,217,255,0.03) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: var(--body) !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--body) !important;
}

/* ── Result Card ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card.pneumonia {
    background: linear-gradient(135deg, rgba(255,77,109,0.08) 0%, rgba(255,77,109,0.03) 100%);
    border-color: rgba(255,77,109,0.35);
}
.result-card.normal {
    background: linear-gradient(135deg, rgba(0,229,160,0.08) 0%, rgba(0,229,160,0.03) 100%);
    border-color: rgba(0,229,160,0.35);
}
.result-label {
    font-family: var(--display) !important;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.7;
}
.result-verdict {
    font-family: var(--display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    margin: 0.4rem 0 !important;
}
.result-card.pneumonia .result-verdict { color: var(--danger) !important; }
.result-card.normal .result-verdict { color: var(--safe) !important; }

/* ── Confidence Bar ── */
.conf-bar-wrap {
    margin-top: 1.2rem;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
}
.conf-bar-fill.pneumonia { background: linear-gradient(90deg, #ff4d6d, #ff8fa3); }
.conf-bar-fill.normal    { background: linear-gradient(90deg, #00e5a0, #7fffcf); }

/* ── Metric Boxes ── */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.2rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
}
.metric-box .val {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--accent);
    display: block;
}
.metric-box .lbl {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 2px;
    display: block;
}

/* ── Image Panel ── */
.img-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
}
.img-panel img {
    border-radius: 10px;
    width: 100%;
}

/* ── Heatmap Panel ── */
.heatmap-note {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.75rem;
    line-height: 1.6;
    letter-spacing: 0.02em;
}

/* ── Clinical Warning ── */
.clinical-warn {
    background: rgba(255,181,71,0.07);
    border: 1px solid rgba(255,181,71,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-top: 1.5rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--warn);
    line-height: 1.7;
    letter-spacing: 0.02em;
}

/* ── Steps Info ── */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}
.step-item:last-child { border-bottom: none; }
.step-num {
    width: 26px; height: 26px; min-width: 26px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 500;
    color: #000;
}
.step-text {
    font-family: var(--body);
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.5;
    padding-top: 2px;
}
.step-text strong { color: var(--text); }

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent2) 0%, var(--accent) 100%) !important;
    color: #000 !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,217,255,0.2) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Streamlit image ── */
[data-testid="stImage"] img { border-radius: 10px; }

/* ── Footer ── */
.footer {
    margin-top: 3rem;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DUMMY MODEL (replace with real trained model)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Replace this with your actual hybrid CNN-ViT model.
    For now, we load a ResNet-50 as a placeholder.
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    model.eval()
    return model


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Apply CLAHE + ImageNet normalization as specified in the paper."""
    img_np = np.array(pil_image.convert("RGB"))

    # CLAHE on L channel
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img_clahe = cv2.merge([l_clahe, a, b])
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
    enhanced = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(enhanced).unsqueeze(0)


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=0):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0][class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_pil: Image.Image, cam: np.ndarray) -> Image.Image:
    """Overlay the Grad-CAM heatmap on the original image."""
    original_resized = original_pil.convert("RGB").resize((224, 224))
    original_np = np.array(original_resized)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 0.55, heatmap_rgb, 0.45, 0)
    return Image.fromarray(overlay)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, tensor: torch.Tensor):
    with torch.no_grad():
        prob = model(tensor).item()
    return prob


# ─────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="header-wrap">
        <div class="logo-area">
            <div class="logo-icon">🫁</div>
            <div class="logo-text">
                <h1>PneumoScan</h1>
                <p>Hybrid CNN-ViT · Grad-CAM XAI · VIT Bhopal University</p>
            </div>
        </div>
        <div class="status-pill">● SYSTEM READY</div>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(prob: float):
    is_pneumonia = prob >= 0.5
    label = "PNEUMONIA DETECTED" if is_pneumonia else "NO PNEUMONIA"
    css_class = "pneumonia" if is_pneumonia else "normal"
    conf = prob if is_pneumonia else (1 - prob)
    conf_pct = f"{conf * 100:.1f}%"
    bar_width = int(conf * 100)

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">Diagnosis Output</div>
        <div class="result-verdict">{label}</div>
        <div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);margin-top:0.3rem;">
            Confidence: {conf_pct}
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill {css_class}" style="width:{bar_width}%"></div>
        </div>
        <div class="metric-row">
            <div class="metric-box">
                <span class="val">{prob*100:.1f}%</span>
                <span class="lbl">P(Pneumonia)</span>
            </div>
            <div class="metric-box">
                <span class="val">{(1-prob)*100:.1f}%</span>
                <span class="lbl">P(Normal)</span>
            </div>
            <div class="metric-box">
                <span class="val">{'HIGH' if conf > 0.85 else 'MED'}</span>
                <span class="lbl">Confidence</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_how_it_works():
    st.markdown('<span class="section-label">How It Works</span>', unsafe_allow_html=True)
    steps = [
        ("Upload", "Upload an AP Chest X-Ray in JPG/PNG format."),
        ("Preprocess", "<strong>CLAHE</strong> enhancement + ImageNet normalization applied."),
        ("Dual Inference", "<strong>ResNet-50</strong> extracts local features; <strong>ViT-B/16</strong> captures global attention."),
        ("Fusion", "Feature vectors are concatenated and passed to the classification head."),
        ("Grad-CAM", "Gradient maps highlight the <strong>regions driving the prediction</strong>."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-item">
            <div class="step-num">{i:02d}</div>
            <div class="step-text"><strong>{title}</strong> — {desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
render_header()

model = load_model()

col_left, col_right = st.columns([1, 1.6], gap="large")

# ── LEFT COLUMN ──────────────────────────────
with col_left:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Input · Chest X-Ray</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop your CXR image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.markdown('<span class="section-label" style="margin-top:1rem;display:block;">Preview</span>',
                    unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded X-Ray")
        st.markdown(f"""
        <div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);margin-top:0.5rem;">
            Resolution: {image.size[0]} × {image.size[1]} px &nbsp;|&nbsp; Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("🔬  Run PneumoScan Analysis")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0;font-family:var(--mono);
                    font-size:0.75rem;color:var(--muted);letter-spacing:0.05em;">
            Upload a chest X-ray to begin analysis
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    render_how_it_works()
    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT COLUMN ─────────────────────────────
with col_right:
    if uploaded and run_btn:
        with st.spinner("Running inference pipeline..."):
            time.sleep(0.6)  # small UX delay for feedback
            tensor = preprocess_image(image)

            # Inference
            prob = run_inference(model, tensor)

            # Grad-CAM (requires grad)
            tensor_grad = tensor.clone().requires_grad_(True)
            target_layer = model.layer4[-1]  # last ResNet block
            gradcam = GradCAM(model, target_layer)

            # Need grad for cam generation
            model_copy = load_model()
            tensor_grad2 = preprocess_image(image).requires_grad_(True)
            gradcam2 = GradCAM(model_copy, model_copy.layer4[-1])
            cam = gradcam2.generate(tensor_grad2)
            heatmap_img = overlay_heatmap(image, cam)

        # Result Card
        render_result_card(prob)

        st.markdown("<br>", unsafe_allow_html=True)

        # Image columns
        img_col1, img_col2 = st.columns(2, gap="medium")
        with img_col1:
            st.markdown('<span class="section-label">Original X-Ray</span>',
                        unsafe_allow_html=True)
            st.image(image.resize((224, 224)), use_container_width=True)

        with img_col2:
            st.markdown('<span class="section-label">Grad-CAM Heatmap</span>',
                        unsafe_allow_html=True)
            st.image(heatmap_img, use_container_width=True)
            st.markdown("""
            <div class="heatmap-note">
                🔴 Red regions = highest activation<br>
                🔵 Blue regions = low activation<br>
                Overlay highlights areas most influential to the model's decision.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clinical-warn">
            ⚠ CLINICAL DISCLAIMER — This tool is a research prototype developed at VIT Bhopal
            University and is NOT approved for clinical diagnosis. Results must be reviewed and
            verified by a licensed radiologist. Do not make medical decisions solely based on
            this output.
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="
            height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: var(--panel);
            border: 1px dashed var(--border);
            border-radius: 16px;
            font-family: var(--mono);
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-align: center;
            gap: 1rem;
        ">
            <div style="font-size:3rem;opacity:0.2;">🫁</div>
            <div>AWAITING INPUT</div>
            <div style="font-size:0.65rem;max-width:260px;line-height:1.8;opacity:0.7;">
                Upload a chest X-ray on the left panel to begin the diagnostic pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            height: 500px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            font-family: var(--mono);
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
        ">
            ← Click "Run PneumoScan Analysis" to proceed
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PNEUMOSCAN &nbsp;·&nbsp; VIT BHOPAL UNIVERSITY &nbsp;·&nbsp; DEPT. OF CS&E &nbsp;·&nbsp; 2026<br>
    Hybrid ResNet-50 + ViT-B/16 &nbsp;·&nbsp; Grad-CAM XAI &nbsp;·&nbsp; Research Prototype
</div>
""", unsafe_allow_html=True)
