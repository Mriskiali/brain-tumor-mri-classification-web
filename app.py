import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------------
# Konfigurasi (harus sama dengan saat training di notebook)
# ----------------------------------------------------------------------------
IMG_SIZE = 128
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # urutan WAJIB sama
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_final.keras")
LOW_CONFIDENCE = 0.60

# Ubah angka ini sesuai hasil di skripsi lo
ACC_INTERNAL = "82,14%"
ACC_EXTERNAL = "82,92%"

CLASS_INFO = {
    "glioma": {
        "label": "Glioma",
        "desc": "Tumor yang berasal dari sel glia, yaitu sel penyokong jaringan saraf di otak dan sumsum tulang belakang.",
    },
    "meningioma": {
        "label": "Meningioma",
        "desc": "Tumor yang tumbuh dari meninges, lapisan selaput pelindung otak dan sumsum tulang belakang.",
    },
    "notumor": {
        "label": "Tidak terdeteksi tumor",
        "desc": "Model tidak menemukan pola yang sesuai dengan tiga jenis tumor yang dikenalnya.",
    },
    "pituitary": {
        "label": "Tumor pituitari",
        "desc": "Tumor yang tumbuh di kelenjar pituitari (hipofisis) di dasar otak.",
    },
}

st.set_page_config(
    page_title="Klasifikasi Tumor Otak MRI",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Spectral:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Public Sans', system-ui, sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.title { font-family: 'Spectral', Georgia, serif; font-size: 2.3rem; font-weight: 700;
         line-height: 1.15; color: #16212B; margin: 0 0 .35rem 0; }
.subtitle { color: #4B5A68; font-size: 1.02rem; max-width: 62ch; margin-bottom: 1.6rem; line-height: 1.55; }

.result { background: #16212B; color: #F3F5F7; border-radius: 6px; padding: 1.4rem 1.6rem; margin: .4rem 0 1.4rem 0; }
.result .kicker { color: #9FB0BF; font-size: .85rem; margin-bottom: .25rem; }
.result .name { font-family: 'Spectral', Georgia, serif; font-size: 2.1rem; font-weight: 700; line-height: 1.15; }
.result .conf { color: #7FD1D8; font-size: 1.05rem; margin-top: .3rem; }
.result .desc { color: #C7D2DB; font-size: .95rem; margin-top: .7rem; max-width: 70ch; line-height: 1.5; }

.warn { background: #FFF4DB; border-left: 4px solid #C98A00; padding: .75rem 1rem; border-radius: 3px;
        color: #5B4000; font-size: .93rem; margin-bottom: 1.2rem; }

.bars { margin-top: .2rem; }
.bar-row { margin-bottom: .85rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .93rem; margin-bottom: .25rem; }
.bar-head .n { color: #16212B; font-weight: 500; }
.bar-head .v { color: #4B5A68; font-variant-numeric: tabular-nums; }
.bar-track { background: #DCE3E9; height: 10px; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; background: #A9B6C2; }
.bar-fill.top { background: #0E7C86; }

.section-h { font-family: 'Spectral', Georgia, serif; font-size: 1.25rem; font-weight: 600; margin: 0 0 .6rem 0; color: #16212B; }
.caption { color: #4B5A68; font-size: .85rem; text-align: center; margin-top: .3rem; }
.disclaimer { color: #4B5A68; font-size: .85rem; border-top: 1px solid #D5DCE2; padding-top: 1rem; margin-top: 2rem; line-height: 1.55; }
[data-testid="stImage"] img { border-radius: 4px; background: #0B1218; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Preprocessing (identik dengan preprocess_image() di notebook)
# ----------------------------------------------------------------------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def crop_brain_roi(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    thr = cv2.erode(thr, None, iterations=2)
    thr = cv2.dilate(thr, None, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 10 or h < 10:
        return image
    return image[y : y + h, x : x + w]


def preprocess_with_stages(rgb: np.ndarray):
    """Input: RGB uint8. Output: (tensor float32 [0,1] 128x128x3, dict tahapan)."""
    blur = cv2.GaussianBlur(rgb, (3, 3), 0)
    clahe = apply_clahe(blur)
    crop = crop_brain_roi(clahe)
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    x = resized.astype("float32") / 255.0
    stages = {
        "Citra asli": rgb,
        "Gaussian blur": blur,
        "CLAHE": clahe,
        "Cropping otak": crop,
        f"Resize {IMG_SIZE}×{IMG_SIZE}": resized,
    }
    return x, stages


# ----------------------------------------------------------------------------
# Model & Grad-CAM
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def find_last_conv_name(model) -> str | None:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def gradcam_heatmap(model, x: np.ndarray, layer_name: str, class_idx: int) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        inp = tf.convert_to_tensor(x[np.newaxis], dtype=tf.float32)
        conv_out, preds = grad_model(inp, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def overlay_heatmap(img_float: np.ndarray, heat: np.ndarray, size: int = 384, alpha: float = 0.45):
    base = cv2.resize((img_float * 255).astype("uint8"), (size, size), interpolation=cv2.INTER_CUBIC)
    heat_r = cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heat_r, 0, 1)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(base * (1 - alpha) + colored * alpha)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Tentang model")
    st.markdown(
        "Arsitektur hybrid: encoder dari convolutional autoencoder digabung dengan "
        "classifier CNN, lalu di-fine-tune pada blok konvolusi terdalam."
    )
    st.markdown("**Akurasi**")
    st.markdown(f"- Data uji internal: {ACC_INTERNAL}\n- Validasi eksternal (BraTS, glioma): {ACC_EXTERNAL}")
    st.markdown("**Tahapan preprocessing**")
    st.markdown(
        "1. Gaussian blur 3×3\n2. CLAHE pada ruang warna LAB\n3. Contour cropping\n"
        f"4. Resize {IMG_SIZE}×{IMG_SIZE}\n5. Normalisasi ke [0, 1]"
    )
    st.markdown("**Keterbatasan**")
    st.markdown(
        "Model hanya menerima citra MRI otak 2D. Gambar di luar domain tersebut "
        "akan tetap diberi prediksi, tetapi hasilnya tidak bermakna."
    )

# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.markdown('<div class="title">Klasifikasi Tumor Otak dari Citra MRI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Unggah satu citra MRI otak. Model akan memprediksi apakah citra menunjukkan '
    "glioma, meningioma, tumor pituitari, atau tidak ada tumor, lengkap dengan area yang paling "
    "memengaruhi keputusan model.</div>",
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("File `classifier_final.keras` tidak ditemukan. Upload file model ke folder yang sama dengan `app.py`.")
    st.stop()

uploaded = st.file_uploader("Pilih citra MRI (JPG, JPEG, atau PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Belum ada gambar. Unggah citra MRI untuk memulai.")
else:
    try:
        rgb = np.array(Image.open(uploaded).convert("RGB"), dtype=np.uint8)
    except Exception:
        st.error("Gambar tidak bisa dibaca. Coba file lain.")
        st.stop()

    model = load_model()
    x, stages = preprocess_with_stages(rgb)

    with st.spinner("Menghitung prediksi..."):
        probs = model.predict(x[np.newaxis], verbose=0)[0]

    top = int(np.argmax(probs))
    top_cls = CLASSES[top]
    info = CLASS_INFO[top_cls]
    conf = float(probs[top])

    st.markdown(
        f'<div class="result"><div class="kicker">Hasil prediksi</div>'
        f'<div class="name">{info["label"]}</div>'
        f'<div class="conf">Keyakinan model {conf * 100:.1f}%</div>'
        f'<div class="desc">{info["desc"]}</div></div>',
        unsafe_allow_html=True,
    )

    if conf < LOW_CONFIDENCE:
        st.markdown(
            f'<div class="warn">Keyakinan model di bawah {int(LOW_CONFIDENCE * 100)}%. '
            "Hasil ini kurang dapat diandalkan. Periksa apakah gambar adalah MRI otak dengan kualitas yang baik.</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-h">Citra dan area perhatian model</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(rgb, use_container_width=True)
            st.markdown('<div class="caption">Citra asli</div>', unsafe_allow_html=True)
        with c2:
            st.image(cv2.resize((x * 255).astype("uint8"), (384, 384), interpolation=cv2.INTER_CUBIC),
                     use_container_width=True)
            st.markdown(f'<div class="caption">Setelah preprocessing</div>', unsafe_allow_html=True)
        with c3:
            layer_name = find_last_conv_name(model)
            try:
                heat = gradcam_heatmap(model, x, layer_name, top)
                st.image(overlay_heatmap(x, heat), use_container_width=True)
                st.markdown('<div class="caption">Grad-CAM</div>', unsafe_allow_html=True)
            except Exception:
                st.caption("Grad-CAM tidak tersedia untuk gambar ini.")
        st.caption(
            "Warna merah pada Grad-CAM menandai area yang paling berpengaruh terhadap prediksi. "
            "Ini penjelasan perilaku model, bukan penanda lokasi tumor yang pasti."
        )

    with right:
        st.markdown('<div class="section-h">Probabilitas per kelas</div>', unsafe_allow_html=True)
        order = np.argsort(-probs)
        rows = ""
        for i in order:
            p = float(probs[i]) * 100
            cls = "bar-fill top" if i == top else "bar-fill"
            rows += (
                f'<div class="bar-row"><div class="bar-head"><span class="n">{CLASS_INFO[CLASSES[i]]["label"]}</span>'
                f'<span class="v">{p:.1f}%</span></div>'
                f'<div class="bar-track"><div class="{cls}" style="width:{p:.1f}%"></div></div></div>'
            )
        st.markdown(f'<div class="bars">{rows}</div>', unsafe_allow_html=True)

    with st.expander("Lihat setiap tahap preprocessing"):
        cols = st.columns(len(stages))
        for col, (name, im) in zip(cols, stages.items()):
            with col:
                st.image(im, use_container_width=True)
                st.markdown(f'<div class="caption">{name}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Aplikasi ini dibuat untuk keperluan penelitian dan pembelajaran. '
    "Hasilnya bukan diagnosis medis dan tidak boleh dipakai untuk mengambil keputusan klinis. "
    "Konsultasikan hasil pemeriksaan MRI dengan dokter atau radiolog.</div>",
    unsafe_allow_html=True,
)
