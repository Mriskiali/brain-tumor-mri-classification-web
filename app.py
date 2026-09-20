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

INFO = {
    "glioma": {
        "label": "Glioma",
        "ringkas": "Tumor yang tumbuh dari sel glia, sel penyokong jaringan saraf di otak.",
        "sections": [
            ("Tentang tumor ini",
             "Glioma tumbuh dari sel glia, yaitu sel penyokong jaringan saraf di otak dan sumsum tulang belakang. "
             "Tingkat keganasannya bervariasi, dari derajat rendah yang tumbuh lambat hingga derajat tinggi yang "
             "tumbuh agresif, seperti glioblastoma."),
            ("Yang biasa tampak pada MRI",
             "Umumnya berupa massa di dalam jaringan otak (intra-aksial) dengan batas yang sering tidak tegas, dan "
             "bisa disertai pembengkakan (edema) di sekitarnya. Pada derajat tinggi, penyerapan kontras sering tidak merata."),
            ("Gejala yang umum",
             "Sakit kepala yang menetap atau makin berat, kejang, kelemahan pada satu sisi tubuh, gangguan bicara atau "
             "penglihatan, serta perubahan perilaku atau daya ingat. Gejalanya bergantung pada lokasi tumor."),
            ("Penanganan umum",
             "Ditentukan oleh derajat dan lokasi tumor, dan biasanya melibatkan dokter bedah saraf dan onkologi. "
             "Pilihannya dapat berupa operasi, radioterapi, dan kemoterapi."),
        ],
    },
    "meningioma": {
        "label": "Meningioma",
        "ringkas": "Tumor yang tumbuh dari selaput pelindung otak (meningen), umumnya jinak.",
        "sections": [
            ("Tentang tumor ini",
             "Meningioma tumbuh dari selaput pelindung otak dan sumsum tulang belakang (meningen). Sebagian besar "
             "bersifat jinak dan tumbuh lambat, tetapi tetap bisa menimbulkan gejala karena menekan jaringan otak di "
             "dekatnya. Lebih sering ditemukan pada perempuan dan pada usia dewasa lanjut."),
            ("Yang biasa tampak pada MRI",
             "Umumnya tampak sebagai massa di luar jaringan otak (ekstra-aksial) yang menempel pada selaput otak, "
             "dengan batas tegas dan penyerapan kontras yang merata dan kuat."),
            ("Gejala yang umum",
             "Sering tanpa gejala dan ditemukan secara tidak sengaja. Bila bergejala: sakit kepala, kejang, gangguan "
             "penglihatan, atau kelemahan yang berkembang perlahan."),
            ("Penanganan umum",
             "Meningioma kecil tanpa gejala sering hanya dipantau dengan MRI berkala. Bila membesar atau bergejala, "
             "pilihannya operasi atau radioterapi, dengan keputusan dari dokter bedah saraf."),
        ],
    },
    "pituitary": {
        "label": "Tumor pituitari",
        "ringkas": "Tumor pada kelenjar pituitari di dasar otak yang mengatur banyak hormon.",
        "sections": [
            ("Tentang tumor ini",
             "Tumor pituitari (umumnya adenoma hipofisis) tumbuh di kelenjar pituitari, kelenjar kecil di dasar otak "
             "yang mengatur banyak hormon tubuh. Sebagian besar bersifat jinak. Ada yang menghasilkan hormon berlebih "
             "dan ada yang tidak."),
            ("Yang biasa tampak pada MRI",
             "Tampak sebagai massa di area sella turcica, yaitu rongga tulang di dasar tengkorak tempat kelenjar "
             "pituitari berada, dan dapat meluas ke atas mendekati saraf penglihatan."),
            ("Gejala yang umum",
             "Gangguan penglihatan (terutama lapang pandang bagian samping), sakit kepala, serta gejala hormonal seperti "
             "haid tidak teratur, keluarnya ASI di luar masa menyusui, atau perubahan fisik akibat kelebihan hormon."),
            ("Penanganan umum",
             "Bergantung pada jenis dan ukuran tumor. Sebagian dapat diobati dengan obat, sebagian memerlukan operasi "
             "lewat hidung (transsfenoidal) atau radioterapi. Biasanya ditangani bersama dokter endokrin dan bedah saraf."),
        ],
    },
    "notumor": {
        "label": "Tidak terdeteksi tumor",
        "ringkas": "Model tidak menemukan pola glioma, meningioma, atau tumor pituitari.",
        "sections": [
            ("Hasil ini berarti",
             "Model tidak menemukan pola yang cocok dengan glioma, meningioma, maupun tumor pituitari pada citra ini."),
            ("Yang perlu diingat",
             "Hasil ini tidak menyingkirkan kelainan lain, dan model bisa saja keliru. Penilaian citra MRI tetap harus "
             "dilakukan oleh dokter spesialis radiologi."),
            ("Bila ada keluhan",
             "Sakit kepala yang menetap, kejang, gangguan penglihatan, atau kelemahan anggota gerak tetap perlu "
             "diperiksakan ke dokter meskipun hasil model tidak menunjukkan tumor."),
        ],
    },
}
DISPLAY_ORDER = ["glioma", "meningioma", "pituitary", "notumor"]

st.set_page_config(
    page_title="Klasifikasi Tumor Otak MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Spectral:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Public Sans', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

.title { font-family: 'Spectral', Georgia, serif; font-size: 2.3rem; font-weight: 700;
         line-height: 1.15; color: #16212B; margin: 0 0 .35rem 0; }
.subtitle { color: #4B5A68; font-size: 1.02rem; margin-bottom: 1.4rem; line-height: 1.55; }

.result { background: #16212B; color: #F3F5F7; border-radius: 6px; padding: 1.4rem 1.6rem; margin: .6rem 0 1.4rem 0; }
.result .kicker { color: #9FB0BF; font-size: .85rem; margin-bottom: .25rem; }
.result .name { font-family: 'Spectral', Georgia, serif; font-size: 2.1rem; font-weight: 700; line-height: 1.15; }
.result .conf { color: #7FD1D8; font-size: 1.05rem; margin-top: .3rem; }
.result .desc { color: #C7D2DB; font-size: .95rem; margin-top: .6rem; line-height: 1.5; }

.warn { background: #FFF4DB; border-left: 4px solid #C98A00; padding: .65rem 1rem; border-radius: 3px;
        color: #5B4000; font-size: .92rem; margin: -.6rem 0 1.2rem 0; }

.bar-row { margin-bottom: .85rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .93rem; margin-bottom: .25rem; }
.bar-head .n { color: #16212B; font-weight: 500; }
.bar-head .v { color: #4B5A68; font-variant-numeric: tabular-nums; }
.bar-track { background: #DCE3E9; height: 10px; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; background: #A9B6C2; }
.bar-fill.top { background: #0E7C86; }

.section-h { font-family: 'Spectral', Georgia, serif; font-size: 1.25rem; font-weight: 600; margin: 0 0 .7rem 0; color: #16212B; }
.caption { color: #4B5A68; font-size: .85rem; text-align: center; margin-top: .3rem; }

.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; margin-top: .2rem; }
.info-card { background: #FFFFFF; border: 1px solid #D5DCE2; border-radius: 4px; padding: 1rem 1.15rem; }
.info-card h4 { font-family: 'Public Sans', sans-serif; font-size: .98rem; font-weight: 600; color: #0E7C86; margin: 0 0 .4rem 0; }
.info-card p { color: #2B3946; font-size: .93rem; line-height: 1.6; margin: 0; }

.class-card { background: #FFFFFF; border: 1px solid #D5DCE2; border-radius: 4px; padding: 1rem 1.15rem; height: 100%; }
.class-card .cn { font-family: 'Spectral', Georgia, serif; font-size: 1.15rem; font-weight: 600; color: #16212B; margin-bottom: .3rem; }
.class-card .cd { color: #4B5A68; font-size: .9rem; line-height: 1.5; }

.disclaimer { color: #4B5A68; font-size: .83rem; border-top: 1px solid #D5DCE2; padding-top: .9rem; margin-top: 2rem; }
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


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Input: RGB uint8. Output: float32 [0,1] berukuran 128x128x3."""
    img = cv2.GaussianBlur(rgb, (3, 3), 0)
    img = apply_clahe(img)
    img = crop_brain_roi(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32") / 255.0


# ----------------------------------------------------------------------------
# Model & Grad-CAM
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def find_last_conv_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def gradcam_heatmap(model, x, layer_name, class_idx):
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


def overlay_heatmap(img_float, heat, size=384, alpha=0.45):
    base = cv2.resize((img_float * 255).astype("uint8"), (size, size), interpolation=cv2.INTER_CUBIC)
    heat_r = cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heat_r, 0, 1)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(base * (1 - alpha) + colored * alpha)


def pct(p: float) -> str:
    return f"{p * 100:.1f}".replace(".", ",") + "%"


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
st.markdown('<div class="title">Klasifikasi Tumor Otak dari Citra MRI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Unggah citra MRI otak untuk mendapatkan prediksi jenis tumornya.</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error("File `classifier_final.keras` tidak ditemukan di folder yang sama dengan `app.py`.")
    st.stop()

uploaded = st.file_uploader("Pilih citra MRI (JPG, JPEG, atau PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.markdown('<div class="section-h" style="margin-top:1.4rem">Jenis yang dikenali</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, key in zip(cols, DISPLAY_ORDER):
        with col:
            st.markdown(
                f'<div class="class-card"><div class="cn">{INFO[key]["label"]}</div>'
                f'<div class="cd">{INFO[key]["ringkas"]}</div></div>',
                unsafe_allow_html=True,
            )
else:
    try:
        rgb = np.array(Image.open(uploaded).convert("RGB"), dtype=np.uint8)
    except Exception:
        st.error("Gambar tidak bisa dibaca. Coba file lain.")
        st.stop()

    model = load_model()
    x = preprocess(rgb)

    with st.spinner("Menghitung prediksi..."):
        probs = model.predict(x[np.newaxis], verbose=0)[0]

    top = int(np.argmax(probs))
    top_cls = CLASSES[top]
    info = INFO[top_cls]
    conf = float(probs[top])

    st.markdown('<div class="section-h" style="margin-top:1.2rem">Laporan hasil klasifikasi</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result"><div class="kicker">Prediksi model</div>'
        f'<div class="name">{info["label"]}</div>'
        f'<div class="conf">Keyakinan {pct(conf)}</div>'
        f'<div class="desc">{info["ringkas"]}</div></div>',
        unsafe_allow_html=True,
    )

    if conf < LOW_CONFIDENCE:
        st.markdown(
            f'<div class="warn">Keyakinan model di bawah {int(LOW_CONFIDENCE * 100)}%, jadi hasil ini kurang dapat diandalkan.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-h">Citra</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(rgb, width="stretch")
            st.markdown('<div class="caption">Citra asli</div>', unsafe_allow_html=True)
        with c2:
            st.image(cv2.resize((x * 255).astype("uint8"), (384, 384), interpolation=cv2.INTER_CUBIC), width="stretch")
            st.markdown('<div class="caption">Setelah diproses</div>', unsafe_allow_html=True)
        with c3:
            try:
                heat = gradcam_heatmap(model, x, find_last_conv_name(model), top)
                st.image(overlay_heatmap(x, heat), width="stretch")
                st.markdown('<div class="caption">Area perhatian model</div>', unsafe_allow_html=True)
            except Exception:
                st.caption("Area perhatian tidak tersedia.")

    with right:
        st.markdown('<div class="section-h">Probabilitas per kelas</div>', unsafe_allow_html=True)
        rows = ""
        for i in np.argsort(-probs):
            p = float(probs[i]) * 100
            cls = "bar-fill top" if i == top else "bar-fill"
            rows += (
                f'<div class="bar-row"><div class="bar-head"><span class="n">{INFO[CLASSES[i]]["label"]}</span>'
                f'<span class="v">{pct(float(probs[i]))}</span></div>'
                f'<div class="bar-track"><div class="{cls}" style="width:{p:.1f}%"></div></div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    st.markdown(f'<div class="section-h" style="margin-top:1.6rem">Informasi: {info["label"]}</div>', unsafe_allow_html=True)
    cards = "".join(
        f'<div class="info-card"><h4>{h}</h4><p>{t}</p></div>' for h, t in info["sections"]
    )
    st.markdown(f'<div class="info-grid">{cards}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Hasil ini adalah prediksi model untuk keperluan pembelajaran, bukan diagnosis medis. '
    "Konsultasikan citra MRI dengan dokter atau radiolog.</div>",
    unsafe_allow_html=True,
)
