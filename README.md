# Klasifikasi Tumor Otak dari Citra MRI

Web demo (Streamlit) untuk model hybrid **Autoencoder + CNN** klasifikasi tumor otak MRI 4 kelas:
glioma, meningioma, pituitary, non-tumor.

**Coba langsung:** [brain-tumor-mri-classification-web.streamlit.app](https://brain-tumor-mri-classification-web.streamlit.app)

Fitur:
- Upload citra MRI -> prediksi kelas + probabilitas per kelas
- Attention heatmap (Grad-CAM style) yang menunjukkan area yang dilihat model
- Penjelasan edukatif per kelas tumor

Riset lengkap (notebook training, metrik, pipeline, external validation BraTS):
[brain-tumor-mri-classification](https://github.com/Mriskiali/brain-tumor-mri-classification)

> Aplikasi ini adalah alat bantu penelitian dan bukan pengganti diagnosis medis.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Mu'afa Riski Ali** — [GitHub](https://github.com/Mriskiali) · [LinkedIn](https://www.linkedin.com/in/muafa-riski-ali-3114b536b/)
