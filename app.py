
"""
Aplikasi Analisis Sentimen Tweet Mitigasi Bencana
Metode: TF-IDF + Neural Network (MLPClassifier)

Dataset default: tweet_data.csv
Kolom yang dibutuhkan: created_at, username, full_text

Catatan:
Jika dataset belum memiliki kolom sentimen, aplikasi akan
membuat label awal menggunakan kamus kata (weak labeling).
Evaluasi model mengukur kesesuaian terhadap label otomatis,
bukan terhadap label anotasi manusia.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen Mitigasi Bencana",
    page_icon="🌧️",
    layout="wide",
)

DATA_FILE = "tweet_data.csv"
LABELS = ["positif", "netral", "negatif"]

EMOJI = {
    "positif": "😊",
    "netral": "😐",
    "negatif": "😠",
}

# ============================================================
# 2. KAMUS NORMALISASI
# ============================================================

NORMALISASI = {
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "dr": "dari",
    "utk": "untuk",
    "untk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "tdk": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "bgt": "banget",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "aja": "saja",
    "jgn": "jangan",
    "sm": "sama",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "dpt": "dapat",
    "dapet": "dapat",
    "jd": "jadi",
    "org": "orang",
    "orng": "orang",
    "bbrp": "beberapa",
    "byk": "banyak",
    "bnyk": "banyak",
    "mksh": "terima kasih",
    "makasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "pls": "tolong",
    "please": "tolong",
    "info": "informasi",
}

# ============================================================
# 3. KAMUS SENTIMEN
# ============================================================

KATA_POSITIF = [
    "berhasil", "efektif", "selamat", "terkendali", "lancar",
    "pulih", "gotong royong", "lega", "tepat waktu", "aman",
    "siap", "sigap", "membantu", "mencegah", "teratasi",
    "membaik", "terima kasih", "bersyukur", "berjalan baik",
    "berjalan lancar", "bekerja efektif", "bekerja dengan baik",
]

KATA_NEGATIF = [
    "gagal", "tidak ada simulasi", "tidak siap", "rusak",
    "tidak dirawat", "lambat", "kecewa", "kesulitan",
    "tertutup", "berbahaya", "buruk", "parah", "lalai",
    "memprihatinkan", "kekurangan", "resah", "rawan",
    "kerusakan", "tidak berfungsi", "belum tersedia",
    "minim", "kurang", "terlambat", "panik",
]

# ============================================================
# 4. PREPROCESSING
# ============================================================

def bersihkan_teks(teks):
    """Membersihkan URL, mention, hashtag, angka, dan karakter khusus."""
    teks = str(teks).lower()
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def normalisasi_teks(teks):
    """Mengubah kata tidak baku menjadi bentuk baku."""
    hasil = []

    for token in str(teks).split():
        token = re.sub(r"(.)\1{2,}", r"\1", token)
        hasil.append(NORMALISASI.get(token, token))

    return " ".join(hasil)


def siapkan_teks(teks):
    return normalisasi_teks(bersihkan_teks(teks))


# ============================================================
# 5. PELABELAN OTOMATIS
# ============================================================

def hitung_kata(teks, daftar_kata):
    teks = str(teks).lower()
    jumlah = 0

    for frasa in daftar_kata:
        pola = r"(?<!\w)" + re.escape(frasa) + r"(?!\w)"
        jumlah += len(re.findall(pola, teks))

    return jumlah


def beri_label(teks):
    """Memberikan label awal berdasarkan kamus sentimen."""

    teks_bersih = siapkan_teks(teks)

    skor_positif = hitung_kata(
        teks_bersih, KATA_POSITIF
    )

    skor_negatif = hitung_kata(
        teks_bersih, KATA_NEGATIF
    )

    if skor_positif > skor_negatif:
        return "positif"

    if skor_negatif > skor_positif:
        return "negatif"

    return "netral"


# ============================================================
# 6. MEMUAT DATASET
# ============================================================

@st.cache_data
def load_data(file_bytes=None):

    if file_bytes is not None:
        from io import BytesIO
        data = pd.read_csv(BytesIO(file_bytes))

    else:
        path = Path(DATA_FILE)

        if not path.exists():
            raise FileNotFoundError(
                f"File {DATA_FILE} tidak ditemukan."
            )

        data = pd.read_csv(path)

    data.columns = data.columns.str.strip().str.lower()

    if "full_text" not in data.columns:
        raise ValueError(
            "Kolom 'full_text' tidak ditemukan."
        )

    data["full_text"] = (
        data["full_text"]
        .fillna("")
        .astype(str)
    )

    data = data[
        data["full_text"].str.strip() != ""
    ].copy()

    data["clean_text"] = (
        data["full_text"].apply(bersihkan_teks)
    )

    data["normalisasi_text"] = (
        data["clean_text"].apply(normalisasi_teks)
    )

    # Menggunakan label dataset jika tersedia.
    # Jika tidak tersedia, gunakan weak labeling.

    if "sentimen" in data.columns:

        data["sentimen"] = (
            data["sentimen"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        data = data[
            data["sentimen"].isin(LABELS)
        ].copy()

        data["sumber_label"] = "Label dari dataset"

    else:

        data["sentimen"] = (
            data["full_text"].apply(beri_label)
        )

        data["sumber_label"] = (
            "Label otomatis berbasis kamus"
        )

    data = data[
        data["normalisasi_text"].str.strip() != ""
    ].copy()

    return data.reset_index(drop=True)


# ============================================================
# 7. TRAINING MODEL NEURAL NETWORK
# ============================================================

@st.cache_resource
def train_model(teks_train, label_train):

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        max_features=5000,
    )

    X_train_tfidf = tfidf.fit_transform(
        list(teks_train)
    )

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        list(label_train)
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    mlp.fit(
        X_train_tfidf,
        y_train_encoded
    )

    return tfidf, mlp, encoder


# ============================================================
# 8. SIDEBAR DAN PEMUATAN DATA
# ============================================================

st.sidebar.title("⚙️ Pengaturan Aplikasi")

st.sidebar.write("**Metode:** Neural Network (MLP)")
st.sidebar.write("**Ekstraksi fitur:** TF-IDF")
st.sidebar.write("**Dataset default:** tweet_data.csv")

file_upload = st.sidebar.file_uploader(
    "Unggah dataset CSV lain (opsional)",
    type=["csv"],
)

try:

    uploaded_bytes = (
        file_upload.getvalue()
        if file_upload is not None
        else None
    )

    df = load_data(uploaded_bytes)

except Exception as error:

    st.error(f"Gagal membaca dataset: {error}")

    st.info(
        "Letakkan tweet_data.csv di folder yang sama "
        "dengan app.py atau unggah melalui sidebar."
    )

    st.stop()


if df["sentimen"].nunique() < 2:

    st.error(
        "Dataset harus memiliki minimal dua kelas sentimen."
    )

    st.stop()


jumlah_kelas = df["sentimen"].value_counts()

if jumlah_kelas.min() < 2:

    st.error(
        "Setiap kelas sentimen harus memiliki minimal "
        "dua data untuk pembagian training dan testing."
    )

    st.stop()


# ============================================================
# 9. TRAIN-TEST SPLIT DAN TRAINING MODEL
# ============================================================

X = df["normalisasi_text"]
y = df["sentimen"]

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    tfidf, mlp, label_encoder = train_model(
        tuple(X_train.astype(str).tolist()),
        tuple(y_train.astype(str).tolist()),
    )

    X_test_tfidf = tfidf.transform(
        X_test.astype(str).tolist()
    )

    y_pred_encoded = mlp.predict(X_test_tfidf)

    y_pred = label_encoder.inverse_transform(
        y_pred_encoded.astype(int)
    )

except Exception as error:

    st.error(
        f"Proses pelatihan model gagal: {error}"
    )

    st.stop()


# ============================================================
# 10. METRIK EVALUASI
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

labels = [
    label for label in LABELS
    if label in y.unique()
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)


# ============================================================
# 11. HEADER DAN INFORMASI
# ============================================================

st.title(
    "🌧️ Analisis Sentimen tentang Mitigasi Bencana"
)

st.write(
    "Aplikasi untuk menganalisis sentimen tweet "
    "mengenai mitigasi bencana menggunakan preprocessing "
    "teks, TF-IDF, dan Neural Network (MLPClassifier)."
)

if (
    "sumber_label" in df.columns
    and df["sumber_label"].iloc[0]
    == "Label otomatis berbasis kamus"
):

    st.warning(
        "Dataset belum memiliki kolom sentimen. "
        "Aplikasi membuat label awal secara otomatis "
        "menggunakan kamus kata positif dan negatif. "
        "Periksa kembali label sebelum menggunakan "
        "hasil evaluasi sebagai hasil penelitian."
    )

else:

    st.info(
        "Aplikasi menggunakan kolom sentimen "
        "yang tersedia pada dataset."
    )


# ============================================================
# 12. RINGKASAN DATA
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Jumlah Data", f"{len(df):,}")
c2.metric("Data Training", f"{len(X_train):,}")
c3.metric("Data Testing", f"{len(X_test):,}")
c4.metric("Akurasi", f"{accuracy:.2%}")

st.divider()


# ============================================================
# 13. TAB APLIKASI
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard & Data",
        "🧹 Cleansing & Normalisasi",
        "🤖 Prediksi",
        "📈 Evaluasi Model",
    ]
)


# ============================================================
# TAB 1 - DASHBOARD & DATA
# ============================================================

with tab1:

    st.subheader("Distribusi Sentimen")

    distribusi = (
        df["sentimen"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        distribusi.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(
            "Jumlah Data Berdasarkan Sentimen"
        )

        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah Data")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        fig, ax = plt.subplots(figsize=(7, 4))

        nilai_pie = distribusi[
            distribusi > 0
        ]

        ax.pie(
            nilai_pie.values,
            labels=nilai_pie.index,
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title("Persentase Sentimen")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Tweet")

    kolom_data = [
        kolom for kolom in
        [
            "created_at",
            "username",
            "full_text",
            "sentimen",
            "sumber_label",
        ]
        if kolom in df.columns
    ]

    st.dataframe(
        df[kolom_data],
        use_container_width=True,
        height=420,
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download Dataset dengan Label Sentimen",
        data=csv_data,
        file_name="tweet_data_dengan_label_sentimen.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 2 - CLEANSING & NORMALISASI
# ============================================================

with tab2:

    st.subheader(
        "Hasil Cleansing dan Normalisasi"
    )

    st.write(
        "Tahapan preprocessing meliputi pengubahan "
        "huruf menjadi kecil, penghapusan URL, mention, "
        "hashtag, angka, karakter khusus, dan normalisasi "
        "kata tidak baku."
    )

    kolom_preprocessing = [
        kolom for kolom in
        [
            "full_text",
            "clean_text",
            "normalisasi_text",
            "sentimen",
        ]
        if kolom in df.columns
    ]

    st.dataframe(
        df[kolom_preprocessing],
        use_container_width=True,
        height=450,
    )

    csv_preprocessing = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download Hasil Cleansing dan Normalisasi",
        data=csv_preprocessing,
        file_name="tweet_data_preprocessing.csv",
        mime="text/csv",
    )

    st.subheader("Kamus Normalisasi")

    kamus_df = pd.DataFrame(
        list(NORMALISASI.items()),
        columns=[
            "Kata Awal",
            "Kata Normalisasi",
        ],
    )

    st.dataframe(
        kamus_df,
        use_container_width=True,
        height=300,
    )

    with st.expander(
        "Lihat kata yang digunakan untuk pelabelan otomatis"
    ):

        st.write("**Kata/frasa positif:**")
        st.write(", ".join(KATA_POSITIF))

        st.write("**Kata/frasa negatif:**")
        st.write(", ".join(KATA_NEGATIF))


# ============================================================
# TAB 3 - PREDIKSI SENTIMEN
# ============================================================

with tab3:

    st.subheader("Prediksi Sentimen Tweet")

    st.write(
        "Masukkan teks yang berkaitan dengan mitigasi "
        "bencana untuk mengetahui prediksi sentimennya."
    )

    teks_input = st.text_area(
        "Masukkan teks:",
        placeholder=(
            "Contoh: Relawan sigap membantu evakuasi "
            "dan kondisi warga mulai pulih."
        ),
        height=140,
    )

    if st.button(
        "🔍 Analisis Sentimen",
        type="primary",
    ):

        if not teks_input.strip():

            st.warning(
                "Silakan masukkan teks terlebih dahulu."
            )

        else:

            # --------------------------------------------
            # PREPROCESSING TEKS INPUT
            # --------------------------------------------

            clean = bersihkan_teks(teks_input)
            normal = normalisasi_teks(clean)

            if not normal:

                st.warning(
                    "Teks kosong setelah proses preprocessing."
                )

            else:

                # ----------------------------------------
                # HITUNG SKOR KATA SENTIMEN
                # ----------------------------------------

                skor_positif = hitung_kata(
                    normal,
                    KATA_POSITIF,
                )

                skor_negatif = hitung_kata(
                    normal,
                    KATA_NEGATIF,
                )

                # ----------------------------------------
                # MENENTUKAN HASIL PREDIKSI
                # ----------------------------------------

                if (
                    skor_positif == 0
                    and skor_negatif == 0
                ):

                    hasil = "netral"
                    gunakan_model = False
                    proba = None

                else:

                    X_input = tfidf.transform(
                        [normal]
                    )

                    hasil_encoded = mlp.predict(
                        X_input
                    )[0]

                    hasil = (
                        label_encoder.inverse_transform(
                            [int(hasil_encoded)]
                        )[0]
                    )

                    proba = mlp.predict_proba(
                        X_input
                    )[0]

                    gunakan_model = True

                # ----------------------------------------
                # HASIL PREPROCESSING
                # ----------------------------------------

                st.divider()

                st.subheader(
                    "Hasil Preprocessing"
                )

                col_a, col_b = st.columns(2)

                with col_a:

                    st.markdown("**Teks asli**")
                    st.write(teks_input)

                with col_b:

                    st.markdown(
                        "**Setelah cleansing**"
                    )

                    st.write(clean)

                st.markdown(
                    "**Setelah normalisasi**"
                )

                st.info(normal)

                # ----------------------------------------
                # HASIL SENTIMEN
                # ----------------------------------------

                st.subheader(
                    "Hasil Prediksi Sentimen"
                )

                st.success(
                    f"### {EMOJI.get(hasil, '💬')} "
                    f"{hasil.upper()}"
                )

                # ----------------------------------------
                # PROBABILITAS MODEL
                # ----------------------------------------

                if gunakan_model:

                    hasil_proba = pd.DataFrame({
                        "Sentimen": [
                            (
                                f"{EMOJI.get(kelas, '')} "
                                f"{kelas}"
                            )
                            for kelas in
                            label_encoder.inverse_transform(
                                mlp.classes_.astype(int)
                            )
                        ],
                        "Probabilitas": proba,
                    })

                    hasil_proba = (
                        hasil_proba.sort_values(
                            "Probabilitas",
                            ascending=False,
                        )
                    )

                    st.subheader(
                        "Probabilitas Sentimen"
                    )

                    st.dataframe(
                        hasil_proba.style.format(
                            {
                                "Probabilitas": "{:.2%}"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    fig, ax = plt.subplots(
                        figsize=(7, 3)
                    )

                    ax.bar(
                        hasil_proba["Sentimen"],
                        hasil_proba["Probabilitas"],
                    )

                    ax.set_ylabel(
                        "Probabilitas"
                    )

                    ax.set_ylim(0, 1)

                    ax.set_title(
                        "Probabilitas Prediksi Sentimen"
                    )

                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Probabilitas merupakan keluaran "
                        "model dan bukan jaminan bahwa "
                        "prediksi selalu benar."
                    )

                else:

                    st.info(
                        "Kalimat tidak mengandung kata "
                        "positif atau negatif berdasarkan "
                        "kamus. Hasil dikategorikan sebagai "
                        "NETRAL."
                    )

                # ----------------------------------------
                # INFORMASI SKOR KAMUS
                # ----------------------------------------

                with st.expander(
                    "Lihat hasil pencocokan kata sentimen"
                ):

                    st.write(
                        f"Jumlah kata positif: {skor_positif}"
                    )

                    st.write(
                        f"Jumlah kata negatif: {skor_negatif}"
                    )


# ============================================================
# TAB 4 - EVALUASI MODEL
# ============================================================

with tab4:

    st.subheader(
        "Evaluasi Model Neural Network"
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Accuracy",
        f"{accuracy:.2%}",
    )

    e2.metric(
        "Precision",
        f"{precision:.2%}",
    )

    e3.metric(
        "Recall",
        f"{recall:.2%}",
    )

    e4.metric(
        "F1-Score",
        f"{f1:.2%}",
    )

    st.caption(
        f"Ukuran vocabulary TF-IDF: "
        f"{len(tfidf.vocabulary_):,} fitur"
    )

    # --------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Aktual {label}"
            for label in labels
        ],
        columns=[
            f"Prediksi {label}"
            for label in labels
        ],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Baris menunjukkan kelas aktual, sedangkan "
        "kolom menunjukkan kelas hasil prediksi."
    )

    # --------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    # --------------------------------------------
    # PERBANDINGAN HASIL PREDIKSI
    # --------------------------------------------

    st.subheader(
        "Perbandingan Label Aktual dan Prediksi"
    )

    hasil_evaluasi = pd.DataFrame({
        "Teks": X_test.values,
        "Label Aktual": y_test.values,
        "Hasil Prediksi": y_pred,
    })

    st.dataframe(
        hasil_evaluasi,
        use_container_width=True,
        height=350,
    )

    csv_evaluasi = hasil_evaluasi.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download Hasil Evaluasi",
        data=csv_evaluasi,
        file_name="hasil_evaluasi_neural_network.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Analisis Sentimen Tweet Mitigasi Bencana | "
    "TF-IDF + Neural Network (MLPClassifier)"
)
