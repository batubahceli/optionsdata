import streamlit as st
from datetime import date, timedelta
import base64
import fitz  # pymupdf

st.set_page_config(
    page_title="Daily Options Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def build_filename(selected_date: date, lang: str) -> str:
    year  = selected_date.year
    month = MONTH_MAP[selected_date.month]
    day   = selected_date.day
    return f"OptDailyAuto_{year}_{month}_{day:02d}_{lang}.pdf"

def display_pdf_as_images(pdf_bytes: bytes):
    """PDF sayfalarını resim olarak gösterir."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(3, 3)  # 2x zoom - yüksek kalite
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, caption=f"Sayfa {page_num + 1}", use_container_width=False)
    doc.close()

# SIDEBAR
st.sidebar.header("📄 Rapor Seçimi")

selected_date = st.sidebar.date_input(
    "Tarih Seç",
    value=date.today() - timedelta(days=1),
    max_value=date.today()
)

lang = st.sidebar.radio(
    "Dil / Language",
    options=["TR", "EN"],
    horizontal=True
)

st.sidebar.markdown("---")

expected_filename = build_filename(selected_date, lang)
st.sidebar.caption("📎 Beklenen dosya adı:")
st.sidebar.code(expected_filename, language=None)

uploaded_file = st.sidebar.file_uploader(
    "PDF Dosyasını Yükle",
    type=["pdf"]
)

# MAIN AREA
st.markdown("## 📈 Daily Options Report")
st.markdown("---")

if uploaded_file is not None:
    if uploaded_file.name != expected_filename:
        st.warning(
            f"⚠️ Yüklenen: **{uploaded_file.name}**\n\n"
            f"Beklenen: **{expected_filename}**"
        )

    pdf_bytes = uploaded_file.read()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ **{uploaded_file.name}** yüklendi.")
    with col2:
        st.download_button(
            label="⬇️ PDF İndir",
            data=pdf_bytes,
            file_name=uploaded_file.name,
            mime="application/pdf"
        )

    st.markdown("---")

    with st.spinner("PDF sayfaları yükleniyor..."):
        display_pdf_as_images(pdf_bytes)

else:
    st.info(
        f"👈 Soldan tarih ve dil seçin, ardından **{expected_filename}** dosyasını yükleyin."
    )
