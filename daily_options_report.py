import streamlit as st
from datetime import date, timedelta
import base64

# =========================================================
# PAGE SETUP
# =========================================================
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

def display_pdf(pdf_bytes: bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_html = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="850px"
            style="border: none; border-radius: 8px;"
        ></iframe>
    """
    st.markdown(pdf_html, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
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
st.sidebar.caption(f"📎 Beklenen dosya adı:")
st.sidebar.code(expected_filename, language=None)

uploaded_file = st.sidebar.file_uploader(
    "PDF Dosyasını Yükle",
    type=["pdf"],
    help=f"Bilgisayarından {expected_filename} dosyasını seç"
)

# =========================================================
# MAIN AREA
# =========================================================
st.markdown("## 📈 Daily Options Report")
st.markdown("---")

if uploaded_file is not None:
    if uploaded_file.name != expected_filename:
        st.warning(
            f"⚠️ Yüklenen dosya adı (**{uploaded_file.name}**) seçilen tarih/dil ile eşleşmiyor.\n\n"
            f"Beklenen: **{expected_filename}**\n\n"
            f"Yine de görüntülemek için devam edebilirsiniz."
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
    display_pdf(pdf_bytes)

else:
    st.info(
        f"👈 Soldan tarih ve dil seçin, ardından **{expected_filename}** dosyasını yükleyin."
    )
