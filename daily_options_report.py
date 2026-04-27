import streamlit as st
from pathlib import Path
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

# =========================================================
# CONSTANTS
# =========================================================
NAS_DIR = Path(r"\\nas2\SHARED\batuhan\daily_reports")

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def build_filename(selected_date: date, lang: str) -> str:
    """Builds the PDF filename from date and language."""
    year  = selected_date.year
    month = MONTH_MAP[selected_date.month]
    day   = selected_date.day
    return f"OptDailyAuto_{year}_{month}_{day:02d}_{lang}.pdf"

def load_pdf(file_path: Path) -> bytes | None:
    """Reads PDF bytes from NAS path."""
    try:
        return file_path.read_bytes()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        return None

def display_pdf(pdf_bytes: bytes):
    """Renders PDF inline using an iframe."""
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
# SIDEBAR — CONTROLS
# =========================================================
st.sidebar.header("📄 Rapor Seçimi")

# Date picker — default: yesterday (reports are usually for previous day)
selected_date = st.sidebar.date_input(
    "Tarih Seç",
    value=date.today() - timedelta(days=1),
    max_value=date.today()
)

# Language selector
lang = st.sidebar.radio(
    "Dil / Language",
    options=["TR", "EN"],
    horizontal=True
)

load_btn = st.sidebar.button("📥 Raporu Getir", use_container_width=True)

# =========================================================
# MAIN AREA
# =========================================================
st.markdown("## 📈 Daily Options Report")
st.markdown("---")

if load_btn:
    filename = build_filename(selected_date, lang)
    file_path = NAS_DIR / filename

    st.caption(f"📁 Aranan dosya: `{file_path}`")

    with st.spinner("NAS'tan rapor yükleniyor..."):
        pdf_bytes = load_pdf(file_path)

    if pdf_bytes:
        st.success(f"✅ Rapor bulundu: **{filename}**")
        
        # Download button
        st.download_button(
            label="⬇️ PDF İndir",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf"
        )
        
        st.markdown("---")
        display_pdf(pdf_bytes)
    else:
        st.warning(
            f"⚠️ **{filename}** bulunamadı.\n\n"
            f"- Seçilen tarihte rapor olmayabilir (hafta sonu / tatil)\n"
            f"- NAS bağlantısını kontrol edin\n"
            f"- Dosya adı formatını kontrol edin"
        )
else:
    st.info("👈 Soldan tarih ve dil seçip **'Raporu Getir'** butonuna tıklayın.")
