import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
import shutil
import os
from datetime import datetime, date

# =========================================================
# 1. PAGE SETUP & CONSTANTS
# =========================================================
st.set_page_config(
    page_title="VIOP Options Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Colors for Top Participants (Consistent coloring across charts)
PARTICIPANT_COLORS = {
    "YAPI KREDI": "#004990",  # Dark Blue
    "IS YATIRIM": "#1C3F95",  # Blue
    "B of A": "#DC143C",      # Red (Bank of America)
    "TEB": "#009639",         # Green
    "AK YATIRIM": "#E2001A",  # Red
    "GARANTI": "#005F27",     # Dark Green
    "INFO": "#E4002B",        # Red
    "TACIRLER": "#FFD700",    # Gold
    "OTHER": "#D3D3D3"        # Light Grey
}

# =========================================================
# 2. DATA LOADING & PROCESSING
# =========================================================

def normalize_yyyymmdd(d):
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y%m%d")
    s = str(d).strip()
    if re.fullmatch(r"\d{8}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file_source, trade_date=None):
    # A. Direct Upload
    if hasattr(file_source, "read"):
        try:
            return pd.read_csv(
                file_source, 
                sep=";", 
                engine="c", 
                low_memory=False,
                usecols=["SEMBOL", "LOT", "TL", "ALAN", "SATAN", "FIYAT"]
            )
        except Exception as e:
            st.error(f"Error reading upload: {e}")
            return None

    # B. NAS Loading with Local Caching
    yyyymmdd = normalize_yyyymmdd(trade_date)
    base_dir = r"\\nas2\SHARED\datav2\bistZamanSatis"
    
    nas_file = None
    for ext in ["", ".csv", ".txt"]:
        candidate = Path(base_dir) / f"ViopDefter{yyyymmdd}{ext}"
        if candidate.exists():
            nas_file = candidate
            break
            
    if nas_file is None:
        return None

    # Create temp cache folder to speed up subsequent loads
    cache_dir = Path("temp_data_cache")
    cache_dir.mkdir(exist_ok=True)
    local_file = cache_dir / nas_file.name

    # Copy if not exists
    if not local_file.exists():
        try:
            with st.spinner(f"📥 Downloading data from NAS to local cache... ({nas_file.name})"):
                shutil.copy2(nas_file, local_file)
        except Exception as e:
            st.error(f"Failed to copy from NAS: {e}")
            return None

    # Read from Local Disk
    try:
        required_cols = ["SEMBOL", "LOT", "TL", "ALAN", "SATAN", "FIYAT"]
        df = pd.read_csv(
            local_file,
            sep=";",
            usecols=lambda c: c in required_cols, 
            engine="c",            
            dtype={                
                "SEMBOL": "string",
                "ALAN": "string", 
                "SATAN": "string",
                "LOT": "float32",
                "TL": "float32",
                "FIYAT": "float32"
            },
            low_memory=False
        )
        return df
    except Exception as e:
        st.error(f"Error reading local cached file: {e}")
        return None

@st.cache_data
def process_data(df, n1_list_to_modify=None):
    if df is None or df.empty:
        return pd.DataFrame()

    if n1_list_to_modify is None:
        n1_list_to_modify = []

    if "SEMBOL" in df.columns:
        df = df.rename(columns={"SEMBOL": "Instrument"})
    
    df["Instrument"] = df["Instrument"].astype(str)
    # Filter for Options only
    mask = df["Instrument"].str.startswith("O_") | df["Instrument"].str.startswith("TM_O_")
    df = df.loc[mask].copy()

    if df.empty:
        return df

    # --- Underlying Extraction Logic ---
    tm_mask = df['Instrument'].str.startswith('TM_')
    
    # TM_ Logic
    df.loc[tm_mask, 'Under'] = df.loc[tm_mask, 'Instrument'].apply(
        lambda x: f"F_{x.split('_')[2][:x.split('_')[2].rfind('E')]}{x.split('_')[2][x.split('_')[2].rfind('E')+3:x.split('_')[2].rfind('E')+7]}"
        if len(x.split('_')) > 2 else np.nan
    )

    # Standard Logic
    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Instrument'].str.extract(r'([A-Z]+_\w+\d{4})')[0]
    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Under'].str[:-5] 
    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Under'].apply(lambda x: x.replace('O_', '', 1) if isinstance(x, str) else x)

    # Advanced Fixes (XU030 / USDTRY / Dates)
    def replace_under(row):
        under = row['Under']
        name = row['Instrument']
        if pd.isna(under): return under
        
        if 'XU030' in under or 'USDTRY' in under:
            return under
        
        if '0326' in under and '0326' in name:
            return under.replace('F_', '').replace('0326', '.E')
        if '1224' in under and '1224' in name:
            return under.replace('F_', '').replace('1224', '.E')
            
        return under

    df['Under'] = df.apply(replace_under, axis=1)
    df = df[df['Under'].notna()]

    # Specific Fixes
    tm_mask = df['Instrument'].str.startswith('TM_')
    mask_usdtry = df['Instrument'].str.contains('USDTRY', case=False, na=False)
    tm_usdtry = tm_mask & mask_usdtry
    df.loc[tm_usdtry, 'Under'] = df.loc[tm_usdtry, 'Under'].str.replace(r'(USDTRY)K', r'\1', regex=True)
    df.loc[mask_usdtry, 'Under'] = df.loc[mask_usdtry, 'Under'].apply(
        lambda x: x[:-5].replace('KE', '') if 'KE' in x else x
    )

    mask_xu30_len20 = df['Instrument'].str.contains('XU030', case=False, na=False) & (df['Instrument'].str.len() == 20)
    mask_xu30_len21 = df['Instrument'].str.contains('XU030', case=False, na=False) & (df['Instrument'].str.len() == 21)
    df.loc[mask_xu30_len20, 'Under'] = df.loc[mask_xu30_len20, 'Under'].apply(lambda x: x[:-4].replace('E', '') if 'E' in x else x)
    df.loc[mask_xu30_len21, 'Under'] = df.loc[mask_xu30_len21, 'Under'].apply(lambda x: x[:-5].replace('E', '') if 'E' in x else x)

    if n1_list_to_modify:
        mod_mask = df['Under'].isin(n1_list_to_modify)
        df.loc[mod_mask, 'Under'] = df.loc[mod_mask, 'Under'] + 'N1'

    # Extract Option Details (Strike, CP, Date)
    _OPT_RE = re.compile(r"^O_(?P<root>[A-Z0-9]+)E(?P<mmyy>\d{4})(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")
    df["temp_sym"] = df["Instrument"].str.replace(r"^TM_", "", regex=True)
    extracted = df["temp_sym"].str.extract(_OPT_RE)
    df = pd.concat([df, extracted], axis=1)
    
    def fmt_expiry(mmyy):
        if pd.isna(mmyy): return np.nan
        s = str(mmyy)
        # Convert YYMM -> YYYY-MM
        return f"{2000+int(s[2:]):04d}-{int(s[:2]):02d}"

    df["expiry_ym"] = df["mmyy"].apply(fmt_expiry)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    
    return df.dropna(subset=["strike", "cp", "expiry_ym"])

def get_participant_details(df_subset):
    buy_stats = df_subset.groupby("ALAN").agg(Buy_Lot=("LOT", "sum"), Buy_TL=("TL", "sum"))
    buy_stats["Buy_Price"] = buy_stats["Buy_TL"] / buy_stats["Buy_Lot"]
    
    sell_stats = df_subset.groupby("SATAN").agg(Sell_Lot=("LOT", "sum"), Sell_TL=("TL", "sum"))
    sell_stats["Sell_Price"] = sell_stats["Sell_TL"] / sell_stats["Sell_Lot"]
    
    combined = pd.merge(buy_stats, sell_stats, left_index=True, right_index=True, how="outer").fillna(0)
    combined.index.name = "Participant"
    combined["Total_Activity"] = combined["Buy_Lot"] + combined["Sell_Lot"]
    
    return combined.sort_values("Total_Activity", ascending=False)

def get_sector_data(df, filter_type):
    """
    Returns data subset and calculated stats for a specific sector.
    Includes custom right-chart logic based on the sector type.
    """
    if filter_type == "XU030":
        subset = df[df["Under"].str.contains("XU030", case=False, na=False)]
    elif filter_type == "USDTRY":
        subset = df[df["Under"].str.contains("USDTRY", case=False, na=False)]
    else: # SSO
        subset = df[
            (~df["Under"].str.contains("XU030", case=False, na=False)) & 
            (~df["Under"].str.contains("USDTRY", case=False, na=False))
        ]
    
    if subset.empty:
        return None, None, None, None, None, None, None

    # 1. Stats
    total_vol = subset["LOT"].sum()
    call_vol = subset[subset["cp"]=="C"]["LOT"].sum()
    put_vol = subset[subset["cp"]=="P"]["LOT"].sum()
    
    # 2. Top Participants (Sector Specific)
    p_buy = subset.groupby("ALAN")["LOT"].sum()
    p_sell = subset.groupby("SATAN")["LOT"].sum()
    p_total = p_buy.add(p_sell, fill_value=0).sort_values(ascending=False).nlargest(10).reset_index()
    p_total.columns = ["Participant", "Volume"]

    # 3. Right Chart Data (Dynamic based on Sector)
    if filter_type == "SSO":
        # Make a copy to avoid SettingWithCopyWarning
        subset = subset.copy()
        
        # Create Label: EKGYO + MMYY (e.g. "EKGYO 0126")
        # expiry_ym is YYYY-MM (e.g., 2026-01)
        # We extract MM (5:7) and YY (2:4)
        subset["Display_Label"] = subset["Under"] + " " + subset["expiry_ym"].str[5:7] + subset["expiry_ym"].str[2:4]
        
        # Group by this new label
        right_chart_data = subset.groupby("Display_Label")["LOT"].sum().nlargest(10).reset_index()
        right_chart_type = "Underlying Expiry"
        
    else:
        # For XU030/USDTRY, show Volume by Expiry (e.g., Feb 2026, Mar 2026)
        right_chart_data = subset.groupby("expiry_ym")["LOT"].sum().reset_index().sort_values("expiry_ym")
        
        # Convert YYYY-MM to nice Month-Year Label (e.g., "Feb 2026")
        right_chart_data["Label"] = pd.to_datetime(right_chart_data["expiry_ym"] + "-01").dt.strftime("%b %Y")
        
        right_chart_type = "Expiry"

    return subset, total_vol, call_vol, put_vol, p_total, right_chart_data, right_chart_type

# =========================================================
# 3. MAIN APP LOGIC
# =========================================================

st.sidebar.header("Data Source")
input_method = st.sidebar.radio("Select Input Method", ["Upload File", "NAS Path (Local Network)"])

raw_df = None

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload ViopDefter .csv/.txt", type=["csv", "txt"])
    if uploaded_file:
        raw_df = load_data(uploaded_file)
else:
    trade_date = st.sidebar.date_input("Trade Date", value=date.today())
    if st.sidebar.button("Load from NAS"):
        with st.spinner("Connecting..."):
            raw_df = load_data(None, trade_date=trade_date)
            if raw_df is None:
                st.error("File not found on NAS.")

if raw_df is not None:
    df = process_data(raw_df)
    
    if df.empty:
        st.warning("No Options (O_ / TM_O_) data found.")
    else:
        # TABS
        main_tab1, main_tab2 = st.tabs(["📊 Market Dashboard", "🔎 Instrument Analysis"])
        
        # =========================================================
        # TAB 1: MACRO DASHBOARD
        # =========================================================
        with main_tab1:
            # 1. Global Header
            tot = df["LOT"].sum()
            cv = df[df["cp"]=="C"]["LOT"].sum()
            pv = df[df["cp"]=="P"]["LOT"].sum()
            cp_r = pv/cv if cv>0 else 0
            
            st.markdown("### 🌍 Global Market Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Volume", f"{int(tot):,}")
            c2.metric("Call Volume", f"{int(cv):,}", delta="Bullish" if cv>pv else None)
            c3.metric("Put Volume", f"{int(pv):,}", delta="-Bearish" if pv>cv else None, delta_color="inverse")
            c4.metric("Put/Call Ratio", f"{cp_r:.2f}")
            st.markdown("---")

            # 2. Sector Deep Dives (Tabs)
            st.markdown("### 🔬 Sector Analysis")
            s_tab1, s_tab2, s_tab3 = st.tabs(["XU030 (Index)", "USDTRY (FX)", "SSO (Stocks)"])

            def render_sector_dashboard(filter_name):
                subset, s_tot, s_cv, s_pv, s_parts, s_right_data, s_right_type = get_sector_data(df, filter_name)
                
                if subset is None:
                    st.info(f"No data for {filter_name}")
                    return

                # A. Sector KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric(f"{filter_name} Volume", f"{int(s_tot):,}")
                k2.metric("Sector P/C Ratio", f"{(s_pv/s_cv if s_cv>0 else 0):.2f}")
                k3.caption("Ratio < 1.0 = Bullish (More Calls)\nRatio > 1.0 = Bearish (More Puts)")
                
                # B. Charts Row
                c_left, c_right = st.columns(2)
                
                with c_left:
                    st.subheader(f"🏆 Top Participants")
                    # Use Consistent Coloring
                    fig_p = px.bar(
                        s_parts, 
                        x="Volume", 
                        y="Participant", 
                        orientation='h', 
                        text_auto='.2s',
                        color="Participant", # Map color by name
                        color_discrete_map=PARTICIPANT_COLORS # Use global dict
                    )
                    fig_p.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                    st.plotly_chart(fig_p, use_container_width=True)
                    
                with c_right:
                    st.subheader(f"📊 Most Active {s_right_type}")
                    if not s_right_data.empty:
                        # Determine X and Y based on type
                        if s_right_type == "Expiry":
                            x_col = "Label"  # Use formatted "Feb 2026"
                        else:
                            x_col = "Display_Label" # Use "EKGYO 0126"
                        
                        fig_r = px.bar(
                            s_right_data, 
                            x=x_col, 
                            y="LOT", 
                            text_auto='.2s',
                            labels={x_col: "Contract", "LOT": "Volume"},
                            color="LOT",
                            color_continuous_scale="Viridis"
                        )
                        
                        # FORCE CATEGORICAL AXIS: Stops "daily" ticks like Jan 11, Jan 25
                        fig_r.update_xaxes(type='category')
                        
                        # Ensure Correct Order for Expiries
                        if s_right_type == "Expiry":
                            fig_r.update_layout(xaxis={'categoryorder':'array', 'categoryarray': s_right_data["Label"].tolist()})
                        else:
                            # For SSO, sort by Volume descending
                            fig_r.update_layout(xaxis={'categoryorder':'total descending'})
                            
                        fig_r.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_r, use_container_width=True)
                    else:
                        st.info("Not enough data.")

                # C. Top Contracts List
                st.markdown(f"**🔥 Top 5 Active {filter_name} Contracts**")
                top_5 = subset.groupby("Instrument")["LOT"].sum().nlargest(5).index.tolist()
                results = []
                for inst in top_5:
                    inst_df = subset[subset["Instrument"] == inst]
                    row = inst_df.iloc[0]
                    tb = inst_df.groupby("ALAN")["LOT"].sum().idxmax()
                    ts = inst_df.groupby("SATAN")["LOT"].sum().idxmax()
                    results.append({
                        "Instrument": inst, "Expiry": row["expiry_ym"], "Strike": row["strike"], "Type": row["cp"],
                        "Volume": inst_df["LOT"].sum(), "Top Buyer": tb, "Top Seller": ts
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            with s_tab1: render_sector_dashboard("XU030")
            with s_tab2: render_sector_dashboard("USDTRY")
            with s_tab3: render_sector_dashboard("SSO")

        # =========================================================
        # TAB 2: INSTRUMENT ANALYSIS
        # =========================================================
        with main_tab2:
            st.sidebar.markdown("---")
            st.sidebar.header("Drill Down Filters")
            
            # 1. Sorted Dropdowns (A-Z)
            sorted_unders = sorted(df["Under"].unique())
            sel_under = st.sidebar.selectbox("Underlying", sorted_unders)
            
            df_u = df[df["Under"] == sel_under]
            # Sorted Chronologically
            expiries = sorted(df_u["expiry_ym"].unique())
            sel_exp = st.sidebar.selectbox("Expiry", expiries)
            
            df_c = df_u[df_u["expiry_ym"] == sel_exp].copy()

            # --- Butterfly Chart Prep ---
            # Buyers (Negative X) vs Sellers (Positive X)
            df_a = df_c[["strike", "cp", "ALAN", "LOT"]].rename(columns={"ALAN":"Participant"})
            df_a["Side"] = "Buy"
            df_s = df_c[["strike", "cp", "SATAN", "LOT"]].rename(columns={"SATAN":"Participant"})
            df_s["Side"] = "Sell"
            
            full = pd.concat([df_a, df_s], ignore_index=True)

            # X Logic: Volume
            full["Plot_X"] = np.where(full["Side"]=="Buy", -full["LOT"], full["LOT"])
            
            # Y Logic: Strike + Offset (Call Up, Put Down)
            strikes = np.sort(full["strike"].unique())
            min_diff = np.min(np.diff(strikes)) if len(strikes)>1 else 1.0
            y_off = min_diff * 0.20
            
            full["Plot_Y"] = np.where(full["cp"]=="C", full["strike"]+y_off, full["strike"]-y_off)

            # Coloring Logic
            top_p = full.groupby("Participant")["LOT"].sum().nlargest(15).index
            full["Group"] = np.where(full["Participant"].isin(top_p), full["Participant"], "OTHER")
            
            chart_data = full.groupby(["Plot_Y", "strike", "cp", "Side", "Group"])["Plot_X"].sum().reset_index()

            st.subheader(f"Butterfly Analysis: {sel_under} ({sel_exp})")
            
            fig = px.bar(
                chart_data, 
                x="Plot_X", 
                y="Plot_Y", 
                color="Group", 
                orientation='h',
                title="<b>Buyers (Left) vs Sellers (Right)</b> | Top Bar: Call (C) - Bottom Bar: Put (P)",
                color_discrete_map=PARTICIPANT_COLORS, 
                height=700,
                hover_data={"strike":True, "cp":True}
            )
            
            fig.update_layout(
                yaxis=dict(
                    tickmode='array', 
                    tickvals=strikes, 
                    ticktext=[str(s) for s in strikes], 
                    title="Strike Price"
                ),
                xaxis_title="Volume ( ← Buy | Sell → )", 
                bargap=0.1, 
                hovermode="y unified"
            )
            
            # White Zero Line
            fig.add_vline(x=0, line_color="white", line_width=1)
            
            sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

            # --- Drill Down Details ---
            if sel and len(sel["selection"]["points"]) > 0:
                pt = sel["selection"]["points"][0]
                click_y, click_x = pt["y"], pt["x"]
                
                # Reverse Map Y to Strike
                c_strike = strikes[np.abs(strikes - click_y).argmin()]
                c_cp = "C" if click_y > c_strike else "P"
                c_side = "Buyers" if click_x < 0 else "Sellers"
                
                st.markdown("---")
                st.markdown(f"#### 🔎 Strike Details: {c_strike} {c_cp} ({c_side})")
                
                drill = df_c[(df_c["strike"]==c_strike) & (df_c["cp"]==c_cp)]
                
                if not drill.empty:
                    det = get_participant_details(drill)
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.caption("Participant Avg Price Map")
                        plot_df = det.reset_index().rename(columns={"index":"Participant"})
                        
                        f = go.Figure()
                        f.add_trace(go.Scatter(
                            x=plot_df[plot_df["Buy_Lot"]>0]["Participant"], 
                            y=plot_df[plot_df["Buy_Lot"]>0]["Buy_Price"], 
                            mode='markers', 
                            marker=dict(color='green', symbol='triangle-up', size=10), 
                            name="Buy Price",
                            hovertemplate="%{x}<br>Buy: %{y:.2f}"
                        ))
                        f.add_trace(go.Scatter(
                            x=plot_df[plot_df["Sell_Lot"]>0]["Participant"], 
                            y=plot_df[plot_df["Sell_Lot"]>0]["Sell_Price"], 
                            mode='markers', 
                            marker=dict(color='red', symbol='triangle-down', size=10), 
                            name="Sell Price",
                            hovertemplate="%{x}<br>Sell: %{y:.2f}"
                        ))
                        f.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Participant", yaxis_title="Price")
                        st.plotly_chart(f, use_container_width=True)
                        
                    with c2:
                        st.caption("Volume & Price Breakdown")
                        st.dataframe(
                            det[["Buy_Lot", "Buy_Price", "Sell_Lot", "Sell_Price"]], 
                            height=400, 
                            use_container_width=True,
                            column_config={
                                "Buy_Lot": st.column_config.ProgressColumn("Buy Vol", format="%d", min_value=0, max_value=int(det["Buy_Lot"].max())),
                                "Sell_Lot": st.column_config.ProgressColumn("Sell Vol", format="%d", min_value=0, max_value=int(det["Sell_Lot"].max())),
                                "Buy_Price": st.column_config.NumberColumn("Avg Buy", format="%.2f"),
                                "Sell_Price": st.column_config.NumberColumn("Avg Sell", format="%.2f"),
                            }
                        )
else:
    st.info("Please select a Data Source in the sidebar to begin.")
