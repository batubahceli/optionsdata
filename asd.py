import pip
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
# 1. PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="VIOP Options Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    cache_dir = Path("temp_data_cache")
    cache_dir.mkdir(exist_ok=True)
    local_file = cache_dir / nas_file.name

    if not local_file.exists():
        try:
            with st.spinner(f"📥 Downloading data from NAS to local cache... ({nas_file.name})"):
                shutil.copy2(nas_file, local_file)
        except Exception as e:
            st.error(f"Failed to copy from NAS: {e}")
            return None

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
    mask = df["Instrument"].str.startswith("O_") | df["Instrument"].str.startswith("TM_O_")
    df = df.loc[mask].copy()

    if df.empty:
        return df

    tm_mask = df['Instrument'].str.startswith('TM_')
    
    df.loc[tm_mask, 'Under'] = df.loc[tm_mask, 'Instrument'].apply(
        lambda x: f"F_{x.split('_')[2][:x.split('_')[2].rfind('E')]}{x.split('_')[2][x.split('_')[2].rfind('E')+3:x.split('_')[2].rfind('E')+7]}"
        if len(x.split('_')) > 2 else np.nan
    )

    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Instrument'].str.extract(r'([A-Z]+_\w+\d{4})')[0]
    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Under'].str[:-5] 
    df.loc[~tm_mask, 'Under'] = df.loc[~tm_mask, 'Under'].apply(lambda x: x.replace('O_', '', 1) if isinstance(x, str) else x)

    def replace_under(row):
        under = row['Under']
        name = row['Instrument']
        if pd.isna(under): return under
        
        if 'XU030' in under or 'USDTRY' in under:
            return under
        
        return under

    df['Under'] = df.apply(replace_under, axis=1)
    df = df[df['Under'].notna()]

    tm_mask = df['Instrument'].str.startswith('TM_') # Refresh mask
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

    _OPT_RE = re.compile(r"^O_(?P<root>[A-Z0-9]+)E(?P<mmyy>\d{4})(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")
    df["temp_sym"] = df["Instrument"].str.replace(r"^TM_", "", regex=True)
    extracted = df["temp_sym"].str.extract(_OPT_RE)
    df = pd.concat([df, extracted], axis=1)
    
    def fmt_expiry(mmyy):
        if pd.isna(mmyy): return np.nan
        s = str(mmyy)
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

def get_summary_stats(df, filter_type="XU030"):
    """
    Returns Top 5 Instruments for a given category with their Top Participants.
    filter_type: "XU030", "USDTRY", or "SSO"
    """
    # 1. Apply Filter
    if filter_type == "XU030":
        subset = df[df["Under"].str.contains("XU030", case=False, na=False)]
    elif filter_type == "USDTRY":
        subset = df[df["Under"].str.contains("USDTRY", case=False, na=False)]
    else: # SSO (Everything else)
        subset = df[
            (~df["Under"].str.contains("XU030", case=False, na=False)) & 
            (~df["Under"].str.contains("USDTRY", case=False, na=False))
        ]
        
    if subset.empty:
        return []

    # 2. Group by Instrument to find Top 5 by Volume
    top_5 = subset.groupby("Instrument")["LOT"].sum().nlargest(5).index.tolist()
    
    results = []
    for inst in top_5:
        # Get data for this instrument
        inst_df = subset[subset["Instrument"] == inst]
        total_vol = inst_df["LOT"].sum()
        
        # Details (Expiry, CP, Strike)
        row = inst_df.iloc[0]
        
        # Find Top Buyer and Top Seller
        top_buyer = inst_df.groupby("ALAN")["LOT"].sum().idxmax()
        buyer_vol = inst_df.groupby("ALAN")["LOT"].sum().max()
        
        top_seller = inst_df.groupby("SATAN")["LOT"].sum().idxmax()
        seller_vol = inst_df.groupby("SATAN")["LOT"].sum().max()
        
        results.append({
            "Instrument": inst,
            "Expiry": row["expiry_ym"],
            "Strike": row["strike"],
            "Type": row["cp"],
            "Total Volume": total_vol,
            "Top Buyer": f"{top_buyer} ({int(buyer_vol):,})",
            "Top Seller": f"{top_seller} ({int(seller_vol):,})"
        })
        
    return pd.DataFrame(results)

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
    # Process Data
    df = process_data(raw_df)
    
    if df.empty:
        st.warning("No Options (O_ / TM_O_) data found.")
    else:
        # ---------------------------------------------------------
        # MAIN LAYOUT: TABS
        # ---------------------------------------------------------
        main_tab1, main_tab2 = st.tabs(["📊 Market Dashboard", "🔎 Instrument Analysis"])
        
        # =========================================================
        # TAB 1: MARKET DASHBOARD (Summary)
        # =========================================================
        with main_tab1:
            st.markdown("### 🔥 Top 5 Most Active Options")
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["XU030 Options", "USDTRY Options", "SSO (Single Stock) Options"])
            
            # --- Helper to display summary tables ---
            def display_summary(filter_name):
                summary_df = get_summary_stats(df, filter_name)
                if summary_df.empty:
                    st.info("No data found for this category.")
                else:
                    # Formatting for cleaner look
                    st.dataframe(
                        summary_df,
                        column_config={
                            "Total Volume": st.column_config.ProgressColumn(
                                "Total Volume", 
                                format="%d", 
                                min_value=0, 
                                max_value=int(summary_df["Total Volume"].max())
                            ),
                            "Type": st.column_config.TextColumn("C/P", width="small"),
                            "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=250
                    )

            with sub_tab1:
                display_summary("XU030")
                
            with sub_tab2:
                display_summary("USDTRY")
                
            with sub_tab3:
                display_summary("SSO")

        # =========================================================
        # TAB 2: INSTRUMENT ANALYSIS (Deep Dive)
        # =========================================================
        with main_tab2:
            st.sidebar.markdown("---")
            st.sidebar.header("Analysis Filters")
            
            # 1. Underlying Filter
            sorted_unders = sorted(df["Under"].unique())
            selected_under = st.sidebar.selectbox("Select Underlying", sorted_unders)
            
            # 2. Expiry Filter
            df_under = df[df["Under"] == selected_under]
            expiries = sorted(df_under["expiry_ym"].unique())
            selected_expiry = st.sidebar.selectbox("Select Expiry", expiries)
            
            # Chart Data
            df_chart = df_under[df_under["expiry_ym"] == selected_expiry].copy()

            # --- PREPARE CHART DATA ---
            # 1. Melt
            df_alan = df_chart[["strike", "cp", "ALAN", "LOT"]].rename(columns={"ALAN": "Participant"})
            df_alan["Side"] = "Buy"
            df_satan = df_chart[["strike", "cp", "SATAN", "LOT"]].rename(columns={"SATAN": "Participant"})
            df_satan["Side"] = "Sell"
            full_stack = pd.concat([df_alan, df_satan], ignore_index=True)

            # 2. Plot Logic
            full_stack["Plot_Val"] = np.where(full_stack["cp"] == "P", -full_stack["LOT"], full_stack["LOT"])
            
            # 3. Dynamic Width & Offset
            unique_strikes = np.sort(full_stack["strike"].unique())
            min_diff = np.min(np.diff(unique_strikes)) if len(unique_strikes) > 1 else 1.0
            offset = min_diff * 0.20
            bar_width = min_diff * 0.35
                
            full_stack["Plot_X"] = np.where(full_stack["Side"] == "Buy", 
                                            full_stack["strike"] - offset, 
                                            full_stack["strike"] + offset)

            # 4. Grouping
            top_n = 15
            top_participants = full_stack.groupby("Participant")["LOT"].sum().nlargest(top_n).index
            full_stack["Participant_Group"] = np.where(full_stack["Participant"].isin(top_participants), full_stack["Participant"], "OTHER")
            
            chart_data = full_stack.groupby(["Plot_X", "strike", "cp", "Side", "Participant_Group"])["Plot_Val"].sum().reset_index()

            # --- RENDER CHART ---
            st.subheader(f"Open Positions: {selected_under} ({selected_expiry})")
            
            fig = px.bar(
                chart_data,
                x="Plot_X",
                y="Plot_Val",
                color="Participant_Group",
                title="<b>Split View: Buyers (Left) vs Sellers (Right)</b><br>(Calls ↑ Positive | Puts ↓ Negative)",
                labels={"Plot_Val": "Volume", "Participant_Group": "Participant"},
                color_discrete_sequence=px.colors.qualitative.Dark24, 
                height=600,
                hover_data={"strike": True, "Side": True, "Plot_X": False}
            )
            
            fig.update_traces(width=bar_width)
            fig.update_layout(
                hovermode="closest",
                xaxis_title="Strike Price",
                yaxis_title="Volume (Lots)",
                bargap=0.0, 
                xaxis=dict(tickmode='array', tickvals=unique_strikes, ticktext=[str(s) for s in unique_strikes])
            )
            # CHANGED TO WHITE BELOW
            fig.add_hline(y=0, line_color="white", line_width=1)

            selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

            # --- DRILL DOWN ---
            if selection and len(selection["selection"]["points"]) > 0:
                point = selection["selection"]["points"][0]
                clicked_x_raw = point["x"]
                clicked_strike = unique_strikes[np.abs(unique_strikes - clicked_x_raw).argmin()]
                clicked_val = point["y"]
                clicked_cp = "C" if clicked_val > 0 else "P"
                side_clicked = "Buyers (ALAN)" if clicked_x_raw < clicked_strike else "Sellers (SATAN)"
                
                st.markdown("---")
                st.markdown(f"### 🔎 Analysis: Strike {clicked_strike} ({clicked_cp})")
                st.caption(f"Clicked: **{side_clicked}**")
                
                mask_drill = (df_chart["strike"] == clicked_strike) & (df_chart["cp"] == clicked_cp)
                df_drill = df_chart[mask_drill]
                
                if not df_drill.empty:
                    details_df = get_participant_details(df_drill)
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**Participant Price Map**")
                        plot_df = details_df.reset_index()
                        if "Participant" not in plot_df.columns and "index" in plot_df.columns:
                            plot_df = plot_df.rename(columns={"index": "Participant"})

                        scatter_fig = go.Figure()
                        scatter_fig.add_trace(go.Scatter(
                            x=plot_df[plot_df["Buy_Lot"]>0]["Participant"], 
                            y=plot_df[plot_df["Buy_Lot"]>0]["Buy_Price"],
                            mode='markers', name='Buy Price',
                            marker=dict(color='green', symbol='triangle-up', size=12),
                            text=plot_df[plot_df["Buy_Lot"]>0]["Buy_Lot"],
                            hovertemplate="<b>%{x}</b><br>Buy: %{y:.2f}<br>Vol: %{text}"
                        ))
                        scatter_fig.add_trace(go.Scatter(
                            x=plot_df[plot_df["Sell_Lot"]>0]["Participant"], 
                            y=plot_df[plot_df["Sell_Lot"]>0]["Sell_Price"],
                            mode='markers', name='Sell Price',
                            marker=dict(color='red', symbol='triangle-down', size=12),
                            text=plot_df[plot_df["Sell_Lot"]>0]["Sell_Lot"],
                            hovertemplate="<b>%{x}</b><br>Sell: %{y:.2f}<br>Vol: %{text}"
                        ))
                        scatter_fig.update_layout(height=450, xaxis_title="Participant", yaxis_title="Price (TL)")
                        st.plotly_chart(scatter_fig, use_container_width=True)

                    with col2:
                        st.markdown("**Detailed Breakdown**")
                        display_df = details_df[["Buy_Lot", "Buy_Price", "Sell_Lot", "Sell_Price"]].copy()
                        st.dataframe(
                            display_df,
                            column_config={
                                "Buy_Lot": st.column_config.ProgressColumn("Buy Vol", format="%d", min_value=0, max_value=int(display_df["Buy_Lot"].max()) if not display_df.empty else 100),
                                "Sell_Lot": st.column_config.ProgressColumn("Sell Vol", format="%d", min_value=0, max_value=int(display_df["Sell_Lot"].max()) if not display_df.empty else 100),
                                "Buy_Price": st.column_config.NumberColumn("Avg Buy Price", format="%.2f ₺"),
                                "Sell_Price": st.column_config.NumberColumn("Avg Sell Price", format="%.2f ₺"),
                            },
                            height=450,
                            use_container_width=True
                        )

else:
    st.info("Please select a Viop Defterfile from '\\nas2\SHARED\datav2\bistZamanSatis' in the sidebar to begin.")

