import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import shutil
from datetime import date

# =========================================================
# 1. PAGE SETUP & CONSTANTS
# =========================================================
st.set_page_config(
    page_title="Daily Opportunity Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minor tweaks (like making metrics look better)
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DATA PROCESSING & HELPER FUNCTIONS
# =========================================================

def style_fig(fig, height=550):
    """Applies clean, modern, and highly readable styling to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(size=14, color="#333333"),
        title_font=dict(size=22, color="#111111", family="Arial, sans-serif"),
        legend=dict(
            font=dict(size=14, color="#333333"),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=70, b=70, l=40, r=40),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
    )
    
    # Modern axis styling for bar charts
    fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=15, family="Arial, sans-serif"))
    fig.update_yaxes(tickfont=dict(size=13), title_font=dict(size=15, family="Arial, sans-serif"))
    
    # Force label + NUMBER + percentage on pie charts
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,.0f} (%{percent})",
        textposition='auto',
        selector=dict(type='pie'),
        hovertemplate="<b>%{label}</b><br>Amount: %{value:,.0f}<br>Share: %{percent}<extra></extra>"
    )
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(file_source, trade_date=None):
    if file_source is not None:
        try:
            if file_source.name.endswith('.csv'):
                return pd.read_csv(file_source, low_memory=False)
            elif file_source.name.endswith('.xlsx'):
                return pd.read_excel(file_source)
        except Exception as e:
            st.error(f"Error reading upload: {e}")
            return None

    if trade_date is None:
        return None
        
    date_str = trade_date.strftime("%d_%b_%y")
    base_dir = Path(r"\\nas2\SHARED\batuhan\daily_analysis")
    file_prefix = f"DailyOpportunityAnalysis_{date_str}"
    
    nas_file = None
    for ext in [".csv", ".xlsx"]:
        candidate = base_dir / f"{file_prefix}{ext}"
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
            with st.spinner(f"📥 Downloading {nas_file.name} from NAS..."):
                shutil.copy2(nas_file, local_file)
        except Exception as e:
            st.error(f"Failed to copy from NAS: {e}")
            return None

    try:
        if local_file.suffix == '.csv':
            return pd.read_csv(local_file, low_memory=False)
        else:
            return pd.read_excel(local_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def process_data(df):
    if df is None or df.empty:
        return pd.DataFrame()
        
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
    else:
        df['Date'] = date.today()

    num_cols = ['Quantity', 'Price', 'Opportunity Size', 'Opportunity Size_EoD',
                'EoD_Vega_PnL', 'EoD_Theta_PnL', 'Delta', 'Edge', 'Credit']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    for col in ['Is Active', 'Is Own']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False})
            else:
                df[col] = df[col].astype(bool)

    if 'Instrument' in df.columns:
        df['Instrument Group'] = df['Instrument'].apply(
            lambda x: '0226' if '0226' in str(x) else
                      '0326' if '0326' in str(x) else
                      '0426' if '0426' in str(x) else 'Other'
        )
        
    if 'Underlying Instrument' in df.columns and 'Quantity' in df.columns and 'Price' in df.columns:
        conditions = [
            df['Underlying Instrument'].str.contains('XU030', na=False),
            df['Underlying Instrument'].str.contains('USD', na=False)
        ]
        choices = [
            df['Quantity'] * df['Price'] * 10,
            df['Quantity'] * df['Price'] * 1
        ]
        df['Volume'] = np.select(conditions, choices, default=df['Quantity'] * df['Price'] * 100)
        
    return df

# =========================================================
# 3. INTERACTIVE PLOTTING FUNCTIONS
# =========================================================

def plot_overview(df, plot_date):
    # Top Level Metrics
    st.markdown("### 🎯 Executive Summary")
    total_opp = df['Opportunity Size'].sum()
    total_captured = df.loc[df['Is Own'], 'Opportunity Size'].sum()
    capture_rate = (total_captured / total_opp * 100) if total_opp > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Opportunity Size", f"{total_opp:,.0f}")
    m2.metric("Total Captured (TW)", f"{total_captured:,.0f}")
    m3.metric("Overall Capture Rate", f"{capture_rate:.1f}%")
    m4.metric("Total Trades Analyzed", f"{len(df):,.0f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    c1, c2 = st.columns(2)
    
    # 1. Is Active
    df_active = df.groupby('Is Active')['Opportunity Size'].sum().reset_index()
    df_active['Is Active'] = df_active['Is Active'].map({True: 'Hidden', False: 'Dimer'})
    
    fig1 = px.pie(df_active, names='Is Active', values='Opportunity Size',
                  title="<b>Opportunity: Hidden vs Dimer</b>",
                  color_discrete_sequence=['#82ca9d', '#8884d8'])
    fig1 = style_fig(fig1, height=450)
    c1.plotly_chart(fig1, use_container_width=True)

    # 2. Is Own
    df_own = df.groupby('Is Own')['Opportunity Size'].sum().reset_index()
    df_own['Is Own'] = df_own['Is Own'].map({True: 'Captured by TW', False: 'Not Captured'})
    
    fig2 = px.pie(df_own, names='Is Own', values='Opportunity Size',
                  title="<b>Total Opportunity Captured</b>",
                  color_discrete_sequence=['#4caf50', '#f44336'])
    fig2 = style_fig(fig2, height=450)
    c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    
    # 3. Top 5 Instruments
    l5 = df.groupby('Instrument')['Opportunity Size'].sum().nlargest(5).reset_index()
    other_opp = df.loc[~df['Instrument'].isin(l5['Instrument']), 'Opportunity Size'].sum()
    l5 = pd.concat([l5, pd.DataFrame({'Instrument': ['Others'], 'Opportunity Size': [other_opp]})], ignore_index=True)
    
    c3, c4, c5 = st.columns([1, 3, 1]) 
    fig3 = px.pie(l5, names='Instrument', values='Opportunity Size',
                  title="<b>Top 5 Instruments by Opportunity Size</b>",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig3 = style_fig(fig3, height=600)
    c4.plotly_chart(fig3, use_container_width=True)

def plot_capture_rates_grid(df, plot_date):
    filtered_df = df[~df['Underlying Instrument'].str.contains('XU030|USDTRY', na=False)]
    subsets = {
        'Total Market': (df, ['#4caf50', '#ff9800']),
        'ONLY SSOs': (filtered_df, ['#00bcd4', '#ffeb3b']),
        'XU030': (df[df['Underlying Instrument'].str.contains('XU030', na=False)], ['#8bc34a', '#e91e63']),
        'USDTRY': (df[df['Underlying Instrument'].str.contains('USDTRY', na=False)], ['#3f51b5', '#ff5722']),
    }

    for group_name, (df_group, colors) in subsets.items():
        st.markdown(f"### 📌 **{group_name}**")
        c1, c2 = st.columns(2)
        grouped = df_group.groupby(['Is Active', 'Is Own'])['Opportunity Size'].sum().reset_index()
        
        for active_state, col_obj in zip([True, False], [c1, c2]):
            subset = grouped[grouped['Is Active'] == active_state].copy()
            state_label = 'Hidden' if active_state else 'Dimer'
            
            if subset.empty:
                col_obj.info(f"No Data for {state_label} in {group_name}")
                continue
                
            subset['Is Own'] = subset['Is Own'].map({True: 'Captured (TW)', False: 'Not Captured'})
            
            fig = px.pie(subset, names='Is Own', values='Opportunity Size', hole=0.4,
                         title=f"<b>{state_label} Capture Rate</b>",
                         color_discrete_sequence=colors)
            fig = style_fig(fig, height=450)
            col_obj.plotly_chart(fig, use_container_width=True)
        st.divider()

def plot_instrument_groups(df):
    grouped_df = df.groupby('Instrument Group')['Opportunity Size'].sum().reset_index()
    grouped_df_is_active = df.groupby(['Instrument Group', 'Is Active'])['Opportunity Size'].sum().reset_index()
    
    pastel_main = ['#FF8A65', '#A5D6A7', '#90CAF9', '#FFE082']
    pastel_map = {
        '0226': ['#FFAB91', '#C5E1A5'], '0326': ['#90CAF9', '#FFF59D'], '0426': ['#CE93D8', '#FFE082']
    }
    
    st.markdown("### 📅 Overall Distribution")
    fig_main = px.pie(grouped_df, names='Instrument Group', values='Opportunity Size', hole=0.4,
                      title="<b>Distribution by Instrument Group</b>", color_discrete_sequence=pastel_main)
    fig_main = style_fig(fig_main, height=550)
    st.plotly_chart(fig_main, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Hidden vs Dimer by Expiry")
    
    instrument_groups = ['0226', '0326', '0426']
    sub_cols = st.columns(3)
    
    for i, instr in enumerate(instrument_groups):
        subset = grouped_df_is_active[grouped_df_is_active['Instrument Group'] == instr].copy()
        if not subset.empty:
            subset['Is Active'] = subset['Is Active'].map({True: 'Hidden', False: 'Dimer'})
            fig = px.pie(subset, names='Is Active', values='Opportunity Size', hole=0.5,
                         title=f"<b>{instr} Cases</b>", color_discrete_sequence=pastel_map[instr])
            fig = style_fig(fig, height=450)
            sub_cols[i].plotly_chart(fig, use_container_width=True)

def plot_market_share(df, plot_date, value_col, colors):
    cases = [
        (df['Underlying Instrument'].str.contains('XU030', na=False), f'XU030'),
        (df['Underlying Instrument'].str.contains('USD', na=False), f'USDTRY'),
        (~df['Underlying Instrument'].str.contains('USD|XU030', na=False), f'SSOs')
    ]

    cols = st.columns(3)
    for i, (filt, title_suffix) in enumerate(cases):
        col_obj = cols[i]
        filtered_df = df[filt]
        if filtered_df.empty:
            col_obj.info(f"No Data for {title_suffix}")
            continue

        val_by_isown = filtered_df.groupby('Is Own')[value_col].sum().reset_index()
        val_by_isown['Is Own'] = val_by_isown['Is Own'].map({True: 'Captured (TW)', False: 'Not Captured'})
        total_val = val_by_isown[value_col].sum()
        
        if total_val != 0:
            fig = px.pie(val_by_isown, names='Is Own', values=value_col, hole=0.4,
                         title=f"<b>{title_suffix}</b>",
                         color_discrete_sequence=colors)
            fig = style_fig(fig, height=450)
            col_obj.plotly_chart(fig, use_container_width=True)
        else:
            col_obj.info(f"Zero Value for {title_suffix}")

def plot_pnl_donuts(df):
    metrics = [
        ('Opportunity Size EoD', 'Opportunity Size_EoD', {
            ('Own', 'Positive'): (df[(df['Is Own']) & (df['Opportunity Size_EoD'] > 0)], ['#FFD54F', '#FFE082', '#FFF9C4', '#FFFDE7']),
            ('Own', 'Negative'): (df[(df['Is Own']) & (df['Opportunity Size_EoD'] < 0)], ['#E57373', '#EF9A9A', '#FFCDD2', '#FFEBEE']),
            ('Not Own', 'Positive'): (df[(~df['Is Own']) & (df['Opportunity Size_EoD'] > 0)], ['#81C784', '#A5D6A7', '#C8E6C9', '#E8F5E9']),
            ('Not Own', 'Negative'): (df[(~df['Is Own']) & (df['Opportunity Size_EoD'] < 0)], ['#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD'])
        }),
        ('EoD Vega PnL', 'EoD_Vega_PnL', {
            ('Own', 'Positive'): (df[(df['Is Own']) & (df['EoD_Vega_PnL'] > 0)], ['#4FC3F7', '#81D4FA', '#B3E5FC', '#E1F5FE']),
            ('Own', 'Negative'): (df[(df['Is Own']) & (df['EoD_Vega_PnL'] < 0)], ['#4DB6AC', '#80CBC4', '#B2DFDB', '#E0F2F1']),
            ('Not Own', 'Positive'): (df[(~df['Is Own']) & (df['EoD_Vega_PnL'] > 0)], ['#F06292', '#F48FB1', '#F8BBD0', '#FCE4EC']),
            ('Not Own', 'Negative'): (df[(~df['Is Own']) & (df['EoD_Vega_PnL'] < 0)], ['#BA68C8', '#CE93D8', '#E1BEE7', '#F3E5F5'])
        }),
        ('EoD Theta PnL', 'EoD_Theta_PnL', {
            ('Own', 'Positive'): (df[(df['Is Own']) & (df['EoD_Theta_PnL'] > 0)], ['#AED581', '#C5E1A5', '#E6EE9C', '#F9FBE7']),
            ('Own', 'Negative'): (df[(df['Is Own']) & (df['EoD_Theta_PnL'] < 0)], ['#7CB342', '#8BC34A', '#9CCC65', '#C5E1A5']),
            ('Not Own', 'Positive'): (df[(~df['Is Own']) & (df['EoD_Theta_PnL'] > 0)], ['#FFD54F', '#FFE082', '#FFF9C4', '#FFFDE7']),
            ('Not Own', 'Negative'): (df[(~df['Is Own']) & (df['EoD_Theta_PnL'] < 0)], ['#FFA726', '#FFB74D', '#FFCC80', '#FFE0B2'])
        })
    ]

    for metric_title, metric_col, subsets in metrics:
        st.markdown(f"## 📉 **{metric_title}**")
        
        items = list(subsets.items())
        
        # Two rows of 2 columns for maximum space
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        grid_cols = [c1, c2, c3, c4]
        
        for idx, ((own_label, sign), (df_sub, base_colors)) in enumerate(items):
            ax = grid_cols[idx]
            
            title = f"<b>{own_label} & {sign}</b>"
            if df_sub.empty:
                ax.info(f"No data for {own_label} & {sign}")
                continue

            sums = df_sub.groupby('Instrument')[metric_col].sum().abs()
            top5 = sums.nlargest(3)
            others = sums[~sums.index.isin(top5.index)].sum()
            
            plot_data = pd.DataFrame({
                'Instrument': list(top5.index) + ['Others'],
                'Value': list(top5.values) + [others]
            })
            total = plot_data['Value'].sum()

            if total == 0:
                ax.info(f"Zero value for {own_label} & {sign}")
                continue

            fig = px.pie(plot_data, names='Instrument', values='Value', hole=0.4,
                         title=f"{title}",
                         color_discrete_sequence=base_colors)
            fig = style_fig(fig, height=450)
            fig.update_layout(showlegend=False) 
            
            # Since legend is hidden, make sure labels are outside
            fig.update_traces(textposition='outside')
            ax.plotly_chart(fig, use_container_width=True)
            
        st.divider()

def plot_delta_bins(df):
    cases = [
        ("USDTRY", "USDTRY", 10),
        ("XU030",  "XU030",  100),
        ("SSO",    None,     100),
    ]

    tabs = st.tabs([f"📈 {c[0]}" for c in cases])

    for tab, (title, pat, div) in zip(tabs, cases):
        with tab:
            df_sub = df.copy()
            if pat:
                df_sub = df_sub[df_sub['Instrument'].str.contains(pat, na=False)]
            else:
                df_sub = df_sub[~df_sub['Instrument'].str.contains("USDTRY|XU030", na=False)]

            df_sub['Delta_abs'] = df_sub['Delta'].abs()
            bins   = np.arange(0, 1.1, 0.2)
            labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
            df_sub['Delta_range'] = pd.cut(df_sub['Delta_abs'], bins=bins, labels=labels, right=False)

            def custom_agg(g):
                own_q = g.loc[g['Is Own'], 'Quantity'].sum()
                mkt_q = g['Quantity'].sum()
                
                non_own = g[~g['Is Own']]
                non_own_q = non_own['Quantity'].sum()
                
                avg_edge = (non_own['Quantity'] * non_own['Edge']).sum() / non_own_q if non_own_q > 0 else np.nan
                avg_req = (non_own['Quantity'] * non_own['Credit']).sum() / non_own_q / div if non_own_q > 0 else np.nan
                
                return pd.Series({'Own_Q': own_q, 'Mkt_Q': mkt_q, 'Avg_Edge_Traded': avg_edge, 'Avg_Edge_Required': avg_req})

            summary = df_sub.groupby('Delta_range', observed=False).apply(custom_agg).reset_index()
            summary['Mkt_%']   = (summary['Own_Q'] / summary['Mkt_Q'] * 100).round(2)
            summary['Other_Q'] = summary['Mkt_Q'] - summary['Own_Q']

            # INTERACTIVE STACKED BAR CHART
            fig = go.Figure()
            fig.add_trace(go.Bar(x=summary['Delta_range'], y=summary['Own_Q'], name='Own Q', 
                                 marker_color='#5c6bc0', text=summary['Own_Q'], textposition='inside',
                                 hovertemplate="<b>Own Q:</b> %{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Bar(x=summary['Delta_range'], y=summary['Other_Q'], name='Other Q', 
                                 marker_color='#b0bec5',
                                 hovertemplate="<b>Other Q:</b> %{y:,.0f}<extra></extra>"))
            
            fig.update_layout(
                barmode='stack',
                title=f"<b>{title} Volume by Δ-Range</b>",
                xaxis_title="<b>Delta Range</b>",
                yaxis_title="<b>Quantity</b>",
                hovermode='x unified'
            )
            
            fig = style_fig(fig, height=600)
            
            # Ensure text on bars is readable
            fig.update_traces(textfont=dict(size=14, color="white", family="Arial, sans-serif"), selector=dict(type='bar'))
            
            st.plotly_chart(fig, use_container_width=True)

            # INTERACTIVE DATA TABLE
            total_own   = summary['Own_Q'].sum()
            total_mkt   = summary['Mkt_Q'].sum()
            non_own_all = df_sub[~df_sub['Is Own']]
            qty_other_all = non_own_all['Quantity'].sum()
            total_other = total_mkt - total_own

            if qty_other_all > 0:
                total_avg_edge = (non_own_all['Quantity'] * non_own_all['Edge']).sum() / qty_other_all
                total_avg_req  = (non_own_all['Quantity'] * non_own_all['Credit']).sum() / qty_other_all / div
            else:
                total_avg_edge, total_avg_req  = np.nan, np.nan

            total_row = pd.DataFrame([{
                'Delta_range': 'All Delta', 'Own_Q': total_own, 'Mkt_Q': total_mkt,
                'Avg_Edge_Traded': total_avg_edge, 'Avg_Edge_Required': total_avg_req,
                'Mkt_%': (total_own / total_mkt * 100).round(2) if total_mkt > 0 else np.nan,
                'Other_Q': total_other
            }])

            summary_table = pd.concat([summary, total_row], ignore_index=True)
            display_table = summary_table[['Delta_range', 'Own_Q', 'Mkt_Q', 'Mkt_%', 'Avg_Edge_Traded', 'Avg_Edge_Required']].copy()
            
            # Professional Table Styling
            def style_dataframe(df):
                return df.style.format({
                    'Own_Q': '{:,.0f}', 'Mkt_Q': '{:,.0f}', 'Mkt_%': '{:.2f}%',
                    'Avg_Edge_Traded': '{:.2f}', 'Avg_Edge_Required': '{:.2f}'
                }).set_properties(**{
                    'background-color': '#ffffff',
                    'color': '#333333',
                    'border-color': '#e9ecef',
                    'font-size': '15px'
                }).apply(lambda x: ['background-color: #f2f6fc; font-weight: bold' if x.name == df.index[-1] else '' for i in x], axis=1)

            st.markdown("### 📋 **Data Breakdown**")
            st.dataframe(style_dataframe(display_table), use_container_width=True, height=300)

# =========================================================
# 4. MAIN APP UI
# =========================================================

st.sidebar.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
st.sidebar.header("Data Source Setup")

input_method = st.sidebar.radio("Select Input Method", ["NAS Path (Local Network)", "Upload File"])

raw_df = None
selected_date = date.today()

if input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload DailyOpportunityAnalysis (.csv/.xlsx)", type=["csv", "xlsx"])
    if uploaded_file:
        raw_df = load_data(uploaded_file)
else:
    selected_date = st.sidebar.date_input("Trade Date", value=date(2026, 2, 24))
    if st.sidebar.button("Load from NAS", type="primary"):
        with st.spinner("Connecting to NAS..."):
            raw_df = load_data(None, trade_date=selected_date)
            if raw_df is None:
                st.sidebar.error("File not found on NAS for selected date.")

if raw_df is not None:
    df = process_data(raw_df)
    
    if df.empty:
        st.warning("Dataframe is empty after loading.")
    else:
        plot_dates = df['Date'].unique()
        display_date = plot_dates[0] if len(plot_dates) > 0 else selected_date

        st.title(f"📊 Daily Opportunity Analysis Dashboard")
        st.caption(f"**Trade Date:** {display_date} | Generated via Streamlit")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1️⃣ Overview",
            "2️⃣ Capture Rates",
            "3️⃣ Instrument Groups",
            "4️⃣ Market Share",
            "5️⃣ PnL Breakdown",
            "6️⃣ Delta Analysis"
        ])
        
        with tab1:
            plot_overview(df, display_date)
            
        with tab2:
            plot_capture_rates_grid(df, display_date)
            
        with tab3:
            plot_instrument_groups(df)
            
        with tab4:
            st.markdown("## 🌍 **Market Share Comparison**")
            st.markdown("### By Quantity")
            plot_market_share(df, display_date, "Quantity", ['#8bc34a', '#f48fb1'])
            st.markdown("---")
            st.markdown("### By Premium Value Traded")
            plot_market_share(df, display_date, "Volume", ['#ffeb3b', '#81d4fa'])
            
        with tab5:
            plot_pnl_donuts(df)
            
        with tab6:
            plot_delta_bins(df)

else:
    # A welcoming empty state
    st.info("👈 **Welcome!** Please select a Trade Date & click 'Load from NAS', or switch to 'Upload File' to begin your analysis.")
