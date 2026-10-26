import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

# ==========================================
# PAGE CONFIGURATION & DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Agri-Horticultural Intelligence Engine",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Dashboard Styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 8px;
    }
    
    /* Metric Card Component */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }
    
    /* Tab Navigation Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🌾 Agri-Horticultural Price Forecasting & Intelligence Engine</div>
    <div>
        <span class="header-badge">SIH 1647</span>
        <span class="header-badge">Registration: URK24CS1003, URK24CS1245</span>
        <span style="color: #94a3b8; font-size: 0.9rem;">Integrated Data Science & ML Analytics Suite</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# STRICT REAL-TIME DATASET LOADING ENGINE
# ==========================================
@st.cache_data
def load_agmarknet_data():
    possible_files = [
        r"C:\Users\Athira\Downloads\Agriculture_price_dataset - Agriculture_price_dataset.csv.csv",
        "Agriculture_price_dataset - Agriculture_price_dataset.csv.csv",
        "agmarknet_data.csv",
        "Agmarknet_Data.csv",
        "data.csv",
        "agri_data.csv"
    ]
    csv_path = None
    
    for fname in possible_files:
        if os.path.exists(fname):
            csv_path = fname
            break

    if not csv_path:
        st.error("⚠️ Dataset file not found! Please place your real CSV dataset in the project directory.")
        st.stop()

    try:
        df_raw = pd.read_csv(csv_path)
        
        column_mapping = {
            "state": "STATE", "State Name": "STATE", "state_name": "STATE", "State": "STATE",
            "district": "District Name", "district_name": "District Name", "District": "District Name",
            "market": "Market Name", "Market": "Market Name", "market_name": "Market Name", "Mandi": "Market Name",
            "commodity": "Commodity", "Commodity Name": "Commodity",
            "min_price": "Average_Min_Price", "Min_Price": "Average_Min_Price",
            "max_price": "Average_Max_Price", "Max_Price": "Average_Max_Price",
            "modal_price": "Average_Modal_Price", "Modal_Price": "Average_Modal_Price",
            "price": "Average_Modal_Price", "Price": "Average_Modal_Price"
        }
        df_raw.rename(columns=column_mapping, inplace=True)
        
        if "STATE" in df_raw.columns:
            df_raw["STATE"] = df_raw["STATE"].astype(str).str.strip()
        else:
            df_raw["STATE"] = "Unspecified Region"

        if "District Name" not in df_raw.columns:
            df_raw["District Name"] = "Local District"

        if "Market Name" not in df_raw.columns:
            df_raw["Market Name"] = "Local Market"
            
        if "Commodity" not in df_raw.columns:
            df_raw["Commodity"] = "Agri Produce"
        
        for col in ["Average_Min_Price", "Average_Max_Price", "Average_Modal_Price"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            else:
                df_raw[col] = df_raw.get("Average_Modal_Price", 0)

        df_raw.dropna(subset=["Average_Modal_Price"], inplace=True)
        return df_raw
    except Exception as e:
        st.error(f"Error reading dataset file '{csv_path}': {e}")
        st.stop()

# Load Real Dataset strictly
df = load_agmarknet_data()

# Sidebar Data Status
st.sidebar.markdown("### 📊 Dataset Engine Status")
st.sidebar.success(f"Loaded **{len(df):,}** real-time records strictly from CSV across **{df['Commodity'].nunique()}** commodities and **{df['STATE'].nunique()}** states.")

# ==========================================
# TAB NAVIGATION
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Anti-Hoarding Watchdog",
    "🔄 Crop Substitution Vector",
    "📦 Emergency Buffer Allocation",
    "🎯 Market Risk Clustering",
    "📈 Trend Forecasting Model"
])

# ==========================================
# TAB 1: ANTI-HOARDING PRICE SPIKE DETECTOR
# ==========================================
with tab1:
    st.markdown("### 🚨 Anti-Hoarding Price Spike Detector (Z-Score Outliers)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        z_threshold = st.slider("Z-Score Sensitivity Threshold", min_value=1.5, max_value=3.5, value=2.5, step=0.1)
        st.write(f"🔍 **Statistical Rule:** Flagging market entries meeting sensitivity threshold **{z_threshold}**.")
        st.markdown('</div>', unsafe_allow_html=True)

    mean_p = df["Average_Modal_Price"].mean()
    std_p = df["Average_Modal_Price"].std()
    
    df_calc = df.copy()
    df_calc["Price_Std"] = std_p
    df_calc["Cluster"] = np.where(df_calc["Average_Modal_Price"] > (mean_p + std_p), 1, 0)
    
    if std_p > 0:
        df_calc["Z_Score"] = (df_calc["Average_Modal_Price"] - mean_p) / std_p
    else:
        df_calc["Z_Score"] = 0
        
    anomalies = df_calc[df_calc["Z_Score"] >= z_threshold].sort_values(by="Z_Score", ascending=False)
    
    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("National Baseline Mean", f"₹{mean_p:.2f}")
        m2.metric("Standard Deviation", f"₹{std_p:.2f}")
        m3.metric("Anomalies Flagged", len(anomalies), delta=f"{len(anomalies)/len(df)*100:.1f}% of market", delta_color="inverse")
        
        if len(anomalies) > 0:
            st.error(f"🚨 Detected {len(anomalies)} markets exhibiting extreme price spikes.")
            
            display_cols = [
                "STATE", "District Name", "Market Name", "Commodity", 
                "Average_Min_Price", "Average_Max_Price", "Average_Modal_Price", 
                "Price_Std", "Cluster", "Z_Score"
            ]
            st.dataframe(anomalies[display_cols], use_container_width=True)
            
            csv = anomalies[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Flagged Inspection Audit (CSV)", csv, "hoarding_audit.csv", "text/csv")
        else:
            st.success("✅ Market prices are within expected statistical variance.")

# ==========================================
# TAB 2: CROP SUBSTITUTION VECTOR
# ==========================================
with tab2:
    st.markdown("### 🔄 Vector-Space Crop Substitution Recommender")
    
    rec_matrix = df.groupby("Commodity")["Average_Modal_Price"].agg(["mean", "std"]).reset_index()
    rec_matrix.columns = ["Commodity", "Mean_Price", "Price_Volatility"]
    rec_matrix.fillna(0, inplace=True)
    
    target_crop = st.selectbox("Select Price-Inflated Commodity", rec_matrix["Commodity"].unique(), key="t2_target")
    
    features = rec_matrix[["Mean_Price", "Price_Volatility"]].values
    dist_matrix = euclidean_distances(features, features)
    
    target_idx = rec_matrix[rec_matrix["Commodity"] == target_crop].index[0]
    rec_matrix["Distance_Metric"] = dist_matrix[target_idx]
    
    recommended_df = rec_matrix.sort_values(by="Distance_Metric").reset_index(drop=True)
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Substitution Ranking")
        st.write(f"Target Crop: **{target_crop}**")
        st.write("Crops ranked by closest market substitute profile:")
        st.dataframe(recommended_df[["Commodity", "Mean_Price", "Distance_Metric"]], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        fig_rec = px.scatter(
            rec_matrix, x="Mean_Price", y="Price_Volatility", text="Commodity",
            title="Feature Space Representation (Mean Price vs Volatility)",
            labels={"Mean_Price": "Mean Price (₹)", "Price_Volatility": "Price Standard Deviation (₹)"},
            template="plotly_dark"
        )
        fig_rec.update_traces(marker=dict(size=16, color="#38bdf8"), textposition="top center")
        st.plotly_chart(fig_rec, use_container_width=True)

# ==========================================
# TAB 3: EMERGENCY BUFFER ALLOCATION
# ==========================================
with tab3:
    st.markdown("### 📦 Volatility-Weighted Emergency Buffer Logistics")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_reserve = st.number_input("Central Emergency Reserve (Tonnes)", min_value=1000, max_value=100000, value=6050, step=500)
        st.write("Emergency buffer is dynamically allocated proportional to each state's volatility risk score.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    state_cv = df.groupby("STATE")["Average_Modal_Price"].agg(["mean", "std"]).reset_index()
    state_cv.fillna(0, inplace=True)
    state_cv["CV"] = np.where(state_cv["mean"] > 0, (state_cv["std"] / state_cv["mean"]) * 100, 0)
    
    total_cv_sum = state_cv["CV"].sum()
    if total_cv_sum > 0:
        state_cv["Allocation_Share"] = state_cv["CV"] / total_cv_sum
    else:
        state_cv["Allocation_Share"] = 1 / len(state_cv)
        
    state_cv["Optimized_Buffer_Tonnes"] = state_cv["Allocation_Share"] * total_reserve
    state_cv = state_cv.sort_values(by="CV", ascending=False).reset_index(drop=True)
    
    with c2:
        top_state = state_cv.iloc[0]
        st.warning(f"⚠️ Priority Allocation: **{top_state['STATE']}** receives **{top_state['Optimized_Buffer_Tonnes']:.4f} Tonnes** (Highest Risk Score: **{top_state['CV']:.4f}%**).")
        
    fig_bar = px.bar(
        state_cv, x="STATE", y="Optimized_Buffer_Tonnes", color="CV",
        title=f"Proportional Buffer Reserve Distribution across All {len(state_cv)} States",
        color_continuous_scale="Reds", template="plotly_dark",
        height=450
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.dataframe(
        state_cv[["STATE", "mean", "std", "CV", "Allocation_Share", "Optimized_Buffer_Tonnes"]], 
        use_container_width=True
    )

# ==========================================
# TAB 4: K-MEANS RISK CLUSTERING
# ==========================================
with tab4:
    st.markdown("### 🎯 Unsupervised K-Means Risk Segmentation")
    
    cluster_df = df.groupby("STATE")["Average_Modal_Price"].agg(["mean", "std"]).reset_index()
    cluster_df.columns = ["STATE", "Mean_Price", "Volatility"]
    cluster_df.fillna(0, inplace=True)
    
    n_clusters = min(3, len(cluster_df))
    
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_df["Cluster_ID"] = kmeans.fit_predict(cluster_df[["Mean_Price", "Volatility"]])
        
        ordered_clusters = cluster_df.groupby("Cluster_ID")["Mean_Price"].mean().sort_values(ascending=False).index
        risk_labels = {ordered_clusters[0]: "High Risk Zone"}
        if len(ordered_clusters) > 1:
            risk_labels[ordered_clusters[1]] = "Medium Risk Zone"
        if len(ordered_clusters) > 2:
            risk_labels[ordered_clusters[2]] = "Low Risk Zone"
            
        cluster_df["Risk_Zone"] = cluster_df["Cluster_ID"].map(risk_labels)
    else:
        cluster_df["Risk_Zone"] = "Standard Market Zone"
    
    fig_km = px.scatter(
        cluster_df, x="Mean_Price", y="Volatility", color="Risk_Zone", text="STATE",
        title=f"Market Risk Segmentation across All {len(cluster_df)} States",
        labels={"Mean_Price": "Average Market Price (₹)", "Volatility": "Price Volatility (₹)"},
        template="plotly_dark", height=550,
        color_discrete_map={"High Risk Zone": "#ef4444", "Medium Risk Zone": "#f59e0b", "Low Risk Zone": "#10b981"}
    )
    fig_km.update_traces(marker=dict(size=14), textposition="top center")
    st.plotly_chart(fig_km, use_container_width=True)
    st.dataframe(cluster_df[["STATE", "Mean_Price", "Volatility", "Risk_Zone"]], use_container_width=True)

# ==========================================
# TAB 5: LINEAR REGRESSION FORECAST
# ==========================================
with tab5:
    st.markdown("### 📈 Supervised Linear Regression Trend Forecasting")
    
    selected_crop_fc = st.selectbox("Select Commodity for Forecast", df["Commodity"].unique(), key="t5_crop")
    
    crop_series = df[df["Commodity"] == selected_crop_fc].reset_index(drop=True)
    
    if len(crop_series) > 1:
        X = np.array(range(len(crop_series))).reshape(-1, 1)
        y = crop_series["Average_Modal_Price"].values
        
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        slope = lr_model.coef_[0]
        
        future_X = np.array(range(len(crop_series), len(crop_series) + 10)).reshape(-1, 1)
        future_preds = lr_model.predict(future_X)
        
        fc_table = pd.DataFrame({
            "Future_Step": [f"Step +{i+1}" for i in range(10)],
            "Projected_Price": np.round(future_preds, 2)
        })
        
        f_col1, f_col2 = st.columns([1, 2])
        
        with f_col1:
            st.metric("Predicted Trend Slope", f"{slope:+.4f} ₹/step")
            st.dataframe(fc_table, use_container_width=True)
            
        with f_col2:
            fig_fc = px.line(
                fc_table, x="Future_Step", y="Projected_Price", markers=True,
                title=f"Next 10-Step Price Projection for {selected_crop_fc}",
                template="plotly_dark"
            )
            fig_fc.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8))
            st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.warning("Insufficient time-series entries available in CSV to train the Linear Regression model for this commodity.")