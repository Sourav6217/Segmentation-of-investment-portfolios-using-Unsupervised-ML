import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Intelligence Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2230 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card h2 { color: #58a6ff; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #8b949e; font-size: 0.85rem; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.08em; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1f6feb22, transparent);
        border-left: 3px solid #58a6ff;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Cluster badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }

    /* Insight box */
    .insight-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .insight-box strong { color: #58a6ff; }

    /* Tab style override */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0 0;
        color: #8b949e;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb;
        border-color: #1f6feb;
        color: white;
    }

    /* DataFrame */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }

    /* Slider */
    .stSlider > div > div { background: #1f6feb33; }

    /* General text */
    h1, h2, h3 { color: #e6edf3 !important; }
    p, li { color: #c9d1d9; }
    label { color: #8b949e !important; }
    .stSelectbox label, .stSlider label { color: #8b949e !important; }

    /* Top gradient bar */
    .top-bar {
        background: linear-gradient(90deg, #1f6feb, #58a6ff, #79c0ff);
        height: 4px;
        border-radius: 2px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cluster Labels ─────────────────────────────────────────────────────────────
CLUSTER_LABELS = {
    0: ("⚖️ Balanced Core", "#f0883e"),
    1: ("🚀 High Growth", "#3fb950"),
    2: ("🏛️ Mega-Cap Stable", "#58a6ff"),
    3: ("🛡️ Defensive Income", "#d29922"),
    4: ("🔬 Emerging Niche", "#bc8cff"),
}

RISK_CLUSTER_MAP = {
    "Low (Capital Preservation)":    2,
    "Medium (Balanced Growth)":      0,
    "High (Aggressive Growth)":      1,
}

RISK_ADVICE = {
    "Low (Capital Preservation)":  "Focus on capital preservation with high-quality, large-cap stabilizers that offer consistent dividends and proven business models.",
    "Medium (Balanced Growth)":    "Maintain a balanced approach with moderate growth and diversified risk across sectors.",
    "High (Aggressive Growth)":    "Focus on aggressive growth through high-revenue-expansion assets and emerging sector leaders.",
}

# ─── Data Pipeline ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df = pd.read_csv("Hackathon.csv")

    # Missing values
    if df["State"].isnull().sum() > 0:
        df["State"] = df["State"].fillna(df["State"].mode()[0])
    df["Ebitda"]            = df["Ebitda"].fillna(df["Ebitda"].median())
    df["Revenuegrowth"]     = df["Revenuegrowth"].fillna(df["Revenuegrowth"].median())
    df["Fulltimeemployees"] = df["Fulltimeemployees"].fillna(df["Fulltimeemployees"].median())

    # Feature engineering
    df["Log_Marketcap"] = np.log10(df["Marketcap"] + 1)
    df["Log_Ebitda"]    = np.log10(df["Ebitda"].abs() + 1)

    def get_cap_tier(cap):
        if cap >= 200e9: return 4
        elif cap >= 10e9: return 3
        elif cap >= 2e9:  return 2
        else:             return 1

    df["Cap_Tier"] = df["Marketcap"].apply(get_cap_tier)

    risk_map = {
        "Technology": "High_Growth", "Communication Services": "High_Growth",
        "Consumer Cyclical": "Moderate", "Financial Services": "Moderate",
        "Industrials": "Moderate", "Energy": "Moderate", "Basic Materials": "Moderate",
        "Healthcare": "Defensive", "Consumer Defensive": "Defensive",
        "Utilities": "Defensive", "Real Estate": "Defensive",
    }
    df["Risk_Group"] = df["Sector"].map(risk_map).fillna("Other")
    df = pd.get_dummies(df, columns=["Risk_Group"], prefix="Group")

    df["Quality_Score"] = ((df["Ebitda"] > 0) & (df["Revenuegrowth"] > 0)).astype(int)

    final_features = (
        ["Weight", "Revenuegrowth", "Log_Marketcap", "Log_Ebitda", "Cap_Tier", "Quality_Score"]
        + [c for c in df.columns if "Group_" in c]
    )

    X = df[final_features]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    df["Cluster_Label"] = df["Cluster"].map(lambda x: CLUSTER_LABELS[x][0])
    df["Cluster_Color"] = df["Cluster"].map(lambda x: CLUSTER_LABELS[x][1])

    sil = silhouette_score(X_scaled, df["Cluster"])

    # Feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, df["Cluster"])
    fi = pd.DataFrame({"Feature": final_features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)

    # Elbow
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    return df, sil, fi, wcss, final_features

# ─── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading and clustering portfolio data…"):
    df, sil_score, feat_imp, wcss, final_features = load_and_process()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Portfolio Intelligence")
    st.markdown('<p style="color:#8b949e;font-size:0.8rem;">S&P 500 Cluster Analysis · Group 5</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔍 Filters")
    sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector", sectors)

    cap_min, cap_max = float(df["Marketcap"].min()), float(df["Marketcap"].max())
    cap_range = st.slider(
        "Market Cap Range ($B)",
        min_value=0.0,
        max_value=round(cap_max / 1e9, 0),
        value=(0.0, round(cap_max / 1e9, 0)),
        step=10.0,
    )

    growth_range = st.slider(
        "Revenue Growth Range",
        min_value=float(df["Revenuegrowth"].min()),
        max_value=float(df["Revenuegrowth"].max()),
        value=(float(df["Revenuegrowth"].min()), float(df["Revenuegrowth"].max())),
    )

    st.divider()
    st.markdown("### 🎯 Advisor")
    risk_pref = st.selectbox(
        "Your Risk Appetite",
        list(RISK_CLUSTER_MAP.keys()),
    )
    n_recs = st.slider("# Recommendations", 5, 20, 10)

    st.divider()
    st.markdown('<p style="color:#8b949e;font-size:0.75rem;text-align:center;">WeSchool Mumbai · PGDM-RBA · 2024-26<br>Group 5 — Hackathon Project</p>', unsafe_allow_html=True)

# ─── Apply Filters ──────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_sector != "All":
    fdf = fdf[fdf["Sector"] == sel_sector]
fdf = fdf[
    (fdf["Marketcap"] >= cap_range[0] * 1e9) &
    (fdf["Marketcap"] <= cap_range[1] * 1e9) &
    (fdf["Revenuegrowth"] >= growth_range[0]) &
    (fdf["Revenuegrowth"] <= growth_range[1])
]

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)
st.markdown("# 📈 Portfolio Intelligence Suite")
st.markdown('<p style="color:#8b949e;">S&P 500 Cluster-Based Segmentation & Advisory System · KMeans (k=5)</p>', unsafe_allow_html=True)

# ─── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="metric-card"><h2>{len(fdf)}</h2><p>Stocks Visible</p></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><h2>{fdf["Sector"].nunique()}</h2><p>Sectors</p></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><h2>{sil_score:.3f}</h2><p>Silhouette Score</p></div>', unsafe_allow_html=True)
with k4:
    avg_growth = fdf["Revenuegrowth"].mean()
    st.markdown(f'<div class="metric-card"><h2>{avg_growth:.1%}</h2><p>Avg Revenue Growth</p></div>', unsafe_allow_html=True)
with k5:
    total_mc = fdf["Marketcap"].sum() / 1e12
    st.markdown(f'<div class="metric-card"><h2>${total_mc:.1f}T</h2><p>Total Market Cap</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂️ Cluster Overview",
    "📊 EDA & Distributions",
    "🤖 Model Insights",
    "🎯 Stock Advisor",
    "📋 Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Cluster Overview
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Portfolio Segmentation Map</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig_scatter = px.scatter(
            fdf,
            x="Log_Marketcap",
            y="Revenuegrowth",
            color="Cluster_Label",
            size="Weight",
            size_max=30,
            hover_data=["Symbol", "Shortname", "Sector", "Currentprice", "Marketcap"],
            color_discrete_map={v[0]: v[1] for v in CLUSTER_LABELS.values()},
            labels={"Log_Marketcap": "Market Cap (Log₁₀ $)", "Revenuegrowth": "Revenue Growth"},
            title="Size vs. Growth — Portfolio Clusters",
            template="plotly_dark",
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#c9d1d9",
            legend_title="Cluster",
            height=450,
        )
        fig_scatter.update_traces(marker=dict(opacity=0.75, line=dict(width=0.4, color="#30363d")))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
        cluster_summary = fdf.groupby("Cluster").agg(
            Count=("Symbol", "count"),
            Avg_MarketCap_B=("Marketcap", lambda x: x.mean() / 1e9),
            Avg_RevenueGrowth=("Revenuegrowth", "mean"),
            Avg_EBITDA_B=("Ebitda", lambda x: x.mean() / 1e9),
            Avg_Weight=("Weight", "mean"),
        ).reset_index()

        for _, row in cluster_summary.iterrows():
            label, color = CLUSTER_LABELS[int(row["Cluster"])]
            st.markdown(f"""
            <div class="insight-box">
                <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{label}</span><br>
                <strong>Count:</strong> {int(row['Count'])} &nbsp;|&nbsp;
                <strong>Avg MCap:</strong> ${row['Avg_MarketCap_B']:.0f}B<br>
                <strong>Avg Growth:</strong> {row['Avg_RevenueGrowth']:.1%} &nbsp;|&nbsp;
                <strong>Avg Weight:</strong> {row['Avg_Weight']:.4f}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sector Distribution Across Clusters</div>', unsafe_allow_html=True)

    sector_cluster = fdf.groupby(["Sector", "Cluster_Label"]).size().reset_index(name="Count")
    fig_bar = px.bar(
        sector_cluster,
        x="Sector",
        y="Count",
        color="Cluster_Label",
        color_discrete_map={v[0]: v[1] for v in CLUSTER_LABELS.values()},
        template="plotly_dark",
        title="Stocks Per Cluster by Sector",
        barmode="stack",
    )
    fig_bar.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        xaxis_tickangle=-35,
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Cluster donut
    col_c, col_d = st.columns(2)
    with col_c:
        cluster_counts = fdf["Cluster_Label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_donut = px.pie(
            cluster_counts,
            values="Count",
            names="Cluster",
            hole=0.55,
            color="Cluster",
            color_discrete_map={v[0]: v[1] for v in CLUSTER_LABELS.values()},
            title="Cluster Composition",
            template="plotly_dark",
        )
        fig_donut.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_d:
        # Treemap by sector + cluster
        fig_tree = px.treemap(
            fdf,
            path=["Sector", "Cluster_Label"],
            values="Marketcap",
            color="Revenuegrowth",
            color_continuous_scale="RdYlGn",
            title="Market Cap Treemap: Sector → Cluster",
            template="plotly_dark",
        )
        fig_tree.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=350)
        st.plotly_chart(fig_tree, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & Distributions
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)

    dist_col = st.selectbox("Select Feature", ["Currentprice", "Marketcap", "Revenuegrowth", "Weight", "Ebitda", "Fulltimeemployees"])

    col_e, col_f = st.columns(2)
    with col_e:
        fig_hist = px.histogram(
            fdf, x=dist_col, nbins=40, color="Sector",
            title=f"Distribution of {dist_col}", template="plotly_dark",
            marginal="box",
        )
        fig_hist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9", height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_f:
        fig_box = px.box(
            fdf, x="Sector", y=dist_col, color="Sector",
            title=f"{dist_col} by Sector", template="plotly_dark",
        )
        fig_box.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
            xaxis_tickangle=-40, showlegend=False, height=380,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)

    num_cols = ["Currentprice", "Marketcap", "Ebitda", "Revenuegrowth", "Weight", "Fulltimeemployees", "Log_Marketcap", "Log_Ebitda"]
    corr = fdf[num_cols].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Financial Feature Correlation Matrix", template="plotly_dark",
        zmin=-1, zmax=1,
    )
    fig_heat.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">Bivariate Analysis</div>', unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        fig_biv = px.scatter(
            fdf, x="Log_Marketcap", y="Revenuegrowth", color="Sector",
            hover_data=["Symbol", "Shortname"],
            title="Market Cap vs Revenue Growth by Sector", template="plotly_dark",
            opacity=0.7,
        )
        fig_biv.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9", height=380)
        st.plotly_chart(fig_biv, use_container_width=True)

    with col_h:
        top20 = fdf.nlargest(20, "Weight")[["Symbol", "Weight", "Sector", "Cluster_Label"]]
        fig_top = px.bar(
            top20, x="Symbol", y="Weight", color="Sector",
            title="Top 20 Stocks by Portfolio Weight", template="plotly_dark",
        )
        fig_top.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
            xaxis_tickangle=-45, height=380,
        )
        st.plotly_chart(fig_top, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">KMeans Elbow Method</div>', unsafe_allow_html=True)

    col_i, col_j = st.columns(2)
    with col_i:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(1, 11)), y=wcss,
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=8, color="#58a6ff", symbol="circle"),
            name="WCSS",
        ))
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#f0883e", annotation_text="Optimal k=5", annotation_font_color="#f0883e")
        fig_elbow.update_layout(
            title="Elbow Method — Optimal Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#c9d1d9", height=380,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_j:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-box">
            <strong>🏆 Model Performance</strong><br><br>
            <strong>Algorithm:</strong> KMeans++ (k=5)<br>
            <strong>Silhouette Score:</strong> {sil_score:.4f} — Moderate to Good Separation<br>
            <strong>Elbow Point:</strong> WCSS drops sharply to k=5<br><br>
            <strong>Why not DBSCAN?</strong> No noise/outlier structure identified in this dataset — all points belong to identifiable clusters.<br><br>
            <strong>Why not Hierarchical?</strong> No clear dendrogram cut-point; financial segmentation requires predefined cluster count for business interpretability.<br><br>
            <strong>Conclusion:</strong> KMeans with k=5 offers the best balance of interpretability and cluster cohesion for portfolio advisory use.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance (Random Forest Validator)</div>', unsafe_allow_html=True)

    fig_fi = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Magma",
        title="Which Features Best Define Each Cluster?", template="plotly_dark",
    )
    fig_fi.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9",
        height=400, yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Cluster Radar Chart — Financial Profile</div>', unsafe_allow_html=True)

    radar_features = ["Revenuegrowth", "Log_Marketcap", "Log_Ebitda", "Weight", "Cap_Tier", "Quality_Score"]
    cluster_means = df.groupby("Cluster")[radar_features].mean()

    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_radar = go.Figure()
    for cluster_id, (label, color) in CLUSTER_LABELS.items():
        if cluster_id in cluster_means.index:
            vals = cluster_means.loc[cluster_id].tolist()
            vals += [vals[0]]
            cats = radar_features + [radar_features[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=label, line_color=color, fillcolor=hex_to_rgba(color, 0.2),
            ))
    fig_radar.update_layout(
        polar=dict(bgcolor="#161b22", radialaxis=dict(color="#8b949e")),
        template="plotly_dark", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", height=420, title="Cluster Financial Fingerprints",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stock Advisor
# ══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🎯 Personalised Stock Advisory</div>', unsafe_allow_html=True)

    col_k, col_l = st.columns([1, 2])

    with col_k:
        target_cluster = RISK_CLUSTER_MAP[risk_pref]
        label, color = CLUSTER_LABELS[target_cluster]
        advice = RISK_ADVICE[risk_pref]

        st.markdown(f"""
        <div class="insight-box" style="border-color:{color}44;">
            <strong style="color:{color};">Your Risk Profile: {risk_pref}</strong><br><br>
            <span class="badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{label}</span><br><br>
            {advice}
        </div>
        """, unsafe_allow_html=True)

        cluster_stats = df[df["Cluster"] == target_cluster][["Revenuegrowth", "Marketcap", "Ebitda"]].describe()
        st.markdown("**Cluster Statistics**")
        st.dataframe(
            cluster_stats.style.format("{:.2f}"),
            use_container_width=True,
        )

    with col_l:
        recs = df[df["Cluster"] == target_cluster][
            ["Symbol", "Shortname", "Sector", "Currentprice", "Marketcap", "Revenuegrowth", "Weight", "Quality_Score"]
        ].sort_values("Weight", ascending=False).head(n_recs)

        recs_display = recs.copy()
        recs_display["Marketcap"] = recs_display["Marketcap"].apply(lambda x: f"${x/1e9:.1f}B")
        recs_display["Currentprice"] = recs_display["Currentprice"].apply(lambda x: f"${x:.2f}")
        recs_display["Revenuegrowth"] = recs_display["Revenuegrowth"].apply(lambda x: f"{x:.1%}")
        recs_display["Weight"] = recs_display["Weight"].apply(lambda x: f"{x:.4f}")
        recs_display["Quality_Score"] = recs_display["Quality_Score"].apply(lambda x: "✅" if x == 1 else "❌")

        st.markdown(f"**Top {n_recs} Recommended Stocks**")
        st.dataframe(recs_display.reset_index(drop=True), use_container_width=True, height=340)

    st.markdown('<div class="section-header">Recommended Portfolio Composition</div>', unsafe_allow_html=True)

    col_m, col_n = st.columns(2)
    with col_m:
        fig_sector_pie = px.pie(
            recs, names="Sector", values="Weight",
            title="Sector Allocation in Recommendations",
            template="plotly_dark", hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sector_pie.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=350)
        st.plotly_chart(fig_sector_pie, use_container_width=True)

    with col_n:
        fig_rec_bar = px.bar(
            recs.sort_values("Weight", ascending=True),
            x="Weight", y="Symbol", orientation="h", color="Sector",
            title="Portfolio Weight Distribution",
            template="plotly_dark",
        )
        fig_rec_bar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9", height=350,
        )
        st.plotly_chart(fig_rec_bar, use_container_width=True)

    st.markdown('<div class="section-header">Risk vs. Return Map</div>', unsafe_allow_html=True)

    fig_risk = px.scatter(
        df,
        x="Revenuegrowth",
        y="Log_Ebitda",
        color="Cluster_Label",
        size="Weight",
        size_max=25,
        hover_data=["Symbol", "Shortname", "Sector"],
        opacity=0.6,
        color_discrete_map={v[0]: v[1] for v in CLUSTER_LABELS.values()},
        title="Risk vs. Return — All Clusters (Revenue Growth vs EBITDA)",
        template="plotly_dark",
    )
    # Highlight recommended
    rec_symbols = recs["Symbol"].tolist()
    rec_pts = df[df["Symbol"].isin(rec_symbols)]
    fig_risk.add_trace(go.Scatter(
        x=rec_pts["Revenuegrowth"],
        y=rec_pts["Log_Ebitda"],
        mode="markers+text",
        marker=dict(size=14, color=color, symbol="star", line=dict(width=1.5, color="white")),
        text=rec_pts["Symbol"],
        textfont=dict(color="white", size=9),
        textposition="top center",
        name="🌟 Your Picks",
    ))
    fig_risk.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#c9d1d9", height=440,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Full Dataset Explorer</div>', unsafe_allow_html=True)

    search_term = st.text_input("🔍 Search by Symbol or Company Name", "")
    show_cols = st.multiselect(
        "Select Columns",
        ["Symbol", "Shortname", "Sector", "Industry", "Currentprice", "Marketcap", "Ebitda", "Revenuegrowth", "Weight", "Cluster_Label", "Quality_Score", "Cap_Tier"],
        default=["Symbol", "Shortname", "Sector", "Currentprice", "Marketcap", "Revenuegrowth", "Weight", "Cluster_Label"],
    )

    display_df = fdf.copy()
    if search_term:
        display_df = display_df[
            display_df["Symbol"].str.contains(search_term, case=False, na=False) |
            display_df["Shortname"].str.contains(search_term, case=False, na=False)
        ]

    display_df_show = display_df[show_cols].reset_index(drop=True)
    # Format numerics
    if "Marketcap" in display_df_show.columns:
        display_df_show["Marketcap"] = display_df_show["Marketcap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else "—")
    if "Currentprice" in display_df_show.columns:
        display_df_show["Currentprice"] = display_df_show["Currentprice"].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "—")
    if "Revenuegrowth" in display_df_show.columns:
        display_df_show["Revenuegrowth"] = display_df_show["Revenuegrowth"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")

    st.dataframe(display_df_show, use_container_width=True, height=480)
    st.caption(f"Showing {len(display_df_show)} of {len(df)} stocks")

    # Download
    csv_bytes = fdf[show_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        csv_bytes,
        file_name="portfolio_filtered.csv",
        mime="text/csv",
    )

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.78rem;border-top:1px solid #21262d;padding-top:16px;">
    Portfolio Intelligence Suite · WeSchool Mumbai PGDM-RBA 2024–26 · Group 5<br>
    Vaishnavi Dube · Parth Murdeshwar · Palak Sahu · Kaustubh Pawar · Ajinkya Sarwankar · Sourav Manna
</div>
""", unsafe_allow_html=True)
