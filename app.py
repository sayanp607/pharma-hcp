import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(
    page_title="Pharma HCP Targeting & Uplift Portal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Corporate Light UI Theme (Injecting Custom CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Header Area */
    .title-container {
        padding: 24px;
        background: linear-gradient(135deg, #0f4c81 0%, #1e3a8a 100%);
        border-bottom: 3px solid #3b82f6;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    .sub-title {
        font-size: 15px;
        color: #93c5fd;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Cards / Containers */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    .kpi-value-green {
        font-size: 28px;
        font-weight: 700;
        color: #10b981;
    }
    
    /* Prediction Container */
    .predict-box {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        border-right: 1px solid #d1fae5;
        border-top: 1px solid #d1fae5;
        border-bottom: 1px solid #d1fae5;
        padding: 22px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .predict-box-low {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        border-right: 1px solid #fee2e2;
        border-top: 1px solid #fee2e2;
        border-bottom: 1px solid #fee2e2;
        padding: 22px;
        border-radius: 12px;
        margin-top: 20px;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border-color: #2563eb !important;
        border-bottom: 2px solid #ffffff !important;
    }
    
    /* Style form labels */
    label {
        font-weight: 600 !important;
        color: #475569 !important;
        font-size: 13.5px !important;
        margin-bottom: 2px !important;
    }
    
    /* Custom Sidebar styling overrides */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
        padding-top: 10px;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 16px;
        font-weight: 600;
        color: #0f4c81;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
        margin-bottom: 25px;
    }
    
    /* Wrap multiselect and slider widgets in light green cards */
    [data-testid="stSidebar"] .stMultiSelect, 
    [data-testid="stSidebar"] .stSlider {
        background-color: #f0fdf4 !important; /* Light green background */
        border: 1px solid #dcfce7 !important;
        border-left: 4px solid #10b981 !important; /* Green accent border */
        border-radius: 10px !important;
        padding: 16px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* Target labels inside the green cards to use dark forest green */
    [data-testid="stSidebar"] label {
        color: #14532d !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
    }
    
    /* Style input selector containers inside the card */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #bbf7d0 !important;
    }
    
    /* Styling Multiselect chips/tags with high-contrast corporate blue */
    span[data-baseweb="tag"] {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        font-weight: 500 !important;
    }
    span[data-baseweb="tag"] button {
        color: #1e40af !important;
    }
    
    /* Slider color overrides to brand green */
    div[data-testid="stSlider"] [data-disabled="false"] {
        background-color: #10b981 !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #10b981 !important;
        border: 2px solid #ffffff !important;
    }
    
    /* Mobile Responsive Overrides */
    @media (max-width: 768px) {
        .main-title {
            font-size: 22px !important;
            line-height: 1.3 !important;
        }
        .sub-title {
            font-size: 13px !important;
        }
        .title-container {
            padding: 16px !important;
            margin-bottom: 15px !important;
        }
        .kpi-card {
            padding: 14px !important;
            margin-bottom: 12px !important;
        }
        .kpi-value, .kpi-value-green {
            font-size: 22px !important;
        }
        
        /* Force side-by-side columns to stack vertically on mobile screens */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 16px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# 3. Data & Model Caching
@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_hcp_pharma.csv')
    return df

@st.cache_resource
def train_model(df):
    exclude_cols = ['hcp_id', 'week_start_date', 'rx_lift_weekly', 'cluster', 'growth_potential', 'response_flag', 'response_probability']
    feature_list = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    
    X = pd.get_dummies(df[feature_list + ['specialty', 'state']], drop_first=True)
    y = df['rx_lift_weekly']
    
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Pre-predict global lift
    df['predicted_lift'] = rf.predict(X)
    
    # K-Means clustering
    segment_cols = ['baseline_rx_monthly', 'rep_calls', 'panel_size']
    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(df[segment_cols].fillna(0))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_seg_scaled)
    
    cluster_means = df.groupby('cluster')['baseline_rx_monthly'].mean().sort_values(ascending=False)
    tier_mapping = {
        cluster_means.index[0]: 'Tier 1 - High Priority',
        cluster_means.index[1]: 'Tier 2 - Maintain',
        cluster_means.index[2]: 'Tier 3 - Deprioritize'
    }
    df['hcp_tier'] = df['cluster'].map(tier_mapping)
    
    X_means = X.mean().to_dict()
    return rf, df, X.columns, X_means

# Initialize Data & Models
try:
    df_raw = load_data()
    model, df, model_columns, model_column_means = train_model(df_raw)
except Exception as e:
    st.error(f"Failed to load dataset or train models: {e}")
    st.stop()

# 4. Premium Top Banner
st.markdown("""
    <div class="title-container">
        <div class="main-title">🧬 PHARMA HCP TARGETING & UPLIFT PORTAL</div>
        <div class="sub-title">Enterprise Commercial Analytics & ML-Driven Resource Allocation Dashboard</div>
    </div>
""", unsafe_allow_html=True)

# 5. Sidebar Controls
st.sidebar.markdown("""
    <div class="sidebar-header">
        🎛️ Targeting Parameters
    </div>
""", unsafe_allow_html=True)
capacity_slider = st.sidebar.slider(
    "Sales Force Capacity (%)",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Set the percentage of doctors your sales representatives have capacity to visit."
)

specialty_filter = st.sidebar.multiselect(
    "Medical Specialties",
    options=df['specialty'].unique(),
    default=df['specialty'].unique()
)

state_filter = st.sidebar.multiselect(
    "US States",
    options=df['state'].unique(),
    default=df['state'].unique()[:5]
)

# Apply Filters
df_filtered = df[(df['specialty'].isin(specialty_filter)) & (df['state'].isin(state_filter))].copy()

if df_filtered.empty:
    st.warning("No doctors fit the selected criteria.")
    st.stop()

# Sort & Target according to pre-computed lift
df_filtered = df_filtered.sort_values('predicted_lift', ascending=False)
capacity_num = int(len(df_filtered) * (capacity_slider / 100))
df_filtered['Targeted'] = False
df_filtered.iloc[:capacity_num, df_filtered.columns.get_loc('Targeted')] = True

# 6. Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📋 Prioritized Call List", "📊 Strategic & ROI Insights", "🔮 Live HCP Simulator"])

with tab1:
    st.markdown("### 📋 Recommended Physician Targeting List")
    st.markdown("This list is sorted by predicted weekly prescription uplift. Reps should target records marked as **Targeted = True** first.")
    
    # Custom KPIs Row
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Eligible Doctors</div>
                <div class="kpi-value">{len(df_filtered):,}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Allocated Visits (Top {capacity_slider}%)</div>
                <div class="kpi-value">{capacity_num:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        avg_predicted_lift = df_filtered[df_filtered['Targeted']]['predicted_lift'].mean()
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10b981;">
                <div class="kpi-title">Expected Avg Lift per Visit</div>
                <div class="kpi-value-green">+{avg_predicted_lift:.3f} Rx</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Styled Grid
    display_cols = ['hcp_id', 'specialty', 'state', 'baseline_rx_monthly', 'panel_size', 'hcp_tier', 'predicted_lift', 'Targeted']
    st.dataframe(
        df_filtered[display_cols].head(100).style.format({
            'baseline_rx_monthly': '{:.1f}',
            'predicted_lift': '+{:.3f}',
            'panel_size': '{:,}'
        }),
        use_container_width=True
    )

with tab2:
    st.markdown("### 📊 Strategic & ROI Insights")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Custom Corporate Light styling for Matplotlib
    plt.rcParams['figure.facecolor'] = '#ffffff'
    plt.rcParams['axes.facecolor'] = '#f8fafc'
    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#1e293b'
    plt.rcParams['xtick.color'] = '#475569'
    plt.rcParams['ytick.color'] = '#475569'
    plt.rcParams['grid.color'] = '#e2e8f0'
    
    with col_chart1:
        st.markdown("<h4 style='text-align: center; color:#1e293b;'>Campaign Lift Performance Comparison</h4>", unsafe_allow_html=True)
        model_lift = df_filtered[df_filtered['Targeted']]['rx_lift_weekly'].mean()
        random_lift = df_filtered['rx_lift_weekly'].mean()
        
        fig, ax = plt.subplots(figsize=(6, 4.2))
        bars = ax.bar(['Random Campaign', 'Model-Targeted'], [random_lift, model_lift], color=['#94a3b8', '#0f4c81'], width=0.45, edgecolor='#cbd5e1')
        ax.set_ylabel("Average Weekly Rx Lift per Doctor", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.04, f"+{yval:.3f} Rx", ha='center', va='bottom', fontweight='bold', color='#1e293b')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("<h4 style='text-align: center; color:#1e293b;'>Targeting Tier Allocation</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        
        tier_counts = df_filtered['hcp_tier'].value_counts()
        colors = ['#ef4444', '#f59e0b', '#10b981']
        
        # Donut Chart
        wedges, texts, autotexts = ax.pie(
            tier_counts, 
            labels=tier_counts.index, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='#ffffff')
        )
        
        for text in texts:
            text.set_color('#1e293b')
        for autotext in autotexts:
            autotext.set_color('#ffffff')
            autotext.set_fontweight('bold')
            
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.markdown("### 🔮 Interactive HCP Response Simulator")
    st.markdown("Planners can adjust hypothetical traits and promotional tactics below to simulate the model's predicted response.")
    
    # Input grids
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.subheader("HCP Profile")
        hcp_specialty = st.selectbox("Specialty Field", df['specialty'].unique())
        hcp_state = st.selectbox("US State / Region", df['state'].unique())
        baseline_rx = st.slider("Baseline Monthly Volume (Rx)", 0.0, 100.0, 20.0)
        panel_size = st.slider("Patient Panel Size (total)", 100, 3000, 1000)
    
    with col_in2:
        st.subheader("Promotional Tactics")
        rep_calls = st.slider("Planned Rep Calls (visits)", 0, 10, 3)
        detail_minutes = st.slider("planned Detailing Duration (mins)", 0.0, 30.0, 10.0)
        samples = st.slider("Samples Units to Distribute", 0, 20, 5)
        competitor_share = st.slider("Competitor Market Share (%)", 0, 100, 30) / 100
        
    if st.button("Predict Expected Rx Uplift Potential"):
        # Create input dataframe based on average values
        input_data = pd.DataFrame([model_column_means], columns=model_columns)
        
        # Overlay user inputs
        input_data['baseline_rx_monthly'] = baseline_rx
        input_data['panel_size'] = panel_size
        input_data['rep_calls'] = rep_calls
        input_data['rep_detail_minutes'] = detail_minutes
        input_data['samples_units'] = samples
        input_data['competitor_share'] = competitor_share
        
        # Reset specialty & state variables to 0
        for col in model_columns:
            if col.startswith('specialty_') or col.startswith('state_'):
                input_data[col] = 0
        
        # Turn selected specialty & state to 1
        spec_col = f"specialty_{hcp_specialty}"
        if spec_col in input_data.columns:
            input_data[spec_col] = 1
            
        state_col = f"state_{hcp_state}"
        if state_col in input_data.columns:
            input_data[state_col] = 1
            
        prediction = model.predict(input_data)[0]
        
        # Custom HTML rendering box depending on recommendation tier
        if prediction > 0.5:
            st.markdown(f"""
                <div class="predict-box">
                    <h4 style="color:#065f46; margin-top:0px;">🎯 TARGET IMMEDIATELY</h4>
                    <p style="margin-bottom:0px; font-size:16px; color:#065f46;">This doctor is highly responsive to promotional activities.</p>
                    <p style="margin-top:5px; font-size:18px; color:#065f46;">Expected Weekly Prescription Uplift: <strong>+{prediction:.3f} Rx</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="predict-box-low">
                    <h4 style="color:#991b1b; margin-top:0px;">⚠️ DIGITAL COVERAGE ONLY</h4>
                    <p style="margin-bottom:0px; font-size:16px; color:#991b1b;">This doctor has low predicted responsiveness to in-person visits.</p>
                    <p style="margin-top:5px; font-size:18px; color:#991b1b;">Expected Weekly Prescription Uplift: <strong>+{prediction:.3f} Rx</strong></p>
                </div>
            """, unsafe_allow_html=True)
