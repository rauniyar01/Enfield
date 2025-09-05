import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def build_energy_emissions_fig():
    data = {
        "Run": ["ensemble_inference", "ebm_training", "ebm_inference"],
        "Energy_kWh": [6.011474282705522e-06, 0.001019365, 8.798740070233181e-07],
        "Emissions_kg": [1.808251464237821e-07, 3.066249093916354e-05, 2.6466610131261406e-08],
    }
    df = pd.DataFrame(data)
    # keep logs stable at tiny values
    df["Energy_kWh"] = df["Energy_kWh"].clip(lower=1e-12)
    df["Emissions_kg"] = df["Emissions_kg"].clip(lower=1e-12)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Energy Consumption per Run", "CO₂ Emissions per Run"),
        horizontal_spacing=0.12
    )
    fig.add_trace(
        go.Bar(
            x=df["Run"], y=df["Energy_kWh"],
            hovertemplate="<b>%{x}</b><br>Energy: %{y:.2e} kWh<extra></extra>",
            name="Energy (kWh)"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Run"], y=df["Emissions_kg"],
            mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>CO₂: %{y:.2e} kg<extra></extra>",
            name="CO₂ (kg)"
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Energy (kWh)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="CO₂ (kg)", type="log", row=1, col=2)
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(
        title="EBM Training & Inference • Energy vs CO₂",
        height=420,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig, df

class EBMSmartMeterPredictor:
    """
    EBM Smart Meter Anomaly Predictor for Streamlit
    """
    def __init__(self):
        self.ebm_model = None
        self.scaler = None
        self.feature_cols = None
    
    def load_model(self, model_path):
        """Load trained EBM model"""
        try:
            model_components = joblib.load(model_path)
            self.ebm_model = model_components['ebm_model']
            self.scaler = model_components['scaler']
            self.feature_cols = model_components['feature_cols']
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def create_features(self, df):
        """Create features matching the training pipeline"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['meter_id', 'timestamp'])
        
        # Basic time features (matching quick training script)
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Statistical features
        df['consumption_zscore'] = df.groupby('meter_id')['hourly_consumption'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Lag features
        df['consumption_lag_1h'] = df.groupby('meter_id')['hourly_consumption'].shift(1)
        df['consumption_lag_24h'] = df.groupby('meter_id')['hourly_consumption'].shift(24)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def predict_anomalies(self, df):
        """Predict anomalies using EBM model"""
        df_features = self.create_features(df)
        
        # Ensure all features exist
        for col in self.feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Prepare features
        X = df_features[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        anomaly_scores = self.ebm_model.predict_proba(X_scaled)[:, 1]
        anomalies = self.ebm_model.predict(X_scaled)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['ebm_anomaly_score'] = anomaly_scores
        result_df['ebm_is_anomaly'] = anomalies
        
        return result_df, df_features
    
    # def get_feature_importance(self):
    #     """Get feature importance for EBM"""
    #     try:
    #         # Use EBM global explanation
    #         global_exp = self.ebm_model.explain_global()
    #         names = global_exp.data()['names']
    #         scores = global_exp.data()['scores']

    #         # Aggregate bin-level scores into a single number per feature
    #         importances = [np.mean(np.abs(np.ravel(s))) for s in scores]

    #         importance_df = pd.DataFrame({
    #             'feature': names,
    #             'importance': importances
    #         }).sort_values('importance', ascending=False).reset_index(drop=True)

    #         return importance_df

    #     except Exception as e:
    #         st.error(f"Error getting feature importance: {str(e)}")
    #         return pd.DataFrame({'feature': self.feature_cols, 'importance': [1.0] * len(self.feature_cols)})
    def _pretty_ebm_term_names(self, raw_names):
        # Build index→name map from the saved training columns
        idx2name = {i: col for i, col in enumerate(self.feature_cols)}

        def pretty_one(term: str) -> str:
            parts = [p.strip() for p in term.split("&")]  # handle interactions
            out = []
            for p in parts:
                if p.startswith("feature_"):
                    try:
                        j = int(p.split("_")[-1])
                        out.append(idx2name.get(j, p))
                    except ValueError:
                        out.append(p)
                else:
                    out.append(p)
            return " & ".join(out)

        return [pretty_one(t) for t in raw_names]

    def get_feature_importance(self):
        """Global importance with human feature names (fallback if model lacks them)."""
        try:
            ge = self.ebm_model.explain_global()
            raw_names = ge.data()['names']    # may be feature_000x ...
            scores = ge.data()['scores']

            # If the model already has readable names, keep them; else map
            if any(str(n).startswith("feature_") for n in raw_names):
                names = self._pretty_ebm_term_names(raw_names)
            else:
                names = raw_names

            importances = [np.mean(np.abs(np.ravel(s))) for s in scores]
            return (pd.DataFrame({"feature": names, "importance": importances})
                    .sort_values("importance", ascending=False).reset_index(drop=True))
        except Exception as e:
            st.error(f"Error getting feature importance: {e}")
            return pd.DataFrame({"feature": [], "importance": []})



# ===============================
# Streamlit App Configuration
# ===============================

st.set_page_config(
    page_title="⚡ ENFIELD - Smart Meter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# App Title and Description
# ===============================

st.markdown('<h1 class="main-header">⚡ ENFIELD - Smart Meter Anomaly Detection with EBM</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        🔍 <strong>Explainable AI</strong> for detecting anomalies in smart meter consumption data
    </p>
    <p style='color: #888;'>
        Powered by Explainable Boosting Machine (EBM) • ENFIELD • SINTEF, Norway
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Initialize Session State
# ===============================

if 'predictor' not in st.session_state:
    st.session_state.predictor = EBMSmartMeterPredictor()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'show_sustainability' not in st.session_state:
    st.session_state.show_sustainability = False    

# ===============================
# Sidebar Configuration
# ===============================

st.sidebar.header("🔧 Model Configuration")
st.sidebar.markdown("---")

# Model loading section
st.sidebar.subheader("📦 Load EBM Model")

# Check if quick model exists
try:
    if st.sidebar.button("🚀 Load EBM Model", type="primary"):
        if st.session_state.predictor.load_model("quick_ebm_model.pkl"):
            st.session_state.model_loaded = True
            st.sidebar.success("✅ EBM model loaded!")
        else:
            st.sidebar.error("❌ Could not load ebm_model.pkl")
except:
    pass

st.sidebar.markdown("**Or upload your own model:**")

uploaded_model = st.sidebar.file_uploader(
    "Upload EBM Model (.pkl)", 
    type=['pkl'],
    help="Upload your trained EBM model file"
)

if uploaded_model is not None:
    try:
        # Save uploaded file temporarily
        with open("temp_model.pkl", "wb") as f:
            f.write(uploaded_model.read())
        
        if st.session_state.predictor.load_model("temp_model.pkl"):
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Custom model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load custom model")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

# Model status
if st.session_state.model_loaded:
    st.sidebar.markdown('<div class="success-box">🎯 <strong>Model Status:</strong> Ready</div>', unsafe_allow_html=True)
    
    # Display model info
    with st.sidebar.expander("📊 Model Information"):
        st.write("**Features used:**")
        if st.session_state.predictor.feature_cols:
            for feature in st.session_state.predictor.feature_cols:
                st.write(f"• {feature}")
else:
    st.sidebar.warning("⚠️ No model loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Quick Start:**")
st.sidebar.markdown("1. Load the Quick EBM Model")
st.sidebar.markdown("2. Upload your meter data CSV")
st.sidebar.markdown("3. Run anomaly detection")
st.sidebar.markdown("4. Explore results & explanations")

# ===============================
# Main Content Area
# ===============================

# Data Upload Section
st.header("📊 Data Upload")

uploaded_file = st.file_uploader(
    "Upload Smart Meter Data (CSV)", 
    type=['csv'],
    help="CSV should contain: meter_id, timestamp, hourly_consumption"
)

# Sample data option
col1, col2 = st.columns([3, 1])
with col1:
    st.info("💡 **Expected format:** CSV with columns `meter_id`, `timestamp`, `hourly_consumption`")
with col2:
    if st.button("📝 Show Sample Format"):
        sample_data = pd.DataFrame({
            'meter_id': ['METER_001', 'METER_001', 'METER_002'],
            'timestamp': ['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 00:00:00'],
            'hourly_consumption': [45.2, 38.7, 52.1]
        })
        st.dataframe(sample_data, use_container_width=True)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['meter_id', 'timestamp', 'hourly_consumption']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            # Data loaded successfully
            st.success("✅ Data loaded successfully!")
            
            # Display data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Records", f"{len(df):,}")
            with col2:
                st.metric("🏠 Unique Meters", f"{df['meter_id'].nunique()}")
            with col3:
                date_range = pd.to_datetime(df['timestamp'])
                days = (date_range.max() - date_range.min()).days + 1
                st.metric("📅 Date Range", f"{days} days")
            with col4:
                avg_consumption = df['hourly_consumption'].mean()
                st.metric("⚡ Avg Consumption", f"{avg_consumption:.1f} kWh")
            
            # Data preview
            with st.expander("🔍 Data Preview (First 10 rows)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality checks
            with st.expander("🔍 Data Quality Overview"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Missing Values")
                    missing_data = df.isnull().sum()
                    if missing_data.sum() == 0:
                        st.success("✅ No missing values found")
                    else:
                        st.dataframe(missing_data[missing_data > 0])
                
                with col2:
                    st.subheader("📊 Consumption Statistics")
                    st.dataframe(df['hourly_consumption'].describe())
            
            # ===============================
            # Anomaly Detection Section
            # ===============================
            
            if st.session_state.model_loaded:
                st.header("🔮 Anomaly Detection")
                
                # Detection settings
                with st.expander("⚙️ Detection Settings", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        anomaly_threshold = st.slider(
                            "Anomaly Score Threshold", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.1,
                            help="Scores above this threshold are classified as anomalies"
                        )
                    with col2:
                        max_records = st.number_input(
                            "Max Records to Process", 
                            min_value=1000, 
                            max_value=len(df), 
                            value=min(50000, len(df)),
                            help="Limit processing for faster results"
                        )
                
                # Run detection
                if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                    with st.spinner("🔍 Detecting anomalies... This may take a moment..."):
                        try:
                            # Process subset if specified
                            df_to_process = df.head(max_records) if max_records < len(df) else df
                            
                            # Run prediction
                            results_df, features_df = st.session_state.predictor.predict_anomalies(df_to_process)
                            
                            # Apply custom threshold
                            results_df['ebm_is_anomaly'] = (results_df['ebm_anomaly_score'] > anomaly_threshold).astype(int)
                            
                            # Store results
                            st.session_state.results_df = results_df
                            st.session_state.features_df = features_df

                            st.session_state.show_sustainability = True
                            
                            # Display summary
                            anomaly_count = results_df['ebm_is_anomaly'].sum()
                            anomaly_rate = results_df['ebm_is_anomaly'].mean() * 100
                            avg_score = results_df['ebm_anomaly_score'].mean()
                            
                            st.success("✅ Anomaly detection completed!")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🚨 Anomalies Found", f"{anomaly_count:,}")
                            with col2:
                                st.metric("📊 Anomaly Rate", f"{anomaly_rate:.2f}%")
                            with col3:
                                st.metric("📈 Avg Score", f"{avg_score:.3f}")
                            with col4:
                                st.metric("✅ Processed", f"{len(results_df):,}")
                            
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
                            st.info("💡 Try reducing the number of records to process")
                
                # ===============================
                # Results Visualization Section
                # ===============================
                
                if st.session_state.results_df is not None:
                    st.header("📊 Results Analysis")
                    
                    results_df = st.session_state.results_df
                    
                    # Meter selection for detailed view
                    selected_meter = st.selectbox(
                        "🏠 Select Meter for Detailed Analysis:",
                        options=results_df['meter_id'].unique(),
                        help="Choose a specific meter to analyze in detail"
                    )
                    
                    # Filter data for selected meter
                    meter_data = results_df[results_df['meter_id'] == selected_meter].copy()
                    meter_data['timestamp'] = pd.to_datetime(meter_data['timestamp'])
                    meter_data = meter_data.sort_values('timestamp')
                    
                    # Time series visualization
                    st.subheader(f"📈 Time Series Analysis - {selected_meter}")
                    
                    # Create interactive plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(
                            f'Hourly Consumption - {selected_meter}',
                            'Anomaly Score Over Time'
                        ),
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Consumption plot
                    fig.add_trace(
                        go.Scatter(
                            x=meter_data['timestamp'],
                            y=meter_data['hourly_consumption'],
                            mode='lines',
                            name='Consumption',
                            line=dict(color='#1f77b4', width=1),
                            hovertemplate='<b>%{x}</b><br>Consumption: %{y:.2f} kWh<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Highlight anomalies
                    anomaly_data = meter_data[meter_data['ebm_is_anomaly'] == 1]
                    if not anomaly_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_data['timestamp'],
                                y=anomaly_data['hourly_consumption'],
                                mode='markers',
                                name='Anomalies',
                                marker=dict(color='red', size=8, symbol='diamond'),
                                hovertemplate='<b>ANOMALY</b><br>%{x}<br>Consumption: %{y:.2f} kWh<br>Score: %{customdata:.3f}<extra></extra>',
                                customdata=anomaly_data['ebm_anomaly_score']
                            ),
                            row=1, col=1
                        )
                    
                    # Anomaly score plot
                    fig.add_trace(
                        go.Scatter(
                            x=meter_data['timestamp'],
                            y=meter_data['ebm_anomaly_score'],
                            mode='lines',
                            name='Anomaly Score',
                            line=dict(color='orange', width=2),
                            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Add threshold line
                    fig.add_hline(
                        y=anomaly_threshold, 
                        line_dash="dash", 
                        line_color="red", 
                        row=2, col=1,
                        annotation_text=f"Threshold ({anomaly_threshold})"
                    )
                    
                    fig.update_layout(
                        height=700,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(title_text="Consumption (kWh)", row=1, col=1)
                    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ===============================
                    # Explainability Section
                    # ===============================
                    
                    st.header("🎯 Model Explainability")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔍 Global Feature Importance")
                        importance_df = st.session_state.predictor.get_feature_importance()
                        
                        # Interactive bar plot
                        fig_importance = px.bar(
                            importance_df.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features",
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Feature importance table
                        st.dataframe(importance_df.head(10), use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Anomaly Pattern Analysis")
                        
                        # Hourly distribution
                        if not anomaly_data.empty:
                            anomaly_data['hour'] = anomaly_data['timestamp'].dt.hour
                            hourly_dist = anomaly_data['hour'].value_counts().sort_index()
                            
                            fig_hourly = px.bar(
                                x=hourly_dist.index,
                                y=hourly_dist.values,
                                title="Anomalies by Hour of Day",
                                labels={'x': 'Hour', 'y': 'Count'}
                            )
                            st.plotly_chart(fig_hourly, use_container_width=True)
                        else:
                            st.info("No anomalies found for selected meter")
                        
                        # Top anomalies table
                        st.subheader("🚨 Top Anomalous Records")
                        top_anomalies = results_df.nlargest(5, 'ebm_anomaly_score')[
                            ['meter_id', 'timestamp', 'hourly_consumption', 'ebm_anomaly_score']
                        ]
                        st.dataframe(top_anomalies, use_container_width=True)
                    
                    # ===============================
                    # Export Section
                    # ===============================
                    
                    st.header("💾 Export Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Full results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Results",
                            data=csv,
                            file_name=f"ebm_anomaly_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Anomalies only
                        anomalies_only = results_df[results_df['ebm_is_anomaly'] == 1]
                        if not anomalies_only.empty:
                            csv_anomalies = anomalies_only.to_csv(index=False)
                            st.download_button(
                                label="🚨 Download Anomalies Only",
                                data=csv_anomalies,
                                file_name=f"anomalies_only_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with col3:
                        # Summary report
                        summary = {
                            'total_records': len(results_df),
                            'anomalies_found': results_df['ebm_is_anomaly'].sum(),
                            'anomaly_rate': f"{results_df['ebm_is_anomaly'].mean() * 100:.2f}%",
                            'avg_anomaly_score': f"{results_df['ebm_anomaly_score'].mean():.3f}",
                            'threshold_used': anomaly_threshold,
                            'unique_meters': results_df['meter_id'].nunique()
                        }
                        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
                        st.download_button(
                            label="📋 Download Summary",
                            data=summary_text,
                            file_name=f"detection_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            
            else:
                st.warning("⚠️ Please load an EBM model first to run anomaly detection")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Please ensure your CSV has the correct format and column names")

else:
    # Landing page content
    st.info("📁 Please upload your smart meter data CSV to begin analysis")
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 **Explainable AI**
        - Global feature importance
        - Transparent decision making
        - No black-box predictions
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ **High Performance**
        - Fast predictions
        - Scalable processing
        """)
    
    with col3:
        st.markdown("""
        ### 🔍 **Smart Detection**
        - Time-aware features
        - Anomaly pattern analysis
        """)

# ===============================
# Sustainability / Footprint
# ===============================
if st.session_state.get("show_sustainability"):
    st.header("🌱 Sustainability Footprint")
    with st.expander("Energy & CO₂ Overview", expanded=True):
        fig_ee, df_ee = build_energy_emissions_fig()
        st.plotly_chart(fig_ee, use_container_width=True)

        st.subheader("Data")
        st.dataframe(
            df_ee.assign(
                Energy_kWh=lambda d: d["Energy_kWh"].map("{:.2e}".format),
                Emissions_kg=lambda d: d["Emissions_kg"].map("{:.2e}".format),
            ),
            use_container_width=True
        )

        csv_ee = df_ee.to_csv(index=False)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download Energy/CO₂ CSV",
                data=csv_ee, file_name="energy_co2_by_run.csv", mime="text/csv",
                use_container_width=True
            )
        with c2:
            png_bytes = None
            try:
                png_bytes = fig_ee.to_image(format="png", scale=2)
            except Exception:
                pass
            st.download_button(
                "🖼️ Download Figure (PNG)",
                data=png_bytes if png_bytes else b"",
                file_name="energy_co2_by_run.png",
                mime="image/png",
                disabled=(png_bytes is None),
                help="Install 'kaleido' to enable PNG export" if png_bytes is None else None,
                use_container_width=True
            )



# ===============================
# Footer
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔬 <strong>EBM Smart Meter Anomaly Detection</strong></p>
        <p>Explainable AI • Trained on 501,579 records • Green AI</p>
        <p>Built with Streamlit, EBM, and domain expertise from XGB/LGB ensemble</p>
    </div>
    """, 
    unsafe_allow_html=True
)