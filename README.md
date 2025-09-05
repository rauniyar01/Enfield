# ENFIELD — Smart Meter Energy Anomaly Detection

A Streamlit app for detecting anomalies in smart meter energy consumption data using an Explainable Boosting Machine (EBM) model.

➡️ **Live app:** https://enfield-anomaly-detection.streamlit.app/

---

## How to Use (Web)

1. **Open the app**  
   Go to **https://enfield-anomaly-detection.streamlit.app/**.

2. **Load the trained model**  
   In the left sidebar, under **Load EBM Model**, click **Browse files** and select your trained model file: **`ebm_model.pkl`**.

3. **Upload your data**  
   In **Data Upload**, drag & drop or browse to select your **CSV** file with smart meter readings.

   **Expected CSV format** (column names):
   - `meter_id`
   - `timestamp` (e.g., ISO datetime)
   - `hourly_consumption` (numeric)

4. **Configure detection**  
   Click **Detection Settings**, then set:
   - **Anomaly Detection Threshold**
   - **Max Records to Process**

5. **Run detection**  
   Click **Run Anomaly Detection**.

6. **Explore results**  
   Review flagged anomalies, charts, and explanations in the app.

---

## Inputs

- **Model file:** `ebm_model.pkl` (Joblib/Pickle EBM model)
- **Data file:** CSV with columns `meter_id`, `timestamp`, `hourly_consumption`

> Tip: Larger files may take longer to upload/process on Streamlit Cloud.

---

## Run Locally (Optional)

```bash
# 1) Clone the repo
git clone https://github.com/rauniyar01/Enfield.git
cd Enfield

# 2) (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Start the app
streamlit run streamlit_app.py

