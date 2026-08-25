# 🧬 Pharma HCP Targeting & Uplift Prediction Portal

An end-to-end commercial analytics platform built using synthetic pharmaceutical sales data. The platform implements machine learning segmentations (K-Means Clustering) and predictive modeling (Random Forest Regressor) to prioritize sales visits to Healthcare Professionals (HCPs) and optimize sales-force return on investment (ROI).

---

## 🚀 Features

*   **Priority Targeting Dashboard:** Allocates rep capacity (10% to 100%) and instantly ranks doctors by predicted weekly prescription uplift.
*   **Strategic & ROI Insights:** Visualizes the sales performance improvement of model-directed visits over random targeting campaigns (yielding a 120%+ average lift improvement).
*   **Dynamic HCP Simulator:** A real-time sandbox tool for commercial planners to input clinician attributes (Specialty, Region, Panel Size, Competitor Share) and promotional strategies (rep calls, detailing minutes, sample units) to simulate the expected weekly sales growth.
*   **Fully Mobile-Responsive:** Styled with high-fidelity corporate blue and green card designs that adapt perfectly to tablet and mobile screens.

---

## 📁 Repository Structure

```text
├── visualizations/             # High-resolution PNG charts showing EDA, PCA, and ROI results
├── .gitignore                  # Keeps large CSV files and Jupyter temporary files out of git
├── Pharma_HCP_Targeting.ipynb  # Comprehensive Jupyter Notebook (Data Processing, EDA, Clustering, RF Modeling)
├── app.py                      # Interactive Streamlit Web Portal
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack & Dependencies

*   **Language:** Python 3.8+
*   **Interactive UI:** Streamlit
*   **Machine Learning:** Scikit-Learn (RandomForestRegressor, KMeans, StandardScaler)
*   **Data Processing:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn

---

## ⚙️ Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/your-pharma-repo-name.git
cd your-pharma-repo-name
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 3. Add the dataset
*Place your `synthetic_hcp_pharma.csv` file into the root folder.*

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
---
