# Retail Analysis Dashboard

A large-scale retail analytics platform built with **PySpark** and **Streamlit**, analyzing 500K+ transaction records to deliver customer segmentation, product bundling insights, and interactive business dashboards.

## Features

- **Overview Dashboard** – KPIs (revenue, orders, customers, AOV), monthly trends, top products
- **Country Analysis** – Revenue/order breakdown by country with world map visualization
- **Time Analysis** – Purchase patterns by hour, day of week, and month; RFM customer segmentation with 3D visualization
- **Association Analysis** – Market basket rules ranked by lift and confidence; frequent itemset discovery

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Big Data Processing | PySpark 3.4 |
| Web Dashboard | Streamlit |
| Visualization | Plotly |
| Data Manipulation | Pandas, NumPy |
| Notebook EDA | Jupyter Notebook |

**Workflow:** Raw data → PySpark (distributed processing) → Pandas → Plotly (visualization)

---

## Project Structure

```
Retail-Analysis/
├── app.py                    # Main Streamlit dashboard application
├── EDA.ipynb                 # Exploratory Data Analysis notebook
├── requirements.txt          # Python dependencies
├── associate_retail_data/    # Precomputed association rule outputs
└── online_retail_data/       # Processed parquet files
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Java 8+ (required for PySpark)

### Installation

```bash
# Clone the repository
git clone https://github.com/huukhang2423/Retail-Analysis.git
cd Retail-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Analysis Methods

### RFM Customer Segmentation
Segments customers based on:
- **Recency** – How recently they purchased
- **Frequency** – How often they purchase
- **Monetary** – How much they spend

Results are visualized in an interactive 3D scatter plot for intuitive segment exploration.

### Market Basket Analysis
Applies the Apriori algorithm to uncover product associations:
- Displays top 20 rules ranked by **lift** and **confidence**
- Supports product bundling strategy and cross-sell recommendations
- Results exportable as CSV for downstream use

### Time Pattern Analysis
- Heatmap of purchase activity by hour × day of week
- Monthly seasonality trends
- Growth rate calculation using Spark Window functions

---

## Dataset

- **Source:** Online Retail Dataset (UCI Machine Learning Repository)
- **Size:** 500K+ transactions
- **Coverage:** Multi-country e-commerce transactions
- **Period:** 2010–2011

---

## Skills Demonstrated

- Distributed computing with PySpark (DataFrames, Window functions, caching)
- Customer analytics (RFM segmentation, cohort thinking)
- Association rule mining (Market Basket Analysis)
- Interactive dashboard development with Streamlit + Plotly
- ETL pipeline: raw Excel → Parquet → analytical outputs
