# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

A runnable starter project with an interactive Streamlit dashboard. It includes:
- Sales overview KPIs and trends
- Customer RFM segmentation (rule-based tiers)
- A baseline demand forecast (scikit-learn Random Forest)
- Category-level inventory planning (EOQ, safety stock, reorder point)
- CSV upload, date filters, currency display and CSV downloads

> **Important:** This starter is designed to run consistently in VS Code, Google Colab, and Streamlit. It uses a lightweight forecasting model instead of requiring Prophet/PyTorch. The dashboard uses synthetic demo data until you upload your own CSV. It does not claim the project brief's reported model metrics; validate any model on a time-based holdout before using it for business decisions.

## 1. Project files

```text
RetailPulse/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_transactions.csv
└── notebooks/
    └── RetailPulse_Colab.ipynb
```

## 2. Run in VS Code (Windows)

1. Install Python 3.10 or newer and VS Code.
2. Extract the ZIP and open the `RetailPulse` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Open the local address printed in the terminal (usually `http://localhost:8501`).

## 3. Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 4. Run in Google Colab

1. Upload `app.py` and `requirements.txt` to the Colab session, or upload the ZIP and extract it.
2. Run this in a notebook cell:

   ```python
   !pip -q install -r requirements.txt
   ```

3. Start Streamlit in the background and expose port 8501 with a tunnel:

   ```python
   !streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
   !npx -y localtunnel --port 8501
   ```

4. Open the public URL printed by LocalTunnel. If it asks for a tunnel password, use the public IP shown by:
   ```python
   !curl -s https://ipv4.icanhazip.com
   ```
   Keep the Colab runtime running while using the dashboard. Public tunnels are temporary; do not upload private customer data to a public tunnel.

An equivalent notebook with these cells is included at `notebooks/RetailPulse_Colab.ipynb`.

## 5. Upload your own data

Upload a CSV with these required columns (common aliases are accepted):
- `InvoiceDate` — transaction date/time
- `CustomerID` — customer identifier
- `InvoiceNo` — order/invoice identifier
- `Quantity` — units sold, must be positive
- `UnitPrice` — price per unit, must be positive
- `Category` — optional product category

The app removes rows with missing required values, invalid dates/numbers, non-positive quantity/price, and exact duplicates. Revenue is calculated as `Quantity × UnitPrice`. For Online Retail II, cancellation invoices and returns should be checked according to your business rules before analysis; this starter removes non-positive quantities but does not infer all possible cancellation conventions.

## 6. Dashboard guide

### Sales overview
Revenue, orders, customer count, average order value, units sold, daily revenue, category revenue, weekday revenue and transaction export.

### Customer intelligence
RFM metrics:
- **Recency:** days since the customer's most recent purchase
- **Frequency:** distinct invoices/orders
- **Monetary:** total revenue

RFM quintile scores are mapped to six practical customer tiers. These are segmentation heuristics, not a trained churn model.

### Demand forecast
A recursive Random Forest model uses lags (1, 7, 14, 28 days), weekday, day-of-year and trend. It predicts revenue for the selected horizon. Recursive forecasts can drift; use a chronological holdout and compare against a naive seasonal baseline before deployment.

### Inventory planner
Estimates category-level daily demand, demand variability, safety stock, reorder point and EOQ. It is not SKU-level stock management. For a production inventory plan, provide SKU-level demand, on-hand stock, supplier lead times, order costs and holding costs.

## 7. Troubleshooting

- **`streamlit` not found:** activate the virtual environment and run `python -m pip install -r requirements.txt`.
- **Upload says required columns missing:** rename your fields to the required names or one of the aliases documented above.
- **No transactions in selected dates:** widen the date range in the sidebar.
- **Colab tunnel fails:** rerun the LocalTunnel cell; free tunnel URLs may change.
- **Large CSV is slow:** pre-aggregate or sample data for exploration; the app caches only the built-in demo dataset.

## 8. Suggested next development steps

1. Add SKU-level inventory and supplier tables.
2. Evaluate forecasts with a chronological holdout (MAE, RMSE, MAPE with zero-safe handling).
3. Add a properly defined churn label and evaluate classification using ROC-AUC, precision, recall and calibration.
4. Save fitted models and add automated tests.
5. Add access control and a private deployment before processing customer-level data.
