from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RetailPulse | Retail Intelligence", page_icon="📈", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background:#111827; padding:16px 18px; border-radius:12px; border:1px solid #263244;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {color:#f9fafb;}
    h1, h2, h3 {letter-spacing:-0.02em;}
    .small-note {color:#64748b; font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 RetailPulse")
st.caption("AI-powered customer analytics, demand forecasting and inventory planning")
st.info("Demo-ready dashboard. Upload a transaction CSV to analyze your own data, or start with the built-in synthetic retail dataset. Demo figures are illustrative, not real business results.")

# ---------- Data ----------
REQUIRED = ["InvoiceDate", "CustomerID", "InvoiceNo", "Quantity", "UnitPrice"]

@st.cache_data
def demo_data(n_days=540, n_customers=450, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    customer_ids = np.arange(10001, 10001 + n_customers)
    rows = []
    for day in dates:
        # Weekly pattern + gentle trend + seasonal noise
        n_orders = max(5, int(rng.poisson(25) * (1 + 0.10*np.sin(2*np.pi*day.dayofweek/7))))
        buyers = rng.choice(customer_ids, size=n_orders, replace=True)
        for j, cid in enumerate(buyers):
            qty = int(rng.choice([1, 2, 3, 4, 5, 8], p=[.38,.25,.16,.1,.08,.03]))
            price = float(np.round(rng.lognormal(mean=2.4, sigma=.65), 2))
            invoice = f"{day:%Y%m%d}-{j:03d}"
            rows.append((day, str(cid), invoice, qty, price, rng.choice(["Home", "Beauty", "Grocery", "Electronics", "Fashion"])))
    df = pd.DataFrame(rows, columns=["InvoiceDate","CustomerID","InvoiceNo","Quantity","UnitPrice","Category"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def normalize_data(df):
    df = df.copy()
    # Map common column names from Online Retail datasets and generic retail exports.
    aliases = {
        "InvoiceDate": ["InvoiceDate", "Invoice Date", "invoicedate", "date", "Date", "OrderDate", "order_date"],
        "CustomerID": ["CustomerID", "Customer ID", "customer_id", "Customer Id", "customer"],
        "InvoiceNo": ["InvoiceNo", "Invoice No", "Invoice", "OrderID", "Order Id", "order_id"],
        "Quantity": ["Quantity", "quantity", "Qty", "qty", "Units"],
        "UnitPrice": ["UnitPrice", "Unit Price", "Price", "price", "unit_price"],
        "Category": ["Category", "category", "ProductCategory", "Product Category"],
    }
    rename = {}
    for canonical, candidates in aliases.items():
        for col in df.columns:
            if str(col).strip() in candidates:
                rename[col] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
    for c in ["Quantity", "UnitPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip()
    df = df.dropna(subset=["InvoiceDate","CustomerID","InvoiceNo","Quantity","UnitPrice"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["CustomerID"] != "")]
    df = df.drop_duplicates()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    if "Category" not in df.columns:
        df["Category"] = "Uncategorised"
    df["Category"] = df["Category"].fillna("Uncategorised").astype(str)
    return df.sort_values("InvoiceDate")

def make_rfm(df):
    as_of = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (as_of - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        LastPurchase=("InvoiceDate", "max")
    ).reset_index()
    # Rank-based 1-5 scores: higher recency-days is worse; frequency/monetary higher is better.
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFMScore"] = rfm[["R","F","M"]].sum(axis=1)
    def segment(score):
        if score >= 13: return "Champions"
        if score >= 10: return "Loyal"
        if score >= 8: return "Potential Loyalists"
        if score == 7: return "Need Attention"
        if score >= 5: return "At Risk"
        return "Dormant"
    rfm["Segment"] = rfm["RFMScore"].apply(segment)
    return rfm

def daily_sales(df):
    daily = df.groupby(df["InvoiceDate"].dt.floor("D")).agg(
        Revenue=("Revenue","sum"), Orders=("InvoiceNo","nunique"), Units=("Quantity","sum")
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D", fill_value=0).rename_axis("Date").reset_index()
    return daily

def forecast_daily(daily, horizon=30):
    """Lightweight sklearn forecast using calendar + lag features; avoids heavy optional dependencies."""
    series = daily.set_index("Date")["Revenue"].asfreq("D", fill_value=0).astype(float)
    frame = pd.DataFrame({"y": series})
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["dow"] = frame.index.dayofweek
    frame["dayofyear"] = frame.index.dayofyear
    frame["trend"] = np.arange(len(frame))
    train = frame.dropna()
    if len(train) < 35:
        # A safe fallback for very small uploads.
        avg = float(series.tail(min(7, len(series))).mean()) if len(series) else 0
        future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.DataFrame({"Date":future_dates, "Forecast":np.maximum(0, avg)})
    features = [c for c in train.columns if c != "y"]
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[features], train["y"])
    preds = []
    values = list(series.values.astype(float))
    last_date = series.index.max()
    for step in range(1, horizon + 1):
        date = last_date + pd.Timedelta(days=step)
        idx = len(values)
        row = {
            "lag_1": values[-1], "lag_7": values[-7] if len(values)>=7 else values[-1],
            "lag_14": values[-14] if len(values)>=14 else values[-1],
            "lag_28": values[-28] if len(values)>=28 else values[-1],
            "dow": date.dayofweek, "dayofyear": date.dayofyear, "trend": idx
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([row])[features])[0]))
        preds.append(pred)
        values.append(pred)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"Date":future_dates, "Forecast":preds})

def inventory_table(df, lead_time, service_z, annual_holding_cost, order_cost):
    products = df.groupby("Category").agg(
        Units=("Quantity","sum"), Revenue=("Revenue","sum"), Days=("InvoiceDate","nunique")
    ).reset_index()
    products["DailyDemand"] = products["Units"] / max(1, products["Days"].max())
    # Estimate demand standard deviation from category/day demand.
    cat_daily = df.groupby(["Category", df["InvoiceDate"].dt.floor("D")])["Quantity"].sum().rename("Demand").reset_index()
    std = cat_daily.groupby("Category")["Demand"].std().fillna(0)
    products["DailyStd"] = products["Category"].map(std).fillna(0)
    products["SafetyStock"] = (service_z * products["DailyStd"] * np.sqrt(lead_time)).round().astype(int)
    products["ReorderPoint"] = (products["DailyDemand"] * lead_time + products["SafetyStock"]).round().astype(int)
    avg_unit_cost = (df["Revenue"].sum() / max(1, df["Quantity"].sum()))
    holding = max(0.01, annual_holding_cost * avg_unit_cost)
    products["EOQ"] = np.sqrt((2 * products["Units"] / max(1, products["Days"].max()) * 365 * order_cost) / holding).round().astype(int)
    return products.sort_values("Revenue", ascending=False)

# Sidebar: dataset selection
st.sidebar.header("⚙️ Data & filters")
uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        df = normalize_data(raw)
        st.sidebar.success(f"Loaded {len(df):,} clean transaction rows")
    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()
else:
    df = demo_data()
    st.sidebar.caption("Using built-in synthetic demo data.")

min_date, max_date = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("Transaction date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = df[(df["InvoiceDate"].dt.date >= start_date) & (df["InvoiceDate"].dt.date <= end_date)].copy()
else:
    filtered = df.copy()

if filtered.empty:
    st.warning("No transactions in this date range. Choose a wider range.")
    st.stop()

currency = st.sidebar.selectbox("Currency display", ["£", "₹", "$", "€"], index=0)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30)
lead_time = st.sidebar.slider("Supplier lead time (days)", 1, 60, 14)
service_level = st.sidebar.selectbox("Target service level", ["90%", "95%", "98%"], index=1)
z_value = {"90%":1.282, "95%":1.645, "98%":2.054}[service_level]
order_cost = st.sidebar.number_input("Cost per purchase order", min_value=1.0, value=50.0, step=10.0)
holding_rate = st.sidebar.slider("Annual holding cost (% of unit value)", 5, 50, 20) / 100

# ---------- KPIs ----------
total_revenue = filtered["Revenue"].sum()
orders = filtered["InvoiceNo"].nunique()
customers = filtered["CustomerID"].nunique()
avg_order = total_revenue / max(orders, 1)
units = filtered["Quantity"].sum()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Revenue", f"{currency}{total_revenue:,.0f}")
c2.metric("Orders", f"{orders:,}")
c3.metric("Customers", f"{customers:,}")
c4.metric("Avg. order value", f"{currency}{avg_order:,.2f}")
c5.metric("Units sold", f"{units:,.0f}")

sales_tab, customer_tab, forecast_tab, inventory_tab, data_tab = st.tabs([
    "📊 Sales overview", "🧑‍🤝‍🧑 Customer intelligence", "🔮 Demand forecast", "📦 Inventory planner", "🧾 Data explorer"
])

with sales_tab:
    st.subheader("Sales performance")
    daily = daily_sales(filtered)
    left, right = st.columns([1.6,1])
    with left:
        fig = px.line(daily, x="Date", y="Revenue", title="Daily revenue", markers=False)
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), xaxis_title="", yaxis_title=f"Revenue ({currency})")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        category = filtered.groupby("Category", as_index=False).agg(Revenue=("Revenue","sum")).sort_values("Revenue", ascending=False)
        fig2 = px.bar(category, x="Category", y="Revenue", title="Revenue by category", color="Revenue")
        fig2.update_layout(margin=dict(l=10,r=10,t=50,b=10), xaxis_title="", yaxis_title=f"Revenue ({currency})", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    weekday = filtered.assign(Weekday=filtered["InvoiceDate"].dt.day_name()).groupby("Weekday", as_index=False)["Revenue"].sum()
    order_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday["Weekday"] = pd.Categorical(weekday["Weekday"], categories=order_days, ordered=True)
    weekday = weekday.sort_values("Weekday")
    st.plotly_chart(px.bar(weekday, x="Weekday", y="Revenue", title="Revenue by weekday"), use_container_width=True)
    st.download_button("⬇️ Download filtered transactions", filtered.to_csv(index=False).encode("utf-8"), "retailpulse_filtered_transactions.csv", "text/csv")

with customer_tab:
    st.subheader("Customer segmentation (RFM)")
    rfm = make_rfm(filtered)
    a,b,c = st.columns(3)
    a.metric("Active customers in range", f"{len(rfm):,}")
    b.metric("Champions", f"{(rfm['Segment']=='Champions').sum():,}")
    c.metric("At-risk / dormant", f"{rfm['Segment'].isin(['At Risk','Dormant']).sum():,}")
    l,r = st.columns([1,1.2])
    with l:
        seg_counts = rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        st.plotly_chart(px.bar(seg_counts, x="Segment", y="Customers", color="Segment", title="Customers by RFM segment"), use_container_width=True)
    with r:
        st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="Segment", size="RFMScore",
                                   hover_data=["CustomerID","Recency"], title="Customer value vs frequency"), use_container_width=True)
    st.caption("RFM = Recency (days since last purchase), Frequency (distinct orders), Monetary (revenue). Segment rules are practical heuristics.")
    st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download customer RFM", rfm.to_csv(index=False).encode("utf-8"), "retailpulse_customer_rfm.csv", "text/csv")

with forecast_tab:
    st.subheader("Demand forecast")
    daily = daily_sales(filtered)
    forecast = forecast_daily(daily, forecast_days)
    historical = daily[["Date","Revenue"]].rename(columns={"Revenue":"Value"}).copy()
    historical["Type"] = "Historical revenue"
    predicted = forecast.rename(columns={"Forecast":"Value"}).copy()
    predicted["Type"] = "Forecast"
    chart_data = pd.concat([historical, predicted], ignore_index=True)
    fig = px.line(chart_data, x="Date", y="Value", color="Type", title=f"Revenue forecast — next {forecast_days} days")
    fig.update_layout(xaxis_title="", yaxis_title=f"Revenue ({currency})")
    st.plotly_chart(fig, use_container_width=True)
    f1,f2,f3 = st.columns(3)
    f1.metric("Forecast total", f"{currency}{forecast['Forecast'].sum():,.0f}")
    f2.metric("Forecast daily average", f"{currency}{forecast['Forecast'].mean():,.2f}")
    f3.metric("Forecast horizon", f"{forecast_days} days")
    st.warning("This is a baseline recursive Random Forest forecast for demonstration. Validate with a time-based holdout before business use; forecasts are not guaranteed.")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download forecast", forecast.to_csv(index=False).encode("utf-8"), "retailpulse_demand_forecast.csv", "text/csv")

with inventory_tab:
    st.subheader("Inventory planning by category")
    inv = inventory_table(filtered, lead_time, z_value, holding_rate, order_cost)
    i1,i2,i3 = st.columns(3)
    i1.metric("Categories", f"{len(inv):,}")
    i2.metric("Lead time", f"{lead_time} days")
    i3.metric("Service target", service_level)
    st.markdown("**What the fields mean**")
    st.write("- **Safety stock**: buffer based on demand variability and target service level.")
    st.write("- **Reorder point**: average daily demand × lead time + safety stock.")
    st.write("- **EOQ**: approximate economic order quantity using the classical order/holding-cost formula.")
    st.plotly_chart(px.bar(inv, x="Category", y="ReorderPoint", title="Estimated reorder point by category", color="Category"), use_container_width=True)
    st.dataframe(inv, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download inventory plan", inv.to_csv(index=False).encode("utf-8"), "retailpulse_inventory_plan.csv", "text/csv")
    st.caption("Inventory values are category-level estimates because the required dataset does not necessarily include SKU-level stock, supplier costs, or lead-time history.")

with data_tab:
    st.subheader("Cleaned transaction data")
    m1,m2,m3 = st.columns(3)
    m1.metric("Rows after cleaning", f"{len(filtered):,}")
    m2.metric("Date coverage", f"{(filtered['InvoiceDate'].max()-filtered['InvoiceDate'].min()).days+1:,} days")
    m3.metric("Categories", f"{filtered['Category'].nunique():,}")
    st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
    st.caption("Showing up to 1,000 rows for responsiveness. Download the filtered transactions from Sales overview.")

st.divider()
st.caption("RetailPulse starter build • Python + pandas + scikit-learn + Plotly + Streamlit")
