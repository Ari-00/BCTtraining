"""
Sales Data Analysis & Revenue Reporting System
Analyzes sales transactions, calculates revenue metrics,
identifies top products/customers, and generates business reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import functools
import time
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECORATORS
# ─────────────────────────────────────────────

def log_execution(func):
    """Decorator: logs function name, execution time, and status."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f"[LOG] ▶  Running  : {func.__name__}()")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[LOG] ✔  Completed: {func.__name__}()  ({elapsed:.4f}s)")
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"[LOG] ✘  Failed   : {func.__name__}()  ({elapsed:.4f}s) — {exc}")
            raise
    return wrapper


def validate_dataframe(required_cols):
    """Decorator factory: validates that a DataFrame argument has required columns."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(df, *args, **kwargs):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{func.__name__}: first argument must be a DataFrame.")
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"{func.__name__}: missing columns {missing}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

@log_execution
def generate_sales_data(n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic sales transaction data."""
    rng = np.random.default_rng(seed)

    products = {
        "Laptop Pro":      (1_200, 2_500),
        "Wireless Mouse":  (25,    80),
        "Mechanical Keyboard": (80, 250),
        "4K Monitor":      (350,   900),
        "USB-C Hub":       (30,    120),
        "Webcam HD":       (60,    200),
        "SSD 1TB":         (80,    180),
        "Headphones Pro":  (150,   500),
        "Desk Lamp LED":   (20,    70),
        "Office Chair":    (200,   800),
    }

    customers = [f"Customer_{str(i).zfill(3)}" for i in range(1, 51)]
    regions   = ["North", "South", "East", "West", "Central"]

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=int(d)) for d in rng.integers(0, 365, n_records)]

    product_names    = rng.choice(list(products.keys()), n_records)
    units_sold       = rng.integers(1, 15, n_records)
    discount_pct     = rng.choice([0, 5, 10, 15, 20], n_records, p=[0.4, 0.2, 0.2, 0.1, 0.1])

    unit_prices = np.array([
        rng.uniform(*products[p]) for p in product_names
    ])

    revenue = np.round(unit_prices * units_sold * (1 - discount_pct / 100), 2)

    df = pd.DataFrame({
        "transaction_id": [f"TXN-{str(i).zfill(5)}" for i in range(1, n_records + 1)],
        "date":           dates,
        "customer":       rng.choice(customers, n_records),
        "product":        product_names,
        "region":         rng.choice(regions, n_records),
        "units_sold":     units_sold,
        "unit_price":     np.round(unit_prices, 2),
        "discount_pct":   discount_pct,
        "revenue":        revenue,
    })

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# CORE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

@log_execution
@validate_dataframe(["revenue", "units_sold", "discount_pct", "unit_price"])
def calculate_overall_metrics(df: pd.DataFrame) -> dict:
    """Calculate high-level KPIs using NumPy for numerical operations."""
    revenues = df["revenue"].to_numpy()

    metrics = {
        "total_transactions":   int(len(df)),
        "total_revenue":        float(np.sum(revenues)),
        "average_revenue":      float(np.mean(revenues)),
        "median_revenue":       float(np.median(revenues)),
        "revenue_std_dev":      float(np.std(revenues)),
        "max_single_sale":      float(np.max(revenues)),
        "min_single_sale":      float(np.min(revenues)),
        "total_units_sold":     int(df["units_sold"].sum()),
        "avg_discount_pct":     float(np.mean(df["discount_pct"].to_numpy())),
        "revenue_25th_pct":     float(np.percentile(revenues, 25)),
        "revenue_75th_pct":     float(np.percentile(revenues, 75)),
        "revenue_growth_index": float(np.sum(np.diff(
            df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().to_numpy()
        ) > 0) / (df["date"].dt.to_period("M").nunique() - 1)),
    }
    return metrics


@log_execution
@validate_dataframe(["product", "revenue", "units_sold"])
def top_products(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify top N products by total revenue and units sold."""
    summary = (
        df.groupby("product")
          .agg(
              total_revenue =("revenue",    "sum"),
              total_units   =("units_sold", "sum"),
              avg_unit_price=("unit_price", "mean"),
              transactions  =("revenue",    "count"),
          )
          .reset_index()
    )
    summary["revenue_share_pct"] = np.round(
        summary["total_revenue"] / summary["total_revenue"].sum() * 100, 2
    )
    summary.sort_values("total_revenue", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary.head(top_n)


@log_execution
@validate_dataframe(["customer", "revenue", "units_sold"])
def top_customers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify top N customers by total spend."""
    summary = (
        df.groupby("customer")
          .agg(
              total_spend  =("revenue",    "sum"),
              total_orders =("revenue",    "count"),
              total_units  =("units_sold", "sum"),
              avg_order_val=("revenue",    "mean"),
          )
          .reset_index()
    )
    summary["customer_rank"] = summary["total_spend"].rank(ascending=False).astype(int)
    summary.sort_values("total_spend", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary.head(top_n)


@log_execution
@validate_dataframe(["date", "revenue", "units_sold"])
def monthly_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute month-by-month revenue trends and growth rates."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("month")
          .agg(
              revenue      =("revenue",    "sum"),
              transactions =("revenue",    "count"),
              units_sold   =("units_sold", "sum"),
              avg_order_val=("revenue",    "mean"),
          )
          .reset_index()
    )

    monthly["revenue_growth_pct"] = monthly["revenue"].pct_change() * 100
    monthly["cumulative_revenue"]  = monthly["revenue"].cumsum()

    # 3-month rolling average using NumPy
    rev_arr = monthly["revenue"].to_numpy()
    rolling_3m = np.array([
        np.mean(rev_arr[max(0, i-2):i+1]) for i in range(len(rev_arr))
    ])
    monthly["rolling_3m_avg"] = np.round(rolling_3m, 2)

    monthly["revenue"]          = monthly["revenue"].round(2)
    monthly["avg_order_val"]    = monthly["avg_order_val"].round(2)
    monthly["revenue_growth_pct"] = monthly["revenue_growth_pct"].round(2)
    monthly["cumulative_revenue"] = monthly["cumulative_revenue"].round(2)
    return monthly


@log_execution
@validate_dataframe(["region", "revenue", "units_sold"])
def regional_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Break down revenue and performance by region."""
    summary = (
        df.groupby("region")
          .agg(
              total_revenue=("revenue",    "sum"),
              transactions =("revenue",    "count"),
              units_sold   =("units_sold", "sum"),
              avg_order_val=("revenue",    "mean"),
          )
          .reset_index()
    )
    summary["revenue_share_pct"] = np.round(
        summary["total_revenue"] / summary["total_revenue"].sum() * 100, 2
    )
    summary.sort_values("total_revenue", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def _section(title: str, width: int = 70) -> str:
    bar = "─" * width
    return f"\n{bar}\n  {title}\n{bar}"


@log_execution
def generate_report(
    metrics:   dict,
    products:  pd.DataFrame,
    customers: pd.DataFrame,
    monthly:   pd.DataFrame,
    regional:  pd.DataFrame,
    output_path: str = "sales_report.txt",
) -> str:
    """Compile all analyses into a formatted business report and export to file."""

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append("        SALES DATA ANALYSIS & REVENUE REPORTING SYSTEM")
    lines.append(f"        Generated: {ts}")
    lines.append("=" * 70)

    # ── Overall KPIs ────────────────────────────────────────────────────
    lines.append(_section("OVERALL PERFORMANCE METRICS"))
    lines.append(f"  Total Transactions     : {metrics['total_transactions']:,}")
    lines.append(f"  Total Revenue          : ${metrics['total_revenue']:>12,.2f}")
    lines.append(f"  Average Revenue/Sale   : ${metrics['average_revenue']:>12,.2f}")
    lines.append(f"  Median Revenue/Sale    : ${metrics['median_revenue']:>12,.2f}")
    lines.append(f"  Revenue Std Deviation  : ${metrics['revenue_std_dev']:>12,.2f}")
    lines.append(f"  Max Single Sale        : ${metrics['max_single_sale']:>12,.2f}")
    lines.append(f"  Min Single Sale        : ${metrics['min_single_sale']:>12,.2f}")
    lines.append(f"  Total Units Sold       : {metrics['total_units_sold']:>13,}")
    lines.append(f"  Average Discount       : {metrics['avg_discount_pct']:>12.1f}%")
    lines.append(f"  25th Pct Revenue       : ${metrics['revenue_25th_pct']:>12,.2f}")
    lines.append(f"  75th Pct Revenue       : ${metrics['revenue_75th_pct']:>12,.2f}")
    lines.append(f"  MoM Growth Rate (pos%) : {metrics['revenue_growth_index']*100:>11.1f}%")

    # ── Top Products ────────────────────────────────────────────────────
    lines.append(_section("TOP PRODUCTS BY REVENUE"))
    header = f"  {'Rank':<5} {'Product':<26} {'Revenue':>12} {'Units':>7} {'Share%':>8} {'Txns':>6}"
    lines.append(header)
    lines.append("  " + "-" * 66)
    for rank, row in products.iterrows():
        lines.append(
            f"  {rank+1:<5} {row['product']:<26} "
            f"${row['total_revenue']:>11,.2f} "
            f"{int(row['total_units']):>7,} "
            f"{row['revenue_share_pct']:>7.1f}% "
            f"{int(row['transactions']):>6,}"
        )

    # ── Top Customers ───────────────────────────────────────────────────
    lines.append(_section("TOP 10 CUSTOMERS BY SPEND"))
    header = f"  {'Rank':<5} {'Customer':<16} {'Total Spend':>12} {'Orders':>7} {'Avg Order':>10}"
    lines.append(header)
    lines.append("  " + "-" * 55)
    for rank, row in customers.iterrows():
        lines.append(
            f"  {rank+1:<5} {row['customer']:<16} "
            f"${row['total_spend']:>11,.2f} "
            f"{int(row['total_orders']):>7,} "
            f"${row['avg_order_val']:>9,.2f}"
        )

    # ── Monthly Trends ──────────────────────────────────────────────────
    lines.append(_section("MONTHLY SALES TREND ANALYSIS"))
    header = f"  {'Month':<9} {'Revenue':>12} {'Growth%':>9} {'3M Avg':>12} {'Cum. Rev':>14} {'Txns':>6}"
    lines.append(header)
    lines.append("  " + "-" * 66)
    for _, row in monthly.iterrows():
        growth = f"{row['revenue_growth_pct']:>+8.1f}%" if pd.notna(row['revenue_growth_pct']) else "     N/A "
        lines.append(
            f"  {str(row['month']):<9} "
            f"${row['revenue']:>11,.2f} "
            f"{growth:>9} "
            f"${row['rolling_3m_avg']:>11,.2f} "
            f"${row['cumulative_revenue']:>13,.2f} "
            f"{int(row['transactions']):>6,}"
        )

    # ── Regional Breakdown ──────────────────────────────────────────────
    lines.append(_section("REGIONAL PERFORMANCE BREAKDOWN"))
    header = f"  {'Region':<10} {'Revenue':>12} {'Share%':>8} {'Transactions':>14} {'Units':>8}"
    lines.append(header)
    lines.append("  " + "-" * 57)
    for _, row in regional.iterrows():
        lines.append(
            f"  {row['region']:<10} "
            f"${row['total_revenue']:>11,.2f} "
            f"{row['revenue_share_pct']:>7.1f}% "
            f"{int(row['transactions']):>14,} "
            f"{int(row['units_sold']):>8,}"
        )

    # ── NumPy Statistical Digest ─────────────────────────────────────────
    lines.append(_section("NUMPY STATISTICAL DIGEST"))
    rev_arr   = monthly["revenue"].to_numpy()
    mom_diffs = np.diff(rev_arr)
    lines.append(f"  Monthly Revenue — Mean   : ${np.mean(rev_arr):>12,.2f}")
    lines.append(f"  Monthly Revenue — Std    : ${np.std(rev_arr):>12,.2f}")
    lines.append(f"  Monthly Revenue — Min    : ${np.min(rev_arr):>12,.2f}")
    lines.append(f"  Monthly Revenue — Max    : ${np.max(rev_arr):>12,.2f}")
    lines.append(f"  Avg Month-on-Month Δ     : ${np.mean(mom_diffs):>+12,.2f}")
    lines.append(f"  Months with Positive Growth : {int(np.sum(mom_diffs > 0))}")
    lines.append(f"  Months with Decline         : {int(np.sum(mom_diffs < 0))}")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70 + "\n")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  SALES ANALYSIS PIPELINE STARTING")
    print("=" * 70 + "\n")

    # 1. Load / generate data
    df = generate_sales_data(n_records=500)
    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Date range    : {df['date'].min().date()} → {df['date'].max().date()}\n")

    # 2. Run analyses
    metrics   = calculate_overall_metrics(df)
    products  = top_products(df, top_n=5)
    customers = top_customers(df, top_n=10)
    monthly   = monthly_trend_analysis(df)
    regional  = regional_analysis(df)

    # 3. Generate report
    report_path = "sales_report.txt"
    report = generate_report(
        metrics, products, customers, monthly, regional,
        output_path=report_path,
    )

    # 4. Print report to console
    print(report)
    print(f"\n  ✔ Report saved → {os.path.abspath(report_path)}\n")


if __name__ == "__main__":
    main()