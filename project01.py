import pandas as pd
import matplotlib.pyplot as plt
import time

# =======================
# Load Data
# =======================

def load_data(filepath):
    print("Loading data...")
    start = time.time()
    
    df = pd.read_csv(filepath)
    
    end = time.time()
    print("Data loaded in", round(end - start, 4), "seconds\n")
    return df


# =======================
# Calculate Revenue Metrics
# =======================

def calculate_metrics(df):
    print("Calculating revenue...")
    start = time.time()
    
    # Create Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]
    
    # Total Revenue
    total_revenue = df["Revenue"].sum()
    
    # Average Revenue
    avg_revenue = df["Revenue"].mean()
    
    end = time.time()
    print("Revenue calculated in", round(end - start, 4), "seconds\n")
    
    return df, total_revenue, avg_revenue


# =======================
# Top Products
# =======================

def top_products(df):
    print("Finding top products...")
    start = time.time()
    
    result = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    
    end = time.time()
    print("Top products calculated in", round(end - start, 4), "seconds\n")
    
    return result


# =======================
# Monthly Sales Trend
# =======================

def monthly_trend(df):
    print("Calculating monthly trend...")
    start = time.time()
    
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")
    
    result = df.groupby("Month")["Revenue"].sum()
    
    end = time.time()
    print("Monthly trend calculated in", round(end - start, 4), "seconds\n")
    
    return result


# =======================
# Export Report
# =======================

def export_report(df, filepath):
    print("Exporting report...")
    start = time.time()
    
    df.to_csv(filepath, index=False)
    
    end = time.time()
    print("Report exported in", round(end - start, 4), "seconds\n")


# =======================
# Main Program
# =======================

# Load data
df = load_data("sales.csv")

# Calculate metrics
df, total, average = calculate_metrics(df)

print("Total Revenue:", total)
print("Average Revenue:", average)

# Show top products
print("\nTop Products:")
print(top_products(df))

# Monthly trend
trend = monthly_trend(df)

print("\nMonthly Sales Trend:")
print(trend)

# Plot monthly trend
trend.plot(marker='o')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# Export final report
export_report(df, "final_report.csv")
