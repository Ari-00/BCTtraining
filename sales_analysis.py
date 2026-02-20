"""
Sales Data Analysis & Revenue Reporting System
==============================================
A Python application that processes sales transaction data,
analyzes revenue trends, identifies top-selling products,
and prepares a final sales report.

Technical Stack:
- Pandas for data analysis
- NumPy for numerical analysis
- Custom decorators for logging
- Business-ready report generation
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps

# ============================================================================
# DECORATORS FOR LOGGING
# ============================================================================

def log_function_call(func):
    """Decorator to log function calls with timestamps and details."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{'='*60}")
        print(f"🔵 LOG: Calling function '{func.__name__}'")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args:
            print(f"   Arguments: {args}")
        if kwargs:
            print(f"   Keyword Arguments: {kwargs}")
        print('='*60)
        
        try:
            result = func(*args, **kwargs)
            print(f"✅ LOG: Function '{func.__name__}' completed successfully")
            print('='*60)
            return result
        except Exception as e:
            print(f"❌ LOG: Function '{func.__name__}' failed with error: {str(e)}")
            print('='*60)
            raise
    
    return wrapper


def log_analysis_step(step_name):
    """Decorator factory for logging analysis steps."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'─'*60}")
            print(f"📊 ANALYSIS: {step_name}")
            print(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
            print('─'*60)
            
            result = func(*args, **kwargs)
            
            print(f"   Completed at: {datetime.now().strftime('%H:%M:%S')}")
            print(f"✓ Finished: {step_name}")
            print('─'*60)
            return result
        return wrapper
    return decorator


# ============================================================================
# DATA LOADING MODULE
# ============================================================================

@log_function_call
@log_analysis_step("Loading Sales Transaction Data")
def load_sales_data(file_path):
    """
    Load sales transaction data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing sales data
        
    Returns:
        DataFrame: Loaded sales data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sales data file not found: {file_path}")
    
    # Load data using Pandas
    df = pd.read_csv(file_path)
    
    # Data validation
    required_columns = ['transaction_id', 'date', 'product_name', 'category', 
                       'quantity', 'unit_price', 'customer_name', 'region']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate total revenue for each transaction
    df['total_revenue'] = df['quantity'] * df['unit_price']
    
    print(f"✅ Loaded {len(df)} transactions from {file_path}")
    print(f"   Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Total Revenue: ${df['total_revenue'].sum():,.2f}")
    
    return df


# ============================================================================
# ANALYSIS MODULE
# ============================================================================

@log_function_call
@log_analysis_step("Calculating Revenue Metrics")
def calculate_revenue_metrics(df):
    """
    Calculate total and average sales metrics.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        dict: Dictionary containing revenue metrics
    """
    # Total revenue using Pandas
    total_revenue = df['total_revenue'].sum()
    
    # Average transaction value
    avg_transaction = df['total_revenue'].mean()
    
    # Median transaction value using NumPy
    median_transaction = np.median(df['total_revenue'])
    
    # Standard deviation using NumPy
    std_deviation = np.std(df['total_revenue'])
    
    # Total quantity sold
    total_quantity = df['quantity'].sum()
    
    # Number of transactions
    num_transactions = len(df)
    
    # Maximum and minimum transactions
    max_transaction = df['total_revenue'].max()
    min_transaction = df['total_revenue'].min()
    
    metrics = {
        'total_revenue': total_revenue,
        'avg_transaction': avg_transaction,
        'median_transaction': median_transaction,
        'std_deviation': std_deviation,
        'total_quantity': total_quantity,
        'num_transactions': num_transactions,
        'max_transaction': max_transaction,
        'min_transaction': min_transaction
    }
    
    print(f"\n📈 Revenue Metrics:")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Average Transaction: ${avg_transaction:,.2f}")
    print(f"   Median Transaction: ${median_transaction:,.2f}")
    print(f"   Standard Deviation: ${std_deviation:,.2f}")
    print(f"   Total Quantity Sold: {total_quantity}")
    print(f"   Number of Transactions: {num_transactions}")
    
    return metrics


@log_function_call
@log_analysis_step("Identifying Top Products")
def identify_top_products(df, n=5):
    """
    Identify top-selling products by revenue.
    
    Args:
        df: Sales DataFrame
        n: Number of top products to return
        
    Returns:
        DataFrame: Top products by revenue
    """
    # Group by product and calculate total revenue
    product_revenue = df.groupby('product_name').agg({
        'quantity': 'sum',
        'total_revenue': 'sum'
    }).reset_index()
    
    # Sort by revenue and get top N
    top_products = product_revenue.nlargest(n, 'total_revenue')
    
    # Add rank
    top_products['rank'] = range(1, len(top_products) + 1)
    top_products = top_products[['rank', 'product_name', 'quantity', 'total_revenue']]
    
    print(f"\n🏆 Top {n} Products by Revenue:")
    for idx, row in top_products.iterrows():
        print(f"   {row['rank']}. {row['product_name']}: ${row['total_revenue']:,.2f} ({row['quantity']} units)")
    
    return top_products


@log_function_call
@log_analysis_step("Identifying Top Customers")
def identify_top_customers(df, n=5):
    """
    Identify top customers by spending.
    
    Args:
        df: Sales DataFrame
        n: Number of top customers to return
        
    Returns:
        DataFrame: Top customers by spending
    """
    # Group by customer and calculate total spending
    customer_spending = df.groupby('customer_name').agg({
        'quantity': 'sum',
        'total_revenue': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    
    customer_spending.columns = ['customer_name', 'quantity', 'total_revenue', 'num_transactions']
    
    # Sort by revenue and get top N
    top_customers = customer_spending.nlargest(n, 'total_revenue')
    
    # Add rank
    top_customers['rank'] = range(1, len(top_customers) + 1)
    top_customers = top_customers[['rank', 'customer_name', 'total_revenue', 'num_transactions', 'quantity']]
    
    print(f"\n👤 Top {n} Customers by Spending:")
    for idx, row in top_customers.iterrows():
        print(f"   {row['rank']}. {row['customer_name']}: ${row['total_revenue']:,.2f} ({row['num_transactions']} transactions)")
    
    return top_customers


@log_function_call
@log_analysis_step("Analyzing Monthly Sales Trends")
def analyze_monthly_trends(df):
    """
    Analyze monthly sales trends.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        DataFrame: Monthly sales summary
    """
    # Extract month information
    df['month'] = df['date'].dt.to_period('M')
    df['month_name'] = df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    
    # Group by month
    monthly_sales = df.groupby('month').agg({
        'total_revenue': 'sum',
        'quantity': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    
    monthly_sales.columns = ['month', 'revenue', 'quantity', 'transactions']
    
    # Calculate month-over-month growth using NumPy
    monthly_sales['revenue_change'] = monthly_sales['revenue'].pct_change() * 100
    
    # Calculate moving average
    monthly_sales['moving_avg'] = monthly_sales['revenue'].rolling(window=3, min_periods=1).mean()
    
    print(f"\n📅 Monthly Sales Trends:")
    print("-" * 70)
    print(f"{'Month':<12} {'Revenue':>12} {'Qty':>8} {'Txns':>6} {'Growth':>10} {'MA(3)':>12}")
    print("-" * 70)
    
    for idx, row in monthly_sales.iterrows():
        growth = row['revenue_change']
        growth_str = f"{growth:+.1f}%" if pd.notna(growth) else "N/A"
        print(f"{str(row['month']):<12} ${row['revenue']:>10,.2f} {row['quantity']:>8} {row['transactions']:>6} {growth_str:>10} ${row['moving_avg']:>10,.2f}")
    
    print("-" * 70)
    
    return monthly_sales


@log_function_call
@log_analysis_step("Analyzing Category Performance")
def analyze_category_performance(df):
    """
    Analyze sales performance by category.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        DataFrame: Category performance summary
    """
    # Group by category
    category_sales = df.groupby('category').agg({
        'total_revenue': 'sum',
        'quantity': 'sum',
        'transaction_id': 'count',
        'product_name': 'nunique'
    }).reset_index()
    
    category_sales.columns = ['category', 'revenue', 'quantity', 'transactions', 'unique_products']
    
    # Calculate percentage of total
    total_revenue = category_sales['revenue'].sum()
    category_sales['revenue_pct'] = (category_sales['revenue'] / total_revenue) * 100
    
    # Sort by revenue
    category_sales = category_sales.sort_values('revenue', ascending=False)
    
    print(f"\n📦 Category Performance:")
    for idx, row in category_sales.iterrows():
        print(f"   {row['category']}: ${row['revenue']:,.2f} ({row['revenue_pct']:.1f}%) - {row['unique_products']} products, {row['transactions']} transactions")
    
    return category_sales


@log_function_call
@log_analysis_step("Analyzing Regional Performance")
def analyze_regional_performance(df):
    """
    Analyze sales performance by region.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        DataFrame: Regional performance summary
    """
    # Group by region
    regional_sales = df.groupby('region').agg({
        'total_revenue': 'sum',
        'quantity': 'sum',
        'transaction_id': 'count',
        'customer_name': 'nunique'
    }).reset_index()
    
    regional_sales.columns = ['region', 'revenue', 'quantity', 'transactions', 'unique_customers']
    
    # Calculate percentage of total
    total_revenue = regional_sales['revenue'].sum()
    regional_sales['revenue_pct'] = (regional_sales['revenue'] / total_revenue) * 100
    
    # Sort by revenue
    regional_sales = regional_sales.sort_values('revenue', ascending=False)
    
    print(f"\n🌍 Regional Performance:")
    for idx, row in regional_sales.iterrows():
        print(f"   {row['region']}: ${row['revenue']:,.2f} ({row['revenue_pct']:.1f}%) - {row['unique_customers']} customers, {row['transactions']} transactions")
    
    return regional_sales


# ============================================================================
# NUMERICAL ANALYSIS WITH NUMPY
# ============================================================================

@log_function_call
@log_analysis_step("Performing Advanced Numerical Analysis")
def perform_numerical_analysis(df):
    """
    Perform advanced numerical analysis using NumPy.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        dict: Dictionary containing numerical analysis results
    """
    revenue_values = df['total_revenue'].values
    quantity_values = df['quantity'].values
    
    analysis = {
        'revenue': {
            'mean': np.mean(revenue_values),
            'median': np.median(revenue_values),
            'std': np.std(revenue_values),
            'variance': np.var(revenue_values),
            'min': np.min(revenue_values),
            'max': np.max(revenue_values),
            'range': np.ptp(revenue_values),
            'percentiles': {
                '25th': np.percentile(revenue_values, 25),
                '50th': np.percentile(revenue_values, 50),
                '75th': np.percentile(revenue_values, 75),
                '90th': np.percentile(revenue_values, 90)
            }
        },
        'quantity': {
            'mean': np.mean(quantity_values),
            'median': np.median(quantity_values),
            'std': np.std(quantity_values),
            'min': np.min(quantity_values),
            'max': np.max(quantity_values),
            'total': np.sum(quantity_values)
        }
    }
    
    print(f"\n🔢 Numerical Analysis Results:")
    print(f"   Revenue - Mean: ${analysis['revenue']['mean']:,.2f}, Std Dev: ${analysis['revenue']['std']:,.2f}")
    print(f"   Revenue - Min: ${analysis['revenue']['min']:,.2f}, Max: ${analysis['revenue']['max']:,.2f}")
    print(f"   Revenue - 25th Percentile: ${analysis['revenue']['percentiles']['25th']:,.2f}")
    print(f"   Revenue - 75th Percentile: ${analysis['revenue']['percentiles']['75th']:,.2f}")
    print(f"   Quantity - Mean: {analysis['quantity']['mean']:.2f}, Total: {analysis['quantity']['total']}")
    
    return analysis


# ============================================================================
# REPORTING MODULE
# ============================================================================

@log_function_call
@log_analysis_step("Generating Sales Report")
def generate_sales_report(df, metrics, top_products, top_customers, monthly_trends, 
                          category_perf, regional_perf, numerical_analysis, output_path):
    """
    Generate a comprehensive business sales report.
    
    Args:
        df: Original sales DataFrame
        metrics: Revenue metrics dictionary
        top_products: Top products DataFrame
        top_customers: Top customers DataFrame
        monthly_trends: Monthly trends DataFrame
        category_perf: Category performance DataFrame
        regional_perf: Regional performance DataFrame
        numerical_analysis: Numerical analysis results
        output_path: Path to save the report
        
    Returns:
        str: Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate report content
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SALES DATA ANALYSIS & REVENUE REPORTING SYSTEM")
    report_lines.append("=" * 80)
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Data Period: {df['date'].min().date()} to {df['date'].max().date()}")
    report_lines.append("=" * 80)
    
    # Executive Summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total Revenue: ${metrics['total_revenue']:,.2f}")
    report_lines.append(f"Total Transactions: {metrics['num_transactions']}")
    report_lines.append(f"Average Transaction Value: ${metrics['avg_transaction']:,.2f}")
    report_lines.append(f"Median Transaction Value: ${metrics['median_transaction']:,.2f}")
    report_lines.append(f"Total Quantity Sold: {metrics['total_quantity']}")
    
    # Top Products
    report_lines.append("\n" + "=" * 80)
    report_lines.append("TOP PRODUCTS BY REVENUE")
    report_lines.append("=" * 80)
    for idx, row in top_products.iterrows():
        report_lines.append(f"{row['rank']}. {row['product_name']}: ${row['total_revenue']:,.2f} ({row['quantity']} units)")
    
    # Top Customers
    report_lines.append("\n" + "=" * 80)
    report_lines.append("TOP CUSTOMERS BY SPENDING")
    report_lines.append("=" * 80)
    for idx, row in top_customers.iterrows():
        report_lines.append(f"{row['rank']}. {row['customer_name']}: ${row['total_revenue']:,.2f} ({row['num_transactions']} transactions)")
    
    # Monthly Trends
    report_lines.append("\n" + "=" * 80)
    report_lines.append("MONTHLY SALES TRENDS")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Month':<12} {'Revenue':>15} {'Quantity':>10} {'Transactions':>12} {'Growth %':>10}")
    report_lines.append("-" * 80)
    for idx, row in monthly_trends.iterrows():
        growth = row['revenue_change']
        growth_str = f"{growth:+.1f}%" if pd.notna(growth) else "N/A"
        report_lines.append(f"{str(row['month']):<12} ${row['revenue']:>13,.2f} {row['quantity']:>10} {row['transactions']:>12} {growth_str:>10}")
    
    # Category Performance
    report_lines.append("\n" + "=" * 80)
    report_lines.append("CATEGORY PERFORMANCE")
    report_lines.append("=" * 80)
    for idx, row in category_perf.iterrows():
        report_lines.append(f"{row['category']}: ${row['revenue']:,.2f} ({row['revenue_pct']:.1f}%)")
    
    # Regional Performance
    report_lines.append("\n" + "=" * 80)
    report_lines.append("REGIONAL PERFORMANCE")
    report_lines.append("=" * 80)
    for idx, row in regional_perf.iterrows():
        report_lines.append(f"{row['region']}: ${row['revenue']:,.2f} ({row['revenue_pct']:.1f}%)")
    
    # Numerical Analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("NUMERICAL ANALYSIS (NumPy)")
    report_lines.append("=" * 80)
    report_lines.append(f"Revenue Statistics:")
    report_lines.append(f"  Mean: ${numerical_analysis['revenue']['mean']:,.2f}")
    report_lines.append(f"  Median: ${numerical_analysis['revenue']['median']:,.2f}")
    report_lines.append(f"  Standard Deviation: ${numerical_analysis['revenue']['std']:,.2f}")
    report_lines.append(f"  Variance: ${numerical_analysis['revenue']['variance']:,.2f}")
    report_lines.append(f"  Range: ${numerical_analysis['revenue']['range']:,.2f}")
    report_lines.append(f"  25th Percentile: ${numerical_analysis['revenue']['percentiles']['25th']:,.2f}")
    report_lines.append(f"  75th Percentile: ${numerical_analysis['revenue']['percentiles']['75th']:,.2f}")
    report_lines.append(f"  90th Percentile: ${numerical_analysis['revenue']['percentiles']['90th']:,.2f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report to file
    report_content = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"\n✅ Sales report generated successfully!")
    print(f"   Report saved to: {output_path}")
    
    return output_path


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

@log_function_call
@log_analysis_step("Creating Monthly Revenue Graph")
def create_monthly_revenue_graph(monthly_trends, output_path):
    """
    Create a bar chart showing monthly revenue.
    
    Args:
        monthly_trends: DataFrame with monthly sales data
        output_path: Path to save the graph image
        
    Returns:
        str: Path to the saved graph
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data
    months = [str(m) for m in monthly_trends['month']]
    revenues = monthly_trends['revenue'].values
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(months, revenues, color='steelblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue ($)', fontsize=12)
    plt.title('Monthly Revenue', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, revenue in enumerate(revenues):
        plt.text(i, revenue + 100, f'${revenue:,.0f}', ha='center', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Monthly revenue graph saved successfully!")
    print(f"   Graph saved to: {output_path}")
    
    return output_path

@log_function_call
@log_analysis_step("Exporting Analysis Data")
def export_analysis_data(df, top_products, top_customers, monthly_trends, output_dir):
    """
    Export analysis results to CSV files.
    
    Args:
        df: Original sales DataFrame
        top_products: Top products DataFrame
        top_customers: Top customers DataFrame
        monthly_trends: Monthly trends DataFrame
        output_dir: Directory to save CSV files
        
    Returns:
        dict: Dictionary with paths to exported files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    # Export cleaned sales data
    sales_data_path = os.path.join(output_dir, 'cleaned_sales_data.csv')
    df.to_csv(sales_data_path, index=False)
    exported_files['sales_data'] = sales_data_path
    
    # Export top products
    products_path = os.path.join(output_dir, 'top_products.csv')
    top_products.to_csv(products_path, index=False)
    exported_files['top_products'] = products_path
    
    # Export top customers
    customers_path = os.path.join(output_dir, 'top_customers.csv')
    top_customers.to_csv(customers_path, index=False)
    exported_files['top_customers'] = customers_path
    
    # Export monthly trends
    trends_path = os.path.join(output_dir, 'monthly_trends.csv')
    monthly_trends.to_csv(trends_path, index=False)
    exported_files['monthly_trends'] = trends_path
    
    print(f"\n✅ Exported analysis data to {output_dir}:")
    for key, path in exported_files.items():
        print(f"   - {key}: {path}")
    
    return exported_files


# ============================================================================
# MAIN EXECUTION
# ============================================================================

@log_function_call
def main():
    """
    Main function to run the complete sales analysis pipeline.
    """
    print("\n" + "=" * 80)
    print("🚀 SALES DATA ANALYSIS & REVENUE REPORTING SYSTEM")
    print("=" * 80)
    
    # Configuration
    DATA_FILE = 'sales_data.csv'
    OUTPUT_DIR = 'reports'
    REPORT_FILE = os.path.join(OUTPUT_DIR, 'sales_report.txt')
    
    # Step 1: Load Sales Data
    print("\n📂 STEP 1: Loading Sales Data...")
    df = load_sales_data(DATA_FILE)
    
    # Step 2: Calculate Revenue Metrics
    print("\n📊 STEP 2: Calculating Revenue Metrics...")
    metrics = calculate_revenue_metrics(df)
    
    # Step 3: Identify Top Products
    print("\n🏆 STEP 3: Identifying Top Products...")
    top_products = identify_top_products(df, n=5)
    
    # Step 4: Identify Top Customers
    print("\n👤 STEP 4: Identifying Top Customers...")
    top_customers = identify_top_customers(df, n=5)
    
    # Step 5: Analyze Monthly Trends
    print("\n📅 STEP 5: Analyzing Monthly Sales Trends...")
    monthly_trends = analyze_monthly_trends(df)
    
    # Step 6: Analyze Category Performance
    print("\n📦 STEP 6: Analyzing Category Performance...")
    category_perf = analyze_category_performance(df)
    
    # Step 7: Analyze Regional Performance
    print("\n🌍 STEP 7: Analyzing Regional Performance...")
    regional_perf = analyze_regional_performance(df)
    
    # Step 8: Perform Numerical Analysis with NumPy
    print("\n🔢 STEP 8: Performing Numerical Analysis with NumPy...")
    numerical_analysis = perform_numerical_analysis(df)
    
    # Step 9: Generate Sales Report
    print("\n📄 STEP 9: Generating Sales Report...")
    report_path = generate_sales_report(
        df, metrics, top_products, top_customers, monthly_trends,
        category_perf, regional_perf, numerical_analysis, REPORT_FILE
    )
    
    # Step 10: Export Analysis Data
    print("\n💾 STEP 10: Exporting Analysis Data...")
    exported_files = export_analysis_data(
        df, top_products, top_customers, monthly_trends, OUTPUT_DIR
    )
    
    # Final Summary
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📊 Total Revenue: ${metrics['total_revenue']:,.2f}")
    print(f"📈 Total Transactions: {metrics['num_transactions']}")
    print(f"📋 Report saved to: {report_path}")
    print("=" * 80)
    
    return {
        'dataframe': df,
        'metrics': metrics,
        'top_products': top_products,
        'top_customers': top_customers,
        'monthly_trends': monthly_trends,
        'category_performance': category_perf,
        'regional_performance': regional_perf,
        'numerical_analysis': numerical_analysis,
        'report_path': report_path,
        'exported_files': exported_files
    }


if __name__ == "__main__":
    results = main()
