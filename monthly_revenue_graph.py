"""
Monthly Revenue Visualization
=============================
This script generates a graphical representation of monthly revenue from sales data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the sales data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['total_revenue'] = df['quantity'] * df['unit_price']

# Calculate monthly revenue
df['month'] = df['date'].dt.to_period('M')
monthly_revenue = df.groupby('month')['total_revenue'].sum().reset_index()
monthly_revenue['month'] = monthly_revenue['month'].astype(str)

# Create the graph
plt.figure(figsize=(12, 6))

# Bar chart with gradient colors
colors = plt.cm.Blues([(i+3)/8 for i in range(len(monthly_revenue))])
bars = plt.bar(monthly_revenue['month'], monthly_revenue['total_revenue'], 
               color=colors, edgecolor='darkblue', linewidth=1.5)

# Add value labels on top of each bar
for bar, revenue in zip(bars, monthly_revenue['total_revenue']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             f'${revenue:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Customize the chart
plt.xlabel('Month', fontsize=14, fontweight='bold')
plt.ylabel('Revenue ($)', fontsize=14, fontweight='bold')
plt.title('Monthly Revenue - 2024', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=10)

# Add gridlines
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add a subtle background
plt.gca().set_facecolor('#f8f9fa')

# Adjust layout
plt.tight_layout()

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)

# Save the graph
output_path = 'reports/monthly_revenue_graph.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Monthly revenue graph saved to: {output_path}")

# Also display the monthly revenue data
print("\n📊 Monthly Revenue Summary:")
print("=" * 40)
for _, row in monthly_revenue.iterrows():
    print(f"  {row['month']}: ${row['total_revenue']:,.2f}")
print("=" * 40)
print(f"  TOTAL: ${monthly_revenue['total_revenue'].sum():,.2f}")
