import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # sample data representing monthly website traffic (in thousands)
# months=['JAN','FEB','MAR','APR','MAY','JUN']
# traffic=[150,200,180,220,250,210]

# # create a line plot
# plt.plot(months,traffic)
# plt.show()

# # create a line plot with cistom appearence
# plt.plot(months,traffic,marker='o',linestyle='--',color='g')

# plt.xlabel('Month')
# plt.ylabel('Monthly  traffic (in thousands)')
# plt.title('monthly website traffic')

# plt.grid(True)

# # display the plot
# plt.show()
#2 Sample data for two products' monthly revenue (in thousands dolars)
# months=['JAN','FEB','MAR','APR','MAY','JUN']
# product_a_revenue=[45, 55, 60, 70 , 80, 85]
# product_b_revenue=[35, 40,  50, 55, 70, 68]
# # Create a line plot for product A with a blue line and circular markers
# plt.plot(months, product_a_revenue, marker='o', linestyle='-', color='b', label='Product A')
# # Create a line plot for product B with a red line and square markers
# plt.plot(months, product_b_revenue, marker='s', linestyle='--', color='r', label='Product B')
# # Add labels and title
# plt.xlabel('Month')
# plt.ylabel('Monthly Revenue (in $1000)')
# plt.title('Monthly Revenue Comparison of Product A and Product B')
# plt.grid(True)
# # Add a legend to differentiate between the two products
# plt.legend()
# #display the plot
# plt.show()
# #3 Expences catagories 
# catagories=[''
# 'Housing','Transport','Food','Entertainment','Utilities']
# #Monthly expenses for Arindam , Agnik, and Arnab
# arindam_expenses=[1200, 300, 400, 150, 200]
# agnik_expenses=[1000, 250, 350, 100, 150]
# arnab_expenses=[1100, 280, 380, 120, 180]
# # Create an array for the x-axis positions\
# x = np.arange(len(catagories))
# # width of the bars
# width = 0.2
# # Create bar plots for each person's expenses
# plt.bar(x - width, arindam_expenses, width, label='Arindam', color='b')
# plt.bar(x, agnik_expenses, width, label='Agnik', color='g')
# plt.bar(x + width, arnab_expenses, width, label='Arnab', color='r')
# # Add labels and title and legend
# plt.xlabel('Expense Categories')
# plt.ylabel('Monthly Expenses (in $)')
# plt.title('Monthly Expenses Comparison')
# plt.xticks(x, catagories)
# plt.legend()
# # Display the plot
# plt.show()
#4 Sample data expenses
categories=['Housing','Transport','Food','Electric bill','Broadband bill','rental bill']
expenses=[1200, 300, 400, 150, 200, 100]
# Create a pie chart 3D
plt.pie(expenses, labels=categories, autopct='%1.1f%%')
# Add a title
plt.title('Monthly Expenses Distribution')
# Display the plot
plt.show()
########Task create csv file, use monthly data

