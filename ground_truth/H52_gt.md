```python
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Create a pandas DataFrame ---
# Define the data as specified in the task
data = {
    'OrderID': [101, 102, 103, 104, 105, 106, 107],
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 'Mouse', 'Laptop'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
    'Quantity': [1, 2, 1, 1, 3, 1, 1],
    'Price_USD': [1200, 25, 75, 300, 50, 25, 1150],
    'CustomerID': ['CUST001', 'CUST002', 'CUST001', 'CUST003', 'CUST002', 'CUST004', 'CUST001']
}

# Create the DataFrame
df = pd.DataFrame(data)

print("--- Original DataFrame ---")
print(df)
print("\n" + "="*40 + "\n")

# --- 2. Calculate total sales revenue ('Quantity' * 'Price_USD') for each 'Product' ---
# First, calculate the revenue for each individual transaction
df['Revenue'] = df['Quantity'] * df['Price_USD']

# Group by 'Product' and sum the 'Revenue' to get total sales per product
product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)

print("--- Total Sales Revenue by Product ---")
print(product_revenue)
print("\n" + "="*40 + "\n")

# --- 3. Generate and display a horizontal bar chart ---
plt.figure(figsize=(10, 6)) # Set figure size for better readability
product_revenue.plot(kind='barh', color='skyblue')

plt.title('Total Revenue by Product', fontsize=16)
plt.xlabel('Total Revenue (USD)', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7) # Add a grid for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

print("\n" + "="*40 + "\n")

# --- 4. Calculate the average order value ---
# Total revenue from all orders is the sum of the 'Revenue' column
total_revenue_all_orders = df['Revenue'].sum()

# Number of unique 'OrderID's
num_unique_orders = df['OrderID'].nunique()

# Calculate average order value
# Handle potential division by zero, though unlikely with this dataset
if num_unique_orders > 0:
    average_order_value = total_revenue_all_orders / num_unique_orders
    print(f"--- Average Order Value ---")
    print(f"Total Revenue from All Orders: ${total_revenue_all_orders:,.2f}")
    print(f"Number of Unique Orders: {num_unique_orders}")
    print(f"Average Order Value: ${average_order_value:,.2f}")
else:
    print("--- Average Order Value ---")
    print("No unique orders found to calculate average order value.")

print("\n" + "="*40 + "\n")

# --- 5. Identify customers who made more than one purchase ---
# Count the occurrences of each CustomerID
customer_purchase_counts = df['CustomerID'].value_counts()

# Filter for customers with more than one purchase
repeat_customers = customer_purchase_counts[customer_purchase_counts > 1]

print("--- Customers with More Than One Purchase ---")
if not repeat_customers.empty:
    print("CustomerID | Number of Purchases")
    print("--------------------------------")
    for customer_id, count in repeat_customers.items():
        print(f"{customer_id:<10} | {count}")
else:
    print("No customers made more than one purchase.")

print("\n" + "="*40 + "\n")
print("--- Analysis Complete ---")
```