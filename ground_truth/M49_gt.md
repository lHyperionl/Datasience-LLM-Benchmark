```python
import pandas as pd

# 1. Create a pandas DataFrame with the specified data
print("--- Creating DataFrame ---")
data = {
    'OrderID': [101, 102, 103, 104, 105, 106, 107],
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 'Mouse', 'Laptop'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
    'Quantity': [1, 2, 1, 1, 3, 1, 1],
    'Price_USD': [1200, 25, 75, 300, 50, 25, 1150],
    'CustomerID': ['CUST001', 'CUST002', 'CUST001', 'CUST003', 'CUST002', 'CUST004', 'CUST001']
}
df = pd.DataFrame(data)
print("DataFrame created successfully:")
print(df)
print("\n" + "="*50 + "\n")

# Calculate the line total for each order item (Quantity * Price_USD)
df['LineTotal'] = df['Quantity'] * df['Price_USD']

# 2. Calculate the total spending for each 'CustomerID'
print("--- Calculating Total Spending per Customer ---")
# Group by 'CustomerID' and sum the 'LineTotal'
customer_total_spending = df.groupby('CustomerID')['LineTotal'].sum()
print("Total spending per Customer:")
print(customer_total_spending)
print("\n" + "="*50 + "\n")

# 3. Identify and print the 'CustomerID' and their total spending for the top 3 customers
# Sort the customer_total_spending Series in descending order and select the top 3
print("--- Identifying Top 3 Customers by Spending ---")
top_3_customers = customer_total_spending.nlargest(3)

print("Top 3 Customers by Total Spending (Descending Order):")
if not top_3_customers.empty:
    for customer_id, total_spending in top_3_customers.items():
        print(f"CustomerID: {customer_id}, Total Spending: ${total_spending:,.2f}")
else:
    print("No customers found or data is insufficient to determine top 3.")
print("\n" + "="*50 + "\n")

# 4. For each 'Category', calculate and print the total quantity of items sold
print("--- Calculating Total Quantity Sold per Category ---")
# Group by 'Category' and sum the 'Quantity'
category_total_quantity = df.groupby('Category')['Quantity'].sum()

print("Total Quantity of Items Sold per Category:")
if not category_total_quantity.empty:
    print(category_total_quantity)
else:
    print("No categories found or data is insufficient.")
print("\n" + "="*50 + "\n")
```