```python
import pandas as pd

# 1. Define the pandas DataFrame representing car sales
# ---
# Data dictionary containing car sales information
data = {
    "CarID": ["C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008"],
    "Brand": ["Toyota", "Honda", "Ford", "Toyota", "BMW", "Honda", "Ford", "BMW"],
    "Model": ["Camry", "Civic", "Focus", "Rav4", "X5", "Accord", "Mustang", "X3"],
    "Year": [2020, 2021, 2019, 2022, 2023, 2020, 2021, 2022],
    "PriceUSD": [25000, 22000, 18000, 30000, 60000, 24000, 45000, 55000],
    "Mileage": [30000, 20000, 45000, 15000, 5000, 35000, 10000, 8000]
}

# Create the DataFrame from the data dictionary
df = pd.DataFrame(data)

# 2. For each "Brand", calculate the average "PriceUSD" and the average "Mileage".
# 3. Create a new DataFrame showing these averages grouped by "Brand".
# ---
# Group the DataFrame by the 'Brand' column.
# Then, use the .agg() method to calculate multiple aggregations (mean)
# for specified columns ('PriceUSD' and 'Mileage').
# Named aggregations (e.g., Average_PriceUSD) are used for clear output column names.
brand_averages_df = df.groupby("Brand").agg(
    Average_PriceUSD=("PriceUSD", "mean"),  # Calculate mean of 'PriceUSD'
    Average_Mileage=("Mileage", "mean")      # Calculate mean of 'Mileage'
)

# The 'Brand' column will automatically become the index of the resulting DataFrame.
# If you prefer 'Brand' to be a regular column, you can add .reset_index() here:
# brand_averages_df = brand_averages_df.reset_index()

# 4. Print the resulting grouped DataFrame.
# ---
print("DataFrame showing average PriceUSD and Mileage per Brand:")
print(brand_averages_df)
```