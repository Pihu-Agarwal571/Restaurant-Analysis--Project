import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TASK 1: Load and inspect the dataset
print("TASK 1: LOAD AND INSPECT THE DATASET")
df = pd.read_csv("22_restaurant_orders.csv")
print(df.head())
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.dtypes)
df.info()

# TASK 2: Missing values and duplicate Order_IDs
print("\nTASK 2: MISSING VALUES AND DUPLICATES")
print(df.isnull().sum())
print("Unique Order_IDs:", df["Order_ID"].nunique())
print("Order_IDs appearing more than once:", (df["Order_ID"].value_counts() > 1).sum())
print("Exact duplicate rows:", df.duplicated().sum())

# TASK 3: Clean missing values in Item, Item_Quantity, Item_Price
print("\nTASK 3: DATA CLEANING")
df = df.drop_duplicates().copy()
df["Item_Quantity"] = pd.to_numeric(df["Item_Quantity"], errors="coerce")
df["Item_Price"] = pd.to_numeric(df["Item_Price"], errors="coerce")
df.loc[df["Item_Quantity"] <= 0, "Item_Quantity"] = np.nan
df.loc[df["Item_Price"] <= 0, "Item_Price"] = np.nan

# Item: filled with "Unknown Item" so the row (and its revenue) is not lost
df["Item"] = df["Item"].fillna("Unknown Item")

# Quantity and Price: filled with the median of the same item, then overall median
df["Item_Quantity"] = df["Item_Quantity"].fillna(df.groupby("Item")["Item_Quantity"].transform("median"))
df["Item_Quantity"] = df["Item_Quantity"].fillna(df["Item_Quantity"].median()).round().astype(int)
df["Item_Price"] = df["Item_Price"].fillna(df.groupby("Item")["Item_Price"].transform("median"))
df["Item_Price"] = df["Item_Price"].fillna(df["Item_Price"].median())

print(df[["Item", "Item_Quantity", "Item_Price"]].isnull().sum())
print("Method: median filling keeps all rows and is not affected by extreme values.")

# TASK 4: Datetime conversion, hour and weekday
print("\nTASK 4: DATETIME CONVERSION")
df["Order_DateTime"] = pd.to_datetime(df["Order_DateTime"], errors="coerce")
df = df.dropna(subset=["Order_DateTime"]).copy()
df["Hour"] = df["Order_DateTime"].dt.hour
df["Weekday"] = df["Order_DateTime"].dt.day_name()
print(df[["Order_DateTime", "Hour", "Weekday"]].head())

# TASK 5: Total orders, items sold, revenue
print("\nTASK 5: TOTALS")
total_orders = df["Order_ID"].nunique()
total_items = df["Item_Quantity"].sum()
total_revenue = (df["Item_Quantity"] * df["Item_Price"]).sum()
print("Order records (rows):", len(df))
print("Unique orders:", total_orders)
print("Total items sold:", total_items)
print("Total revenue:", round(total_revenue, 2))

# TASK 6: Item_Revenue column
print("\nTASK 6: ITEM_REVENUE")
df["Item_Revenue"] = df["Item_Quantity"] * df["Item_Price"]
print(df[["Order_ID", "Item", "Item_Quantity", "Item_Price", "Item_Revenue"]].head())

# TASK 7: Top 10 most ordered dishes
print("\nTASK 7: TOP 10 DISHES")
items = df[df["Item"] != "Unknown Item"].groupby("Item").agg(
    Total_Quantity=("Item_Quantity", "sum"),
    Total_Revenue=("Item_Revenue", "sum")
).sort_values("Total_Quantity", ascending=False)
top10 = items.head(10)
print(top10)

plt.figure(figsize=(9, 5))
plt.barh(top10.index[::-1], top10["Total_Quantity"][::-1], color="steelblue")
plt.title("Top 10 Most Ordered Dishes")
plt.xlabel("Total Quantity Sold")
plt.tight_layout()
plt.show()

# TASK 8: Category-wise revenue
print("\nTASK 8: CATEGORY-WISE REVENUE")
category_revenue = df.groupby("Category")["Item_Revenue"].sum().sort_values(ascending=False)
print(category_revenue.round(2))

plt.figure(figsize=(8, 5))
plt.bar(category_revenue.index, category_revenue.values, color="darkorange")
plt.title("Category-wise Revenue")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# TASK 9: Average bill by order type (bill = sum of Item_Revenue per Order_ID)
print("\nTASK 9: AVERAGE BILL BY ORDER TYPE")
order_df = df.groupby("Order_ID").agg(
    Bill_Value=("Item_Revenue", "sum"),
    Order_Type=("Order_Type", "first"),
    Customer_ID=("Customer_ID", "first"),
    Hour=("Hour", "first"),
    Weekday=("Weekday", "first")
).reset_index()

avg_bill = order_df.groupby("Order_Type")["Bill_Value"].mean().sort_values(ascending=False)
print(avg_bill.round(2))

plt.figure(figsize=(6, 5))
plt.bar(avg_bill.index, avg_bill.values, color="seagreen")
plt.title("Average Bill by Order Type")
plt.xlabel("Order Type")
plt.ylabel("Average Bill")
plt.tight_layout()
plt.show()

# TASK 10: Peak hours and busiest day
print("\nTASK 10: PEAK HOURS AND BUSIEST DAY")
hour_counts = order_df["Hour"].value_counts().sort_index()
peak_hours = hour_counts.sort_values(ascending=False).head(3)
print("Top 3 peak hours:")
print(peak_hours)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_counts = order_df["Weekday"].value_counts().reindex(days, fill_value=0)
busiest_day = weekday_counts.idxmax()
print(weekday_counts)
print("Busiest day:", busiest_day)

plt.figure(figsize=(9, 5))
plt.bar(hour_counts.index, hour_counts.values, color="purple")
plt.title("Orders by Hour")
plt.xlabel("Hour")
plt.ylabel("Orders")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(weekday_counts.index, weekday_counts.values, color="teal")
plt.title("Orders by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# TASK 11: List and tuple
print("\nTASK 11: LIST AND TUPLE")
top_dishes = top10.index.tolist()
meal_categories = ("Breakfast", "Lunch", "Snacks", "Dinner")
print("Top dishes (list):", top_dishes)
print("Meal categories (tuple):", meal_categories)

# TASK 12: Set of unique customers and order total function
print("\nTASK 12: UNIQUE CUSTOMERS AND ORDER TOTAL")
unique_customers = set(df["Customer_ID"].dropna())
print("Unique customers:", len(unique_customers))

def calculate_order_total(order_id):
    rows = df[df["Order_ID"] == order_id]
    if rows.empty:
        return None
    return round(rows["Item_Revenue"].sum(), 2)

sample_id = df["Order_ID"].iloc[0]
print("Total bill for Order", sample_id, ":", calculate_order_total(sample_id))

# TASK 13: NumPy statistics
print("\nTASK 13: NUMPY BILL STATISTICS")
bills = order_df["Bill_Value"].to_numpy()
print("Mean bill:", round(np.mean(bills), 2))
print("Median bill:", round(np.median(bills), 2))
print("Maximum bill:", round(np.max(bills), 2))

# TASK 14: IQR outliers and Time_Category
print("\nTASK 14: IQR OUTLIERS AND TIME CATEGORY")
q1 = np.percentile(bills, 25)
q3 = np.percentile(bills, 75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
print("Q1:", round(q1, 2), "Q3:", round(q3, 2), "IQR:", round(iqr, 2))
print("Upper limit:", round(upper_limit, 2))
large_bills = order_df[order_df["Bill_Value"] > upper_limit]
print("Unusually large bills:", len(large_bills))

def get_time_category(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["Time_Category"] = df["Hour"].apply(get_time_category)
order_df["Time_Category"] = order_df["Hour"].apply(get_time_category)
print(order_df["Time_Category"].value_counts())

# TASK 15: MenuItem class and recommendations
print("\nTASK 15: MENUITEM CLASS AND RECOMMENDATIONS")

class MenuItem:
    def __init__(self, name, category, price):
        self.name = name
        self.category = category
        self.price = price

    def show(self):
        return f"{self.name} | {self.category} | {self.price:.2f}"

for dish in top_dishes[:5]:
    rows = df[df["Item"] == dish]
    print(MenuItem(dish, rows["Category"].mode()[0], rows["Item_Price"].median()).show())

low_sellers = items.sort_values("Total_Quantity").head(3).index.tolist()

print("\nRecommendations:")
print(f"1. Always keep '{top_dishes[0]}' (best seller) and other top dishes in stock and promote them.")
print(f"2. Expand the '{category_revenue.index[0]}' category (highest revenue) and review '{category_revenue.index[-1]}' (lowest revenue).")
print(f"3. Add extra staff in peak hours {peak_hours.index.tolist()} and on {busiest_day}.")
print(f"4. '{avg_bill.index[0]}' orders have the highest average bill; offer combo deals for '{avg_bill.index[-1]}' orders.")
print(f"5. Review low-selling items {low_sellers}; {len(large_bills)} unusually large bills suggest adding family combo platters.")

# CONCLUSION
print("\nCONCLUSION")
print(f"{total_orders} unique orders, {total_items} items sold, revenue {total_revenue:,.2f}.")
print(f"Top dish: {top_dishes[0]}. Top category: {category_revenue.index[0]}.")
print(f"Average bill: {np.mean(bills):.2f}. Busiest day: {busiest_day}. Peak hours: {peak_hours.index.tolist()}.")