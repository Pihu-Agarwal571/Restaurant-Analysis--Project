# =====================================================================
# Project 22: Restaurant Orders & Menu Performance Analysis
# Tools: Pandas, NumPy, Matplotlib
# =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def heading(text):
    """Prints a clear heading for each task."""
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


# =====================================================================
# TASK 1: Load and inspect the dataset
# =====================================================================
heading("TASK 1: LOAD AND INSPECT THE DATASET")

FILE_NAME = "22_restaurant_orders.csv"

# If the file is not in the current folder, offer an upload option (Google Colab)
if not os.path.exists(FILE_NAME):
    try:
        from google.colab import files
        print("CSV not found. Please upload '22_restaurant_orders.csv' now...")
        uploaded = files.upload()
        FILE_NAME = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError(
            "Place '22_restaurant_orders.csv' in the same folder as this notebook/script."
        )

df = pd.read_csv(FILE_NAME)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape (rows, columns) ---")
print(df.shape)

print("\n--- Column names ---")
print(list(df.columns))

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Basic information ---")
df.info()

print("\n--- Basic statistics (numeric columns) ---")
print(df.describe())

# ---- Match actual column names to the standard names used in this project ----
# (Nothing is invented: we only rename columns that really exist.)
expected_columns = {
    "Order_ID":       ["order_id", "orderid", "order_no", "order_number"],
    "Customer_ID":    ["customer_id", "customerid", "cust_id"],
    "Order_DateTime": ["order_datetime", "order_date_time", "datetime", "order_time",
                       "order_date", "timestamp"],
    "Item":           ["item", "item_name", "dish", "dish_name", "menu_item"],
    "Category":       ["category", "menu_category", "item_category"],
    "Item_Quantity":  ["item_quantity", "quantity", "qty"],
    "Item_Price":     ["item_price", "price", "unit_price"],
    "Order_Type":     ["order_type", "ordertype", "order_mode", "type"],
}

normalized = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
rename_map = {}
missing_columns = []

for standard_name, options in expected_columns.items():
    found = None
    for option in [standard_name.lower()] + options:
        if option in normalized:
            found = normalized[option]
            break
    if found is None:
        missing_columns.append(standard_name)
    else:
        rename_map[found] = standard_name

df = df.rename(columns=rename_map)

print("\n--- Column matching result ---")
print("Matched columns :", rename_map)
print("Missing columns :", missing_columns if missing_columns else "None")

# Columns that are absolutely required for the analysis
required = ["Order_ID", "Order_DateTime", "Item", "Item_Quantity", "Item_Price"]
stop_list = [c for c in required if c in missing_columns]
if stop_list:
    raise ValueError(f"Required column(s) not found in the dataset: {stop_list}. "
                     "Check the column names printed above.")

# Optional columns: if missing, only the related parts are skipped
HAS_CUSTOMER = "Customer_ID" in df.columns
HAS_CATEGORY = "Category" in df.columns
HAS_ORDER_TYPE = "Order_Type" in df.columns

if not HAS_CUSTOMER:
    print("NOTE: Customer_ID is missing -> unique-customer analysis will be skipped.")
if not HAS_CATEGORY:
    print("NOTE: Category is missing -> category analysis will be skipped.")
if not HAS_ORDER_TYPE:
    print("NOTE: Order_Type is missing -> order-type comparison will be skipped.")


# =====================================================================
# TASK 2: Missing values and duplicate Order_ID records
# =====================================================================
heading("TASK 2: MISSING VALUES AND DUPLICATE ORDER_IDs")

# Missing values in every column
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df) * 100).round(2)
missing_table = pd.DataFrame({"Missing_Count": missing_counts,
                              "Missing_Percent": missing_percent})
print("\n--- Missing values per column ---")
print(missing_table)

# Duplicate Order_ID records
# NOTE: One order can have several item rows, so repeated Order_IDs can be normal.
rows_with_repeated_id = df["Order_ID"].duplicated(keep=False).sum()
repeated_ids = (df["Order_ID"].value_counts() > 1).sum()
exact_duplicate_rows = df.duplicated().sum()

print("\n--- Duplicate checks ---")
print("Total rows (order records)                       :", len(df))
print("Unique Order_IDs                                 :", df["Order_ID"].nunique())
print("Order_IDs that appear more than once             :", repeated_ids)
print("Rows that share an Order_ID with another row     :", rows_with_repeated_id)
print("Completely identical (exact duplicate) rows      :", exact_duplicate_rows)

if repeated_ids > 0:
    print("\nSample of rows with repeated Order_ID:")
    print(df[df["Order_ID"].duplicated(keep=False)].sort_values("Order_ID").head(10))


# =====================================================================
# TASK 3: Clean missing values (Item, Item_Quantity, Item_Price)
# =====================================================================
heading("TASK 3: DATA CLEANING")

# Step 1: Remove exact duplicate rows (same data repeated = data-entry duplicates)
before = len(df)
df = df.drop_duplicates().copy()
print("Exact duplicate rows removed:", before - len(df))

# Step 2: Remove rows with no Order_ID (they cannot be linked to any order)
before = len(df)
df = df.dropna(subset=["Order_ID"]).copy()
print("Rows removed for missing Order_ID:", before - len(df))

# Step 3: Strip extra spaces in text columns and treat empty text as missing
text_columns = ["Item"]
if HAS_CATEGORY:
    text_columns.append("Category")
if HAS_ORDER_TYPE:
    text_columns.append("Order_Type")

for col in text_columns:
    df[col] = df[col].where(df[col].isnull(), df[col].astype(str).str.strip())
    df[col] = df[col].replace("", np.nan)


# Step 4: Convert quantity and price to numbers. Invalid text becomes NaN.
def to_clean_number(series):
    """Removes symbols like currency signs and converts to a number."""
    text = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


df["Item_Quantity"] = to_clean_number(df["Item_Quantity"])
df["Item_Price"] = to_clean_number(df["Item_Price"])

# Quantity and price must be positive; zero/negative values are invalid
invalid_qty = (df["Item_Quantity"] <= 0).sum()
invalid_price = (df["Item_Price"] <= 0).sum()
df.loc[df["Item_Quantity"] <= 0, "Item_Quantity"] = np.nan
df.loc[df["Item_Price"] <= 0, "Item_Price"] = np.nan
print("Invalid (<= 0) quantities set to missing:", invalid_qty)
print("Invalid (<= 0) prices set to missing    :", invalid_price)

print("\nMissing values BEFORE filling:")
print(df[["Item", "Item_Quantity", "Item_Price"]].isnull().sum())

# Step 5: Fill missing values (instead of deleting rows)
# - Item     : filled with "Unknown Item" (the order row and its revenue are kept;
#              this label is excluded from dish rankings later).
# - Quantity : filled with the MEDIAN quantity of the same item (median is not
#              affected by extreme values); if still missing, overall median is used.
#              Quantity is a whole number, so it is rounded (minimum 1).
# - Price    : filled with the MEDIAN price of the same item, because the same dish
#              normally has a similar price; if still missing, overall median is used.
df["Item"] = df["Item"].fillna("Unknown Item")

item_median_qty = df.groupby("Item")["Item_Quantity"].transform("median")
df["Item_Quantity"] = df["Item_Quantity"].fillna(item_median_qty)
df["Item_Quantity"] = df["Item_Quantity"].fillna(df["Item_Quantity"].median())
df["Item_Quantity"] = df["Item_Quantity"].round().clip(lower=1).astype(int)

item_median_price = df.groupby("Item")["Item_Price"].transform("median")
df["Item_Price"] = df["Item_Price"].fillna(item_median_price)
df["Item_Price"] = df["Item_Price"].fillna(df["Item_Price"].median())

# Fill missing text categories with a label so no rows are lost
if HAS_CATEGORY:
    df["Category"] = df["Category"].fillna("Unknown")
if HAS_ORDER_TYPE:
    df["Order_Type"] = df["Order_Type"].fillna("Unknown")

print("\nMissing values AFTER filling:")
print(df[["Item", "Item_Quantity", "Item_Price"]].isnull().sum())
print("\nShape after cleaning:", df.shape)

print("\nWhy this method? Rows were NOT deleted for missing Item/Quantity/Price. "
      "Median filling keeps the data size, is robust to outliers, and uses "
      "information from the same dish, so the dataset stays usable.")


# =====================================================================
# TASK 4: Convert Order_DateTime and extract hour and weekday
# =====================================================================
heading("TASK 4: DATETIME CONVERSION")

# Invalid date text becomes NaT (missing datetime)
df["Order_DateTime"] = pd.to_datetime(df["Order_DateTime"], errors="coerce")
print("Invalid/missing datetime values:", df["Order_DateTime"].isnull().sum())

# If one row of an order has no time, copy it from another row of the same order
df["Order_DateTime"] = df["Order_DateTime"].fillna(
    df.groupby("Order_ID")["Order_DateTime"].transform("first")
)

# Rows that still have no datetime cannot be used for time analysis
before = len(df)
df = df.dropna(subset=["Order_DateTime"]).copy()
print("Rows removed (datetime still unknown):", before - len(df))

# Extract hour and weekday using Pandas datetime functionality
df["Hour"] = df["Order_DateTime"].dt.hour
df["Weekday"] = df["Order_DateTime"].dt.day_name()

print("\nDatetime column type:", df["Order_DateTime"].dtype)
print(df[["Order_ID", "Order_DateTime", "Hour", "Weekday"]].head())


# =====================================================================
# TASK 5: Total orders, total items sold, total revenue
# =====================================================================
heading("TASK 5: TOTAL ORDERS, ITEMS SOLD AND REVENUE")

total_order_records = len(df)                      # rows in the dataset
total_unique_orders = df["Order_ID"].nunique()     # real number of orders
total_items_sold = df["Item_Quantity"].sum()       # sum of quantities
total_revenue = (df["Item_Quantity"] * df["Item_Price"]).sum()

print("Total order records (rows)    :", total_order_records)
print("Total unique orders           :", total_unique_orders)
print("Total items sold (quantity)   :", total_items_sold)
print(f"Total revenue                 : {total_revenue:,.2f}")


# =====================================================================
# TASK 6: Create Item_Revenue column
# =====================================================================
heading("TASK 6: ITEM_REVENUE COLUMN")

df["Item_Revenue"] = df["Item_Quantity"] * df["Item_Price"]
print(df[["Order_ID", "Item", "Item_Quantity", "Item_Price", "Item_Revenue"]].head())
print(f"\nSum of Item_Revenue: {df['Item_Revenue'].sum():,.2f} "
      "(matches the total revenue above)")


# =====================================================================
# TASK 7: Top 10 most frequently ordered dishes
# =====================================================================
heading("TASK 7: TOP 10 MOST ORDERED DISHES")

# Group by dish: quantity sold, times ordered (rows), and revenue
item_summary = df.groupby("Item").agg(
    Total_Quantity=("Item_Quantity", "sum"),
    Times_Ordered=("Order_ID", "count"),
    Total_Revenue=("Item_Revenue", "sum")
).reset_index()

# "Unknown Item" is not a real dish, so it is excluded from rankings
item_summary = item_summary[item_summary["Item"] != "Unknown Item"]
item_summary = item_summary.sort_values(["Total_Quantity", "Total_Revenue"],
                                        ascending=False)
top10 = item_summary.head(10).reset_index(drop=True)
top10.index = top10.index + 1
print(top10)

# Bar chart: Top 10 dishes
plt.figure(figsize=(9, 5))
plt.barh(top10["Item"], top10["Total_Quantity"], color="steelblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Most Ordered Dishes (by Quantity Sold)")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Dish")
plt.tight_layout()
plt.show()


# =====================================================================
# TASK 8: Rank categories by total revenue
# =====================================================================
heading("TASK 8: CATEGORY-WISE REVENUE")

if HAS_CATEGORY:
    category_revenue = (df.groupby("Category")["Item_Revenue"].sum()
                        .sort_values(ascending=False).reset_index())
    category_revenue.columns = ["Category", "Total_Revenue"]
    category_revenue["Revenue_Share_%"] = (
        category_revenue["Total_Revenue"] / category_revenue["Total_Revenue"].sum() * 100
    ).round(2)
    category_revenue["Total_Revenue"] = category_revenue["Total_Revenue"].round(2)
    category_revenue.insert(0, "Rank", range(1, len(category_revenue) + 1))
    print(category_revenue.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.bar(category_revenue["Category"], category_revenue["Total_Revenue"],
            color="darkorange")
    plt.title("Category-wise Total Revenue")
    plt.xlabel("Category")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("Skipped: Category column is not available in the dataset.")


# =====================================================================
# TASK 9: Average bill by order type (Dine-In / Takeaway / Delivery)
# =====================================================================
heading("TASK 9: AVERAGE BILL BY ORDER TYPE")

# IMPORTANT: One order may have many item rows.
# So we first add up Item_Revenue for each Order_ID to get the real bill.
agg_dict = {
    "Bill_Value": ("Item_Revenue", "sum"),
    "Items_In_Order": ("Item_Quantity", "sum"),
    "Order_DateTime": ("Order_DateTime", "first"),
    "Hour": ("Hour", "first"),
    "Weekday": ("Weekday", "first"),
}
if HAS_CUSTOMER:
    agg_dict["Customer_ID"] = ("Customer_ID", "first")
if HAS_ORDER_TYPE:
    agg_dict["Order_Type"] = ("Order_Type", "first")

order_df = df.groupby("Order_ID").agg(**agg_dict).reset_index()

print("Order-level table created (one row = one complete bill).")
print("Item rows:", len(df), "| Unique orders (bills):", len(order_df))
print(order_df.head())

if HAS_ORDER_TYPE:
    mixed = (df.groupby("Order_ID")["Order_Type"].nunique() > 1).sum()
    print("\nOrders having more than one Order_Type value:", mixed,
          "(first value is used)")

    avg_bill_by_type = (order_df.groupby("Order_Type")["Bill_Value"]
                        .agg(["count", "mean"]).round(2)
                        .sort_values("mean", ascending=False))
    avg_bill_by_type.columns = ["Number_of_Orders", "Average_Bill"]
    print("\n--- Average bill per order type ---")
    print(avg_bill_by_type)

    plt.figure(figsize=(7, 5))
    plt.bar(avg_bill_by_type.index, avg_bill_by_type["Average_Bill"], color="seagreen")
    plt.title("Average Bill Value by Order Type")
    plt.xlabel("Order Type")
    plt.ylabel("Average Bill")
    plt.tight_layout()
    plt.show()
else:
    print("Skipped: Order_Type column is not available in the dataset.")


# =====================================================================
# TASK 10: Peak ordering hours and busiest weekday
# =====================================================================
heading("TASK 10: PEAK HOURS AND BUSIEST DAY")

# Orders are counted as UNIQUE orders (not item rows)
hour_counts = order_df["Hour"].value_counts().sort_index()
top_hours = hour_counts.sort_values(ascending=False).head(3)

print("--- Orders per hour ---")
print(hour_counts)
print("\nTop 3 peak ordering hours:")
for hour, count in top_hours.items():
    print(f"  {hour:02d}:00 -> {count} orders")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
weekday_counts = order_df["Weekday"].value_counts().reindex(weekday_order, fill_value=0)
busiest_day = weekday_counts.idxmax()

print("\n--- Orders per weekday ---")
print(weekday_counts)
print(f"\nBusiest day of the week: {busiest_day} ({weekday_counts.max()} orders)")

plt.figure(figsize=(9, 5))
plt.bar(hour_counts.index, hour_counts.values, color="purple")
plt.title("Number of Orders by Hour of the Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Orders")
plt.xticks(hour_counts.index)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(weekday_counts.index, weekday_counts.values, color="teal")
plt.title("Number of Orders by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =====================================================================
# TASK 11: List and Tuple
# =====================================================================
heading("TASK 11: LIST AND TUPLE")

# List: names of the top dishes (a list can be changed)
top_dish_names = top10["Item"].tolist()
print("List of top dishes:")
print(top_dish_names)

# Tuple: fixed meal categories (a tuple cannot be changed)
meal_categories = ("Breakfast", "Lunch", "Snacks", "Dinner")
print("\nTuple of fixed meal categories:")
print(meal_categories)
print("Type of list :", type(top_dish_names))
print("Type of tuple:", type(meal_categories))


# =====================================================================
# TASK 12: Set of unique customers + function for an order's total bill
# =====================================================================
heading("TASK 12: UNIQUE CUSTOMERS (SET) AND ORDER TOTAL FUNCTION")

if HAS_CUSTOMER:
    # A set stores only unique values
    unique_customers = set(df["Customer_ID"].dropna())
    print("Number of unique customers:", len(unique_customers))
    print("Sample customer IDs:", list(unique_customers)[:5])
else:
    print("Skipped unique-customer set: Customer_ID column is not available.")


def calculate_order_total(order_id):
    """Returns the total bill (sum of Item_Revenue) for the given Order_ID."""
    order_rows = df[df["Order_ID"] == order_id]
    if order_rows.empty:
        return None
    return round(order_rows["Item_Revenue"].sum(), 2)


# Example: change 'selected_order_id' to test any other Order_ID
selected_order_id = df["Order_ID"].iloc[0]
result = calculate_order_total(selected_order_id)
print(f"\nSelected Order_ID: {selected_order_id}")
print("Items in this order:")
print(df[df["Order_ID"] == selected_order_id][["Item", "Item_Quantity",
                                               "Item_Price", "Item_Revenue"]])
if result is None:
    print("Order_ID not found.")
else:
    print(f"Total bill for this order: {result:,.2f}")


# =====================================================================
# TASK 13: NumPy - mean, median, maximum bill
# =====================================================================
heading("TASK 13: NUMPY BILL STATISTICS")

bill_array = order_df["Bill_Value"].to_numpy()   # one value per complete order

mean_bill = np.mean(bill_array)
median_bill = np.median(bill_array)
max_bill = np.max(bill_array)

print(f"Mean bill value   : {mean_bill:,.2f}")
print(f"Median bill value : {median_bill:,.2f}")
print(f"Maximum bill value: {max_bill:,.2f}")


# =====================================================================
# TASK 14: IQR outliers and Time_Category
# =====================================================================
heading("TASK 14: UNUSUALLY LARGE BILLS (IQR) AND TIME CATEGORY")

# IQR method on order-level bills
q1 = np.percentile(bill_array, 25)
q3 = np.percentile(bill_array, 75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

print(f"Q1 (25th percentile): {q1:,.2f}")
print(f"Q3 (75th percentile): {q3:,.2f}")
print(f"IQR                 : {iqr:,.2f}")
print(f"Lower limit         : {lower_limit:,.2f}")
print(f"Upper limit         : {upper_limit:,.2f}")

large_bills = order_df[order_df["Bill_Value"] > upper_limit]
large_bills = large_bills.sort_values("Bill_Value", ascending=False)
print("\nNumber of unusually large bills:", len(large_bills))
print(f"Percentage of all orders: {len(large_bills) / len(order_df) * 100:.2f}%")
if len(large_bills) > 0:
    print("\nTop 10 largest bills:")
    print(large_bills[["Order_ID", "Bill_Value"]].head(10).to_string(index=False))


# Time_Category based on order hour
def get_time_category(hour):
    """Morning 5-11, Afternoon 12-16, Evening 17-20, Night 21-4."""
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

time_order = ["Morning", "Afternoon", "Evening", "Night"]
time_summary = order_df.groupby("Time_Category")["Bill_Value"].agg(["count", "mean"])
time_summary = time_summary.reindex(time_order).fillna(0).round(2)
time_summary.columns = ["Number_of_Orders", "Average_Bill"]
print("\n--- Orders by Time_Category ---")
print(time_summary)
busiest_time = time_summary["Number_of_Orders"].idxmax()


# =====================================================================
# TASK 15: MenuItem class and menu-planning recommendations
# =====================================================================
heading("TASK 15: MENUITEM CLASS AND RECOMMENDATIONS")


class MenuItem:
    """A simple class to store details of one menu item."""

    def __init__(self, name, category, price):
        self.name = name
        self.category = category
        self.price = price

    def display(self):
        return f"{self.name} | Category: {self.category} | Price: {self.price:.2f}"


# Create MenuItem objects for the top 5 dishes using actual data
menu_objects = []
for dish in top_dish_names[:5]:
    dish_rows = df[df["Item"] == dish]
    dish_price = dish_rows["Item_Price"].median()
    if HAS_CATEGORY:
        dish_category = dish_rows["Category"].mode()[0]
    else:
        dish_category = "N/A"
    menu_objects.append(MenuItem(dish, dish_category, dish_price))

print("Top 5 dishes stored as MenuItem objects:")
for m in menu_objects:
    print(" -", m.display())

# ---- Five recommendations built from the calculated results ----
recommendations = []

# 1. Popular dishes
best = top10.iloc[0]
recommendations.append(
    f"Keep '{best['Item']}' (best seller: {best['Total_Quantity']} units sold) and the "
    f"other top dishes always in stock, and feature them as 'Chef's Special / "
    f"Most Popular' on the menu."
)

# 2. Category performance
if HAS_CATEGORY:
    top_cat = category_revenue.iloc[0]
    low_cat = category_revenue.iloc[-1]
    recommendations.append(
        f"Expand variety in the '{top_cat['Category']}' category, which earns the most "
        f"revenue ({top_cat['Revenue_Share_%']}% of the total). Review the "
        f"'{low_cat['Category']}' category (lowest revenue, "
        f"{low_cat['Revenue_Share_%']}%) with promotions or menu changes."
    )
else:
    recommendations.append(
        "Add a Category column to the data so that category revenue can be tracked "
        "and menu sections can be planned accordingly."
    )

# 3. Staffing and preparation
peak_hours_text = ", ".join([f"{h:02d}:00" for h in top_hours.index])
recommendations.append(
    f"Schedule extra staff and pre-prepare ingredients for the peak hours "
    f"({peak_hours_text}), the busiest day ({busiest_day}) and the "
    f"{busiest_time} period, which has the most orders."
)

# 4. Order types
if HAS_ORDER_TYPE:
    best_type = avg_bill_by_type.index[0]
    best_type_bill = avg_bill_by_type["Average_Bill"].iloc[0]
    low_type = avg_bill_by_type.index[-1]
    low_type_bill = avg_bill_by_type["Average_Bill"].iloc[-1]
    recommendations.append(
        f"'{best_type}' orders have the highest average bill ({best_type_bill:,.2f}); "
        f"promote them. For '{low_type}' (average {low_type_bill:,.2f}), offer combo "
        f"deals or add-ons to increase the bill value."
    )
else:
    recommendations.append(
        "Record the order type (Dine-In/Takeaway/Delivery) so that average bills can "
        "be compared and offers designed for each type."
    )

# 5. Low sellers and large bills
low_sellers = item_summary.sort_values("Total_Quantity").head(5)["Item"].tolist()
recommendations.append(
    f"Review or redesign the lowest-selling items ({', '.join(low_sellers)}). "
    f"Also, {len(large_bills)} orders had unusually large bills (above "
    f"{upper_limit:,.2f}); create group/family combo platters to encourage more "
    f"such high-value orders."
)

print("\n--- FIVE MENU-PLANNING RECOMMENDATIONS ---")
for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")


# =====================================================================
# FINAL CONCLUSION (all values come from the analysis above)
# =====================================================================
heading("FINAL CONCLUSION")

print(f"- The dataset has {total_order_records} item records forming "
      f"{total_unique_orders} unique orders, with {total_items_sold} items sold "
      f"and total revenue of {total_revenue:,.2f}.")
print(f"- Most popular dish: '{top10.iloc[0]['Item']}' "
      f"({top10.iloc[0]['Total_Quantity']} units).")
if HAS_CATEGORY:
    print(f"- Highest-revenue category: '{category_revenue.iloc[0]['Category']}'.")
print(f"- Average bill per order: {mean_bill:,.2f} (median {median_bill:,.2f}, "
      f"maximum {max_bill:,.2f}).")
if HAS_ORDER_TYPE:
    print(f"- Order type with the highest average bill: '{avg_bill_by_type.index[0]}'.")
print(f"- Peak hours: {peak_hours_text}; busiest weekday: {busiest_day}; "
      f"busiest time period: {busiest_time}.")
print(f"- {len(large_bills)} orders were unusually large bills (IQR method).")
print("- The recommendations above can help in stock planning, staffing, "
      "pricing offers and menu improvement.")