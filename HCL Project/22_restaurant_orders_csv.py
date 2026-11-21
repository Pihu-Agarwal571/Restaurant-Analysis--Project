import numpy as np
import pandas as pd

np.random.seed(42)

menu = {
    "Paneer Tikka": ("Starters", 250), "Veg Spring Roll": ("Starters", 160),
    "Chicken Wings": ("Starters", 280), "Dal Makhani": ("Main Course", 220),
    "Butter Chicken": ("Main Course", 340), "Veg Biryani": ("Main Course", 240),
    "Chicken Biryani": ("Main Course", 300), "Butter Naan": ("Breads", 40),
    "Garlic Naan": ("Breads", 50), "Gulab Jamun": ("Desserts", 90),
    "Ice Cream": ("Desserts", 110), "Cold Coffee": ("Beverages", 120),
    "Masala Chai": ("Beverages", 50), "Fresh Lime Soda": ("Beverages", 70),
}
items = list(menu.keys())
types = ["Dine-In", "Takeaway", "Delivery"]

# Hours 8 se 23 tak ke weights (total apne aap 1 me convert hoga)
hours = list(range(8, 24))
weights = np.array([2, 3, 6, 10, 8, 4, 3, 3, 6, 10, 12, 10, 8, 5, 3, 2])
hour_probs = weights / weights.sum()

rows = []
for order_id in range(1001, 1401):
    customer = f"C{np.random.randint(1, 151):03d}"
    date = pd.Timestamp("2026-10-01") + pd.to_timedelta(np.random.randint(0, 30), unit="D")
    hour = int(np.random.choice(hours, p=hour_probs))
    order_time = date + pd.Timedelta(hours=hour, minutes=int(np.random.randint(0, 60)))
    order_type = np.random.choice(types, p=[0.5, 0.2, 0.3])
    for dish in np.random.choice(items, size=np.random.randint(1, 5), replace=False):
        cat, price = menu[dish]
        rows.append([order_id, customer, order_time, dish, cat,
                     np.random.randint(1, 5), price, order_type])

sample = pd.DataFrame(rows, columns=["Order_ID", "Customer_ID", "Order_DateTime", "Item",
                                     "Category", "Item_Quantity", "Item_Price", "Order_Type"])

# Thode missing values daal rahe hain taaki cleaning practice ho sake
for col in ["Item", "Item_Quantity", "Item_Price"]:
    sample.loc[sample.sample(frac=0.02, random_state=1).index, col] = np.nan

sample.to_csv("22_restaurant_orders.csv", index=False)
print("Sample file created:", sample.shape)