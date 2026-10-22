# =========================================
# GLOBAL RETAIL & CUSTOMER SEGMENTATION
# RFM ANALYSIS USING PYTHON
# =========================================

import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# 1. LOAD DATASET
# =========================================

df = pd.read_csv(
    r"C:\Users\Lenovo\Downloads\Global_Superstore_Cleaned (1).csv"
)

print("\nFirst 5 Rows:")
print(df.head())


# =========================================
# 2. DATASET INFORMATION
# =========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)


# =========================================
# 3. DATE CONVERSION
# =========================================

df["order_date"] = pd.to_datetime(df["order_date"])

print("\nOrder Date Range:")
print(
    df["order_date"].min(),
    "to",
    df["order_date"].max()
)


# =========================================
# 4. BASIC EDA
# =========================================

print("\nBasic Sales Summary:")

print(
    df[["sales"]].describe()
)


# =========================================
# 5. SALES BY CATEGORY
# =========================================

category_sales = (
    df.groupby("category", as_index=False)
    .agg(
        Total_Sales=("sales", "sum"),
        Total_Orders=("order_id", "nunique")
    )
    .sort_values("Total_Sales", ascending=False)
)

print("\nSales by Category:")
print(category_sales)


# =========================================
# 6. SALES BY REGION
# =========================================

region_sales = (
    df.groupby("region", as_index=False)
    .agg(
        Total_Sales=("sales", "sum"),
        Total_Orders=("order_id", "nunique")
    )
    .sort_values("Total_Sales", ascending=False)
)

print("\nSales by Region:")
print(region_sales)


# =========================================
# 7. MONTHLY SALES TREND
# =========================================

monthly_sales = (
    df.groupby(
        df["order_date"].dt.to_period("M")
    )
    .agg(
        Total_Sales=("sales", "sum")
    )
    .reset_index()
)

monthly_sales["order_date"] = monthly_sales[
    "order_date"
].astype(str)

print("\nMonthly Sales:")
print(monthly_sales.head(12))


# =========================================
# 8. RFM ANALYSIS - RECENCY
# =========================================

recency = (
    df.groupby(
        ["customer_id", "customer_name"]
    )
    .agg(
        Last_Purchase=("order_date", "max")
    )
    .reset_index()
)

reference_date = df["order_date"].max()

recency["Recency"] = (
    reference_date - recency["Last_Purchase"]
).dt.days

print("\nRecency Analysis:")
print(
    recency.sort_values("Recency").head(10)
)


# =========================================
# 9. RFM ANALYSIS - FREQUENCY
# =========================================

frequency = (
    df.groupby(
        ["customer_id", "customer_name"]
    )
    .agg(
        Frequency=("order_id", "nunique")
    )
    .reset_index()
)

print("\nFrequency Analysis:")
print(
    frequency.sort_values(
        "Frequency",
        ascending=False
    ).head(10)
)


# =========================================
# 10. RFM ANALYSIS - MONETARY
# =========================================

monetary = (
    df.groupby(
        ["customer_id", "customer_name"]
    )
    .agg(
        Monetary=("sales", "sum")
    )
    .reset_index()
)

print("\nMonetary Analysis:")
print(
    monetary.sort_values(
        "Monetary",
        ascending=False
    ).head(10)
)


# =========================================
# 11. COMBINE RFM VALUES
# =========================================

rfm = (
    recency[
        [
            "customer_id",
            "customer_name",
            "Recency"
        ]
    ]
    .merge(
        frequency,
        on=[
            "customer_id",
            "customer_name"
        ]
    )
    .merge(
        monetary,
        on=[
            "customer_id",
            "customer_name"
        ]
    )
)

print("\nRFM Table:")
print(rfm.head(10))


# =========================================
# 12. RFM SCORING
# =========================================

# Lower Recency = Better
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
)

# Higher Frequency = Better
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

# Higher Monetary = Better
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

# Convert scores to integers
rfm["R_Score"] = rfm["R_Score"].astype(int)
rfm["F_Score"] = rfm["F_Score"].astype(int)
rfm["M_Score"] = rfm["M_Score"].astype(int)


print("\nRFM Scores:")
print(
    rfm[
        [
            "customer_id",
            "customer_name",
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score"
        ]
    ].head(10)
)


# =========================================
# 13. CUSTOMER SEGMENTATION
# =========================================

def segment_customer(row):

    if (
        row["R_Score"] >= 4
        and row["F_Score"] >= 4
        and row["M_Score"] >= 4
    ):
        return "Champions"

    elif (
        row["R_Score"] >= 3
        and row["F_Score"] >= 4
    ):
        return "Loyal Customers"

    elif (
        row["R_Score"] <= 2
        and row["F_Score"] >= 3
    ):
        return "At Risk"

    elif (
        row["R_Score"] >= 4
        and row["F_Score"] <= 2
    ):
        return "New Customers"

    else:
        return "Lost Customers"


rfm["Customer_Segment"] = rfm.apply(
    segment_customer,
    axis=1
)


print("\nCustomer Segmentation:")

print(
    rfm[
        [
            "customer_id",
            "customer_name",
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "Customer_Segment"
        ]
    ].head(20)
)


# =========================================
# 14. SEGMENT SUMMARY
# =========================================

segment_summary = (
    rfm.groupby(
        "Customer_Segment",
        as_index=False
    )
    .agg(
        Customer_Count=("customer_id", "count"),
        Total_Sales=("Monetary", "sum"),
        Average_Customer_Value=("Monetary", "mean")
    )
    .sort_values(
        "Total_Sales",
        ascending=False
    )
)

segment_summary[
    "Total_Sales"
] = segment_summary[
    "Total_Sales"
].round(2)

segment_summary[
    "Average_Customer_Value"
] = segment_summary[
    "Average_Customer_Value"
].round(2)


print("\nSegment Summary:")
print(segment_summary)


# =========================================
# 15. EXPORT RFM RESULTS
# =========================================

rfm.to_csv(
    "rfm_customer_segments.csv",
    index=False
)

segment_summary.to_csv(
    "customer_segment_summary.csv",
    index=False
)

print("\nRFM analysis completed successfully!")

print(
    "\nFiles created:"
)

print("1. rfm_customer_segments.csv")
print("2. customer_segment_summary.csv")