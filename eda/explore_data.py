import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")
december = pd.read_csv("data/december-chart-inputs.csv")

# print("===== TRAIN =====")
# print("Shape:", train.shape)
# print(train.head())
# print(train.info())

# print("\n===== VALIDATION =====")
# print("Shape:", validation.shape)
# print(validation.head())
# print(validation.info())

# print("\n===== DECEMBER =====")
# print("Shape:", december.shape)
# print(december.head())
# print(december.info())


# print("\n===== COLUMN NAMES =====")
# print(train.columns.tolist())


# print("\n===== MISSING VALUES: TRAIN =====")
# print(train.isnull().sum())


# print("\n===== MISSING VALUES: VALIDATION =====")
# print(validation.isnull().sum())


# print("\n===== DUPLICATES =====")
# print("Train duplicates:", train.duplicated().sum())
# print("Validation duplicates:", validation.duplicated().sum())


# print("\n===== UNIQUE VALUES =====")

# for column in train.columns:
#     print(f"{column}: {train[column].nunique()}")

# print("\n===== NUMERICAL SUMMARY =====")
# print(train.describe())

# print("\n===== DATE INFORMATION =====")

# print("Minimum date:", train["date"].min())
# print("Maximum date:", train["date"].max())

# print("\nDate counts:")
# print(train["date"].value_counts().sort_index())
# print("\n===== EQUIPMENT TYPES =====")
# print(train["equipment"].value_counts())

# print("\n===== EQUIPMENT PERCENTAGE =====")
# print(train["equipment"].value_counts(normalize=True) * 100)


# print("\n===== WEIGHT CHECK =====")
# print("Negative weights:", (train["weight"] < 0).sum())
# print("Zero weights:", (train["weight"] == 0).sum())
# print("Missing weights:", train["weight"].isna().sum())

# print("\nNegative weight values:")
# print(train.loc[train["weight"] < 0, "weight"].value_counts().sort_index())


# print("\n===== WEIGHT SUMMARY =====")
# print(train["weight"].describe())


# print("\n===== POSTED RATE SUMMARY =====")
# print(train["posted_rate"].describe())


# print("\n===== POSTED RATE QUANTILES =====")
# print(train["posted_rate"].quantile([
#     0,
#     0.01,
#     0.05,
#     0.25,
#     0.50,
#     0.75,
#     0.90,
#     0.95,
#     0.99,
#     1.00
# ]))


# print("\n===== MARKET INDEX SUMMARY =====")
# print(train["market_index"].describe())


# print("\n===== QUOTE SIGNAL SUMMARY =====")
# print(train["quote_signal"].describe())


# print("\n===== NEGATIVE WEIGHT DETAILS ========@@")

# negative_weight = train[train["weight"] < 0]

# print("Number of negative weights:", len(negative_weight))

# print("\nNegative weight sample:")
# print(
#     negative_weight[
#         ["load_id", "pickup", "delivery", "weight", "date", "posted_rate"]
#     ].head(10)
# )


# print("\n===== LOCATION CHECK =====")

# print("Unique pickup locations:", train["pickup"].nunique())
# print("Unique delivery locations:", train["delivery"].nunique())

# print("\nPickup coordinates per pickup city:")
# print(
#     train.groupby("pickup")[["pickup_lat", "pickup_lon"]]
#     .nunique()
#     .sort_values("pickup_lat", ascending=False)
#     .head(10)
# )

# print("\nDelivery coordinates per delivery city:")
# print(
#     train.groupby("delivery")[["delivery_lat", "delivery_lon"]]
#     .nunique()
#     .sort_values("delivery_lat", ascending=False)
#     .head(10)
# )


# print("\n===== NUMERICAL CORRELATION WITH POSTED RATE =====")

# numeric_columns = [
#     "pickup_lat",
#     "pickup_lon",
#     "delivery_lat",
#     "delivery_lon",
#     "distance",
#     "weight",
#     "market_index",
#     "quote_signal",
#     "posted_rate"
# ]

# correlation = train[numeric_columns].corr()["posted_rate"].sort_values(
#     ascending=False
# )

# print(correlation)


# print("\n===== AVERAGE POSTED RATE BY EQUIPMENT =====")

# print(
#     train.groupby("equipment")["posted_rate"]
#     .agg(["count", "mean", "median", "min", "max"])
#     .sort_values("mean", ascending=False)
# )


# print("\n===== AVERAGE POSTED RATE BY MONTH =====")

# train["date_temp"] = pd.to_datetime(train["date"])

# monthly_rate = (
#     train.groupby(train["date_temp"].dt.to_period("M"))["posted_rate"]
#     .agg(["count", "mean", "median"])
# )

# print(monthly_rate)

# train.drop(columns=["date_temp"], inplace=True)


# print("\n===== NEGATIVE WEIGHT INVESTIGATION =====")

# negative = train[train["weight"] < 0].copy()

# print("Negative weight rows:", len(negative))

# print("\nNegative weight by equipment:")
# print(negative["equipment"].value_counts())

# print("\nNegative weight by month:")
# print(
#     negative.groupby(
#         pd.to_datetime(negative["date"]).dt.to_period("M")
#     ).size()
# )

# print("\nNegative weight rate summary:")
# print(negative["posted_rate"].describe())


# print("\n===== HIGH POSTED RATE INVESTIGATION =====")

# print("Rates above 6000:")
# print((train["posted_rate"] > 6000).sum())

# print("Rates above 10000:")
# print((train["posted_rate"] > 10000).sum())

# print("Rates above 20000:")
# print((train["posted_rate"] > 20000).sum())


# print("\nHighest posted rates:")
# print(
#     train.nlargest(
#         20,
#         "posted_rate"
#     )[
#         [
#             "load_id",
#             "pickup",
#             "delivery",
#             "distance",
#             "equipment",
#             "weight",
#             "date",
#             "market_index",
#             "quote_signal",
#             "posted_rate"
#         ]
#     ]
# )

import matplotlib.pyplot as plt

# print("\n===== VISUALIZATION =====")

# # 1. Distance vs Posted Rate
# plt.figure(figsize=(9, 6))
# plt.scatter(
#     train["distance"],
#     train["posted_rate"],
#     alpha=0.2,
#     s=10
# )
# plt.xlabel("Distance")
# plt.ylabel("Posted Rate")
# plt.title("Distance vs Posted Rate")
# plt.grid(alpha=0.2)
# plt.show()


# # 2. Posted Rate Distribution
# plt.figure(figsize=(9, 6))
# plt.hist(
#     train["posted_rate"],
#     bins=50
# )
# plt.xlabel("Posted Rate")
# plt.ylabel("Number of Loads")
# plt.title("Posted Rate Distribution")
# plt.grid(alpha=0.2)
# plt.show()


# # 3. Posted Rate by Equipment
# plt.figure(figsize=(8, 6))
# train.boxplot(
#     column="posted_rate",
#     by="equipment"
# )
# plt.xlabel("Equipment")
# plt.ylabel("Posted Rate")
# plt.title("Posted Rate by Equipment")
# plt.suptitle("")
# plt.grid(alpha=0.2)
# plt.show()
# print("\n===== FEATURE RELATIONSHIPS =====")

# # Market Index vs Posted Rate
# plt.figure(figsize=(9, 6))
# plt.scatter(
#     train["market_index"],
#     train["posted_rate"],
#     alpha=0.2,
#     s=10
# )
# plt.xlabel("Market Index")
# plt.ylabel("Posted Rate")
# plt.title("Market Index vs Posted Rate")
# plt.grid(alpha=0.2)
# plt.show()


# # Quote Signal vs Posted Rate
# plt.figure(figsize=(9, 6))
# plt.scatter(
#     train["quote_signal"],
#     train["posted_rate"],
#     alpha=0.2,
#     s=10
# )
# plt.xlabel("Quote Signal")
# plt.ylabel("Posted Rate")
# plt.title("Quote Signal vs Posted Rate")
# plt.grid(alpha=0.2)
# plt.show()


# # Weight vs Posted Rate
# plt.figure(figsize=(9, 6))
# plt.scatter(
#     train["weight"],
#     train["posted_rate"],
#     alpha=0.2,
#     s=10
# )
# plt.xlabel("Weight")
# plt.ylabel("Posted Rate")
# plt.title("Weight vs Posted Rate")
# plt.grid(alpha=0.2)
# plt.show()

print("\n===== TRAIN vs VALIDATION =====")

# Compare numerical features
numeric_features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal"
]

print("\n===== NUMERICAL COMPARISON =====")

comparison = pd.DataFrame({
    "Train Mean": train[numeric_features].mean(),
    "Validation Mean": validation[numeric_features].mean(),
    "Train Median": train[numeric_features].median(),
    "Validation Median": validation[numeric_features].median()
})

print(comparison)


# Compare categorical features
print("\n===== EQUIPMENT COMPARISON =====")

equipment_comparison = pd.DataFrame({
    "Train %": train["equipment"].value_counts(normalize=True) * 100,
    "Validation %": validation["equipment"].value_counts(normalize=True) * 100
})

print(equipment_comparison)


# Compare number of unique locations
print("\n===== LOCATION COMPARISON =====")

print("Train pickup locations:", train["pickup"].nunique())
print("Validation pickup locations:", validation["pickup"].nunique())

print("Train delivery locations:", train["delivery"].nunique())
print("Validation delivery locations:", validation["delivery"].nunique())


# Compare distance distribution
print("\n===== DISTANCE COMPARISON =====")

print("Train distance:")
print(train["distance"].describe())

print("\nValidation distance:")
print(validation["distance"].describe())


# Compare weight distribution
print("\n===== WEIGHT COMPARISON =====")

print("Train weight:")
print(train["weight"].describe())

print("\nValidation weight:")
print(validation["weight"].describe())


# Compare missing values
print("\n===== MISSING VALUE COMPARISON =====")

missing_comparison = pd.DataFrame({
    "Train Missing": train.isnull().sum(),
    "Validation Missing": validation.isnull().sum()
})

print(missing_comparison)

print("\n===== NEW LOCATIONS IN VALIDATION =====")

train_pickups = set(train["pickup"].unique())
validation_pickups = set(validation["pickup"].unique())

new_pickups = validation_pickups - train_pickups

print("New pickup locations:", len(new_pickups))
print(sorted(new_pickups))


train_deliveries = set(train["delivery"].unique())
validation_deliveries = set(validation["delivery"].unique())

new_deliveries = validation_deliveries - train_deliveries

print("\nNew delivery locations:", len(new_deliveries))
print(sorted(new_deliveries))


print("\n===== MARKET INDEX QUANTILES =====")

print("Train:")
print(
    train["market_index"].quantile(
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    )
)

print("\nValidation:")
print(
    validation["market_index"].quantile(
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    )
)