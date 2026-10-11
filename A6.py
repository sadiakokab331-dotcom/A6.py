import seaborn as sns
import matplotlib.pyplot as plt

planets = sns.load_dataset("planets")

print("--- First 5 Rows of the Dataset ---")
print(planets.head())

print("\n--- Dataset Information ---")
print(planets.info())

plt.figure(figsize=(10, 6))

sns.histplot(data=planets, x="orbital_period", log_scale=True, color="skyblue")

plt.title("Distribution of Planet Orbital Periods (Log Scale)")
plt.xlabel("Orbital Period (days)")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(
    data=planets, x="distance", kde=True, log_scale=True, color="teal"
)

plt.title("Distribution of Planet Distance with KDE Curve")
plt.xlabel("Distance (Light Years / Parsecs)")
plt.ylabel("Density / Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(12, 8))

top_methods = planets["method"].value_counts().nlargest(5).index
filtered_planets = planets[planets["method"].isin(top_methods)]

sns.scatterplot(
    data=filtered_planets,
    x="distance",
    y="orbital_period",
    hue="method",
    style="method",
    alpha=0.8,
)

plt.xscale("log")
plt.yscale("log")

plt.title("Distance vs. Orbital Period by Discovery Method (Top 5)")
plt.xlabel("Distance (log scale)")
plt.ylabel("Orbital Period (days, log scale)")
plt.legend(title="Discovery Method", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

numeric_cols = planets.select_dtypes(include=["float64", "int64"])
correlation_matrix = numeric_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

plt.title("Correlation Matrix of Numeric Planet Features")
plt.tight_layout()
plt.show()
