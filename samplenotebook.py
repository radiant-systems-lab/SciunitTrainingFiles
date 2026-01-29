import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)

print("Libraries imported successfully")

# Create a simple dataset
x = np.arange(0, 10, 1)
y = np.sin(x)

df = pd.DataFrame({
    "x": x,
    "sin_x": y
})

print(df.head())

# Compute some statistics
mean_value = np.mean(y)
std_value = np.std(y)

print("Mean of sin(x):", mean_value)
print("Std of sin(x):", std_value)

# Save results to CSV
output_csv = "results.csv"
df.to_csv(output_csv, index=False)

print(f"Saved data to {output_csv}")

plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o')
plt.title("sin(x) vs x")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)

plot_file = "sin_plot.png"
plt.savefig(plot_file)
plt.close()

print(f"Plot saved to {plot_file}")

summary = {
    "mean_sin_x": mean_value,
    "std_sin_x": std_value
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("summary.csv", index=False)

print("Summary saved to summary.csv")
