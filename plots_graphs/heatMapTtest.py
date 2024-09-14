import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Replace the values in the matrix with your actual T test results
data = [[None, 0.2367, 0.2694, 0.2950, 0.2847, 0.1379],
        [0.1338, None, 0.1742, 0.2374, 0.2395, 0.0255],
        [0.0467, 0.2045, None, 0.2204, 0.2316, 0.0474],
        [0.0353, 0.0517, 0.0609, None, 0.2427, 0.0613],
        [0.0139, 0.0031, 0.0027, 0.0002, None, 0.1433],
        [0.0049, 0.0087, 0.0430, 0.0776, 0.0782, None]]

# Create a DataFrame from the data
columns = ["Test 12", "Test 6", "In Vivo 3", "In Vivo 6", "PreTra 1", "PreTra 2"]
df = pd.DataFrame(data, columns=columns, index=columns)

# Create a heatmap using seaborn
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".4f", linewidths=.5)
plt.title("T Test Results")
plt.show()