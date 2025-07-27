import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load the dataset
try:
    df = pd.read_csv("depth_features_dataset.csv")
except FileNotFoundError:
    print("Error: 'depth_features_dataset.csv' not found.")
    print("Please run the modified depth_dataset2.py script first to generate it.")
    exit()

# Extract the two columns for comparison
analytical_depth = df['analytical_depth']
target_distance = df['target_distance']

# Calculate the Mean Absolute Error of the analytical formula alone
mae = mean_absolute_error(target_distance, analytical_depth)
print(f"MAE between 'analytical_depth' and 'target_distance': {mae:.4f}")

# Plot the comparison
plt.figure(figsize=(10, 6))
plt.scatter(target_distance, analytical_depth, alpha=0.6, label='Analytical vs. True')
plt.plot([target_distance.min(), target_distance.max()], 
         [target_distance.min(), target_distance.max()], 
         'r--', lw=2, label='Perfect Prediction')

plt.title('Analytical Depth vs. True Distance')
plt.xlabel('True Distance')
plt.ylabel('Analytical Depth Prediction')
plt.grid(True)
plt.legend()
plt.show()