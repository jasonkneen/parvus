import numpy as np  # Import numpy for random data generation
import pandas as pd  # Import pandas for saving as CSV

# Generate random data with 100 rows and 50 columns
data = np.random.rand(100, 50)

# Save the data as a CSV file
pd.DataFrame(data).to_csv("sample_data.csv", index=False)