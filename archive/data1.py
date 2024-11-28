import numpy as np
np.random.seed(42)
sample_data = np.random.rand(100, 50)  # 100 rows, 50 features
np.save("sample_data.npy", sample_data)