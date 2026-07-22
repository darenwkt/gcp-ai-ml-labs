import numpy as np
import pandas as pd
import os

def generate_datasets():
    os.makedirs("data", exist_ok=True)
    
    np.random.seed(42)
    
    # Generate Normal training data: 2-dimensional gaussian distribution
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    training_data = np.random.multivariate_normal(mean, cov, 1000)
    
    df_train = pd.DataFrame(training_data, columns=["feature1", "feature2"])
    train_path = "data/training_data.csv"
    df_train.to_csv(train_path, index=False)
    print(f"Generated training data at {train_path}")

    # Generate Skewed Serving data (shifting the mean and covariance to simulate skew)
    skewed_mean = [1.5, -1.0]
    skewed_cov = [[2.0, -0.2], [-0.2, 1.5]]
    serving_data = np.random.multivariate_normal(skewed_mean, skewed_cov, 500)
    
    df_serve = pd.DataFrame(serving_data, columns=["feature1", "feature2"])
    serve_path = "data/serving_data_skewed.csv"
    df_serve.to_csv(serve_path, index=False)
    print(f"Generated skewed serving data at {serve_path}")

if __name__ == "__main__":
    generate_datasets()
