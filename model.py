import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json, os

class ClimateModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y):
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y.reshape(n, -1))
        self.trained = True
        pred = self.model.predict(X_scaled)
        mse = np.mean((pred - y.reshape(n, -1))**2)
        mae = np.mean(np.abs(pred - y.reshape(n, -1)))
        metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}
        with open("training_metrics.json", "w") as f:
            json.dump(metrics, f)
        return metrics

    def predict(self, X):
        if not self.trained:
            return X[..., 0]
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        X_scaled = self.scaler.transform(X_flat)
        pred = self.model.predict(X_scaled)
        h, w = X.shape[1], X.shape[2]
        return pred.reshape(n, h, w)

if __name__ == "__main__":
    X = np.random.rand(50, 16, 15, 7).astype(np.float32)
    y = np.random.rand(50, 16, 15).astype(np.float32)
    model = ClimateModel()
    metrics = model.train(X, y)
    print("Training complete:", metrics)
    pred = model.predict(X[:3])
    print("Prediction shape:", pred.shape)