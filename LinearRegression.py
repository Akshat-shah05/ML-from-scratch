import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# -------------------------------
# Define our Linear Regression Model
# -------------------------------
class LinearRegression:
    def __init__(self, lr=0.001, iters=1000):
        self.lr = lr
        self.epochs = iters
        self.w = None  # weights (will be a numpy array)
        self.b = 0     # bias (a scalar)
    
    # Computes the predictions: y_pred = X @ w + b
    def fw_b(self, X):
        return np.dot(X, self.w) + self.b

    # Computes the Mean Squared Error (MSE) and residuals
    def mean_squared_error_loss(self, X, Y):
        m = X.shape[0]
        predictions = self.fw_b(X)
        residuals = predictions - Y
        loss = (1 / (2 * m)) * np.sum(residuals ** 2)
        return loss, residuals
    
    # Predict function for external use
    def predict(self, X):
        return self.fw_b(X)


# -------------------------------
# Load and Prepare the Data
# -------------------------------
data = pd.read_csv("./datasets/Student_Performance.csv")

data.iloc[:, 2] = data.iloc[:, 2].map({"Yes": 1, "No": 0}).fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(data.iloc[:, :-1])  # Standardize features
Y = data.iloc[:, -1].to_numpy(dtype=float)

# Split the data (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# -------------------------------
# Set Up PCA for Visualization
# -------------------------------
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)

# -------------------------------
# Set Up the Real-Time Plot
# -------------------------------
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the original training data in the PCA space (scatter plot)
scatter = ax.scatter(X_train_pca, Y_train, color='blue', alpha=0.6, label='Data')

# Create an (initially empty) line for the regression line
line_plot, = ax.plot([], [], color='red', linewidth=2, label='Regression line')

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Target")
ax.legend()
ax.set_title("Epoch 0")

x_min = X_train_pca.min()
x_max = X_train_pca.max()
x_line = np.linspace(x_min, x_max, 100)


# -------------------------------
# Initialize and Train the Model with Real-Time Plot Updates
# -------------------------------
model = LinearRegression(lr=0.001, iters=7000)

m, n = X_train.shape
# Initialize weights as zeros (one weight per feature)
model.w = np.zeros(n, dtype=float)
model.b = 0

# How often to update the plot (in epochs)
plot_interval = 100

# Training loop with real-time visualization
for epoch in range(model.epochs):
    loss, residuals = model.mean_squared_error_loss(X_train, Y_train)
    
    # Compute gradients
    dw = (1 / m) * np.dot(X_train.T, residuals)
    db = (1 / m) * np.sum(residuals)
    
    # Update parameters using gradient descent
    model.w -= model.lr * dw
    model.b -= model.lr * db
    
    # Update the plot every 'plot_interval' iterations
    if epoch % plot_interval == 0:
        Y_pred = model.predict(X_train)
        
        slope, intercept = np.polyfit(X_train_pca.flatten(), Y_pred, 1)
        y_line = slope * x_line + intercept
        
        line_plot.set_data(x_line, y_line)
        ax.set_title(f"Epoch: {epoch}, Loss: {loss:.4f}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()
plt.show()


# -------------------------------
# Final Accuracy Evaluation
# -------------------------------
def compute_metrics(model, X, Y):
    predictions = model.predict(X)
    mse = np.mean((predictions - Y) ** 2)
    ss_total = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y - predictions) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return mse, r2

train_mse, train_r2 = compute_metrics(model, X_train, Y_train)
test_mse, test_r2 = compute_metrics(model, X_test, Y_test)

print("\nFinal Performance:")
print(f"Training MSE: {train_mse:.4f}, R-squared: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, R-squared: {test_r2:.4f}")
