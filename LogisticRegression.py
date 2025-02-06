import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr, epochs, reg_lambda=0):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda # L2 Regularization

    def compute_loss(self, Y_hat):
        """
        Compute the binary cross-entropy loss.
        """
        # Clip Y_hat values to avoid taking the log of 0
        Y_hat = np.clip(Y_hat, 1e-10, 1 - 1e-10)
        loss = - (1 / self.m) * np.sum(
            self.Y * np.log(Y_hat) + (1 - self.Y) * np.log(1 - Y_hat)
        )
        if self.reg_lambda:
            loss += (self.reg_lambda / (2 * self.m)) * np.sum(self.w ** 2)
        return loss

    def fit(self, X, Y):
        # Convert X to a NumPy array of floats (if not already numeric)
        self.X = X.values.astype(np.float64)
        self.Y = Y
        self.m, self.n = self.X.shape
        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.epochs):
            self.update_weights()
            if i % 100 == 0:
                # Compute predictions using the sigmoid function
                Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))
                # Calculate the loss using the compute_loss function
                loss = self.compute_loss(Y_hat)
                print(f'epoch: {i}, loss: {loss}')

    def update_weights(self):
        # Compute the sigmoid function output 
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        # Compute gradients for weights and bias
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        # L2 Regularization
        if self.reg_lambda:
            dw += (self.reg_lambda / self.m) * self.w

        # Update the weights and bias
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def predict(self, X):
        # Compute probabilities and then convert them to binary outputs
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred

# Load the data
data = pd.read_csv("./datasets/diabetes2.csv")
print(data.head())

X = data.iloc[:, :-1]
Y = data.iloc[:, -1].to_numpy(dtype=float)

# training and testing split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the logistic regression model on the training data
LR = LogisticRegression(lr=0.0001, epochs=50000, reg_lambda=0.01)
LR.fit(X_train, y_train)

# Use the model to predict on the test data
predictions = LR.predict(X_test)

# Calculate accuracy on the test set
accuracy = np.mean(predictions == y_test)
print("Test Accuracy:", accuracy)
