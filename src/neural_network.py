import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_size=10, learning_rate=0.10, random_state=None):
        self.h = hidden_layer_size
        self.ln = learning_rate
        self.random_state = random_state

    def _init_params(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.W1 = np.random.rand(self.h, self.n_feat) - 0.5
        self.b1 = np.zeros((self.h, 1))

        self.W2 = np.random.rand(self.n_classes, self.h) - 0.5
        self.b2 = np.zeros((self.n_classes, 1))

    def _feedforward(self, W1, b1, W2, b2):
        def relu(x):
            return np.maximum(0, x)

        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        Z1 = W1.dot(self.X) + b1
        A1 = relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        return Z1, A1, Z2, A2

    def _backpropagation(self, Z1, A1, A2):
        def relu_deriv(x):
            return x > 0

        dZ2 = A2 - self.y
        dW2 = 1/self.m * dZ2.dot(A1.T)
        db2 = (1/self.m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * relu_deriv(Z1)
        dW1 = 1/self.m * dZ1.dot(self.X.T)
        db1 = (1/self.m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def _update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.ln * dW1
        self.b1 -= self.ln * db1
        self.W2 -= self.ln * dW2
        self.b2 -= self.ln * db2

    def fit(self, X_train, y_train, epochs=1000):
        def gradient_descent(epochs):
            for i in range(epochs):
                Z1, A1, Z2, A2 = self._feedforward(self.W1, self.b1, self.W2, self.b2)

                dW1, db1, dW2, db2 = self._backpropagation(Z1, A1, A2)

                self._update_params(dW1, db1, dW2, db2)

                if (i + 1) % 50 == 0:
                    predictions = np.argmax(A2, axis=0)
                    labels = np.argmax(self.y, axis=0)
                    accuracy = np.mean(predictions == labels)

                    print(
                        f"Época {i + 1}/{epochs} - Acurácia: {accuracy:.2%}"
                    )


        self.m, self.n_feat = X_train.shape
        self.n_classes = len(np.unique(y_train))

        self.X = X_train.T
        self.y = np.eye(self.n_classes, dtype=int)[y_train].T

        self._init_params()

        gradient_descent(epochs)
        return self

    def predict(self, X):
        X = X.T

        Z1 = self.W1.dot(X) + self.b1
        A1 = np.maximum(0, Z1)

        Z2 = self.W2.dot(A1) + self.b2
        exp_z = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
        A2 = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        return np.argmax(A2, axis=0)