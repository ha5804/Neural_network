import matplotlib.pyplot as plt
import numpy as np

class MyData:
    def __init__(self):
        self.dir_data  = './data/'
        
    def get_data(self, split: int=1):
        file_data   = f'{self.dir_data}/data{split}.npy'
        data        = np.load(file_data) 
        return data
    
class MyPlot:
    def __init__(self):
        self.fig_size =(8,8)
    
    def plot_data(self, data):
        fig = plt.figure(figsize = self.fig_size)
        
        for i in range(25):
            im = data[i]
            ax = fig.add_subplot(5, 5, i+1)
            ax.imshow(im, cmap='gray')
            ax.axis('off')
            ax.axis('equal')
        plt.tight_layout()
        plt.show()
        
    
class activation:
    def __init__(self):
        pass

    def relu(self, x):
        return np.maximum(0, x)
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return e_x / np.sum(e_x, axis = 1, keepdims = True)
    
    def cross_entropy(self, y_hat, label):
        n = y_hat.shape[0]
        log_likelihood = -np.log(y_hat[range(n), label])
        return np.sum(log_likelihood) / n
    
import numpy as np

# Activation + Loss
class Activation:
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def cross_entropy(self, y_hat, label):
        label = label.reshape(-1)
        n = y_hat.shape[0]
        log_likelihood = -np.log(y_hat[range(n), label])
        return np.sum(log_likelihood) / n

# MLP 신경망
class MLP(Activation):
    def __init__(self, input_size, hidden_size, num_label=4):
        super().__init__()
        self.w = np.random.randn(input_size, hidden_size) * 0.01
        self.b = np.zeros((1, hidden_size))
        self.w_2 = np.random.randn(hidden_size, num_label) * 0.01
        self.b_2 = np.zeros((1, num_label))
        self.learning_rate = 0.1

    def forward(self, x):
        self.Z1 = x @ self.w + self.b
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.w_2 + self.b_2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, x, label):
        n = x.shape[0]
        dZ2 = self.A2.copy()
        dZ2[range(n), label.reshape(-1)] -= 1
        dZ2 /= n

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.w_2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = x.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.w -= self.learning_rate * dW1
        self.b -= self.learning_rate * db1
        self.w_2 -= self.learning_rate * dW2
        self.b_2 -= self.learning_rate * db2

    def train(self, x, label):
        y_hat = self.forward(x)
        loss = self.cross_entropy(y_hat, label)
        self.backward(x, label)
        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
