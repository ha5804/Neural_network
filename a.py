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
        return np.maximun(0, x)
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return e_x / np.sum(e_x, axis = 1, keepdims = True)
    
    def cross_entropy(self, pred, label):
        n = pred.shape[0]
        log_likelihood = -np.log(pred[range(n), label])
        return np.sum(log_likelihood) / n
    
class MLP(activation):
    def __init__(self, input_size, hidden_size, num_label = 4):
        self.w = np.zeros((input_size, hidden_size))
        self.b = np.zeros((1, hidden_size))
        self.w_2 = np.zeros((hidden_size, num_label))
        self.b_2 = np.zeros((1, num_label))
    
    def forward(self, x):
        y = self.w @ x + self.b
        A1 = self.relu(y)
        z = self.w_2 @ A1 + self.b_2 

    