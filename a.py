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


