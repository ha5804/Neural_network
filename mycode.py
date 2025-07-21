import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import csv
from PIL import Image
import statistics

# ================================================
class MyResult:
    def __init__(self):
        self.name   = '태하영'
        self.id     = '01053151074'
        self.email  = 'zixpotf5803@cau.ac.kr'
        self.tel    = '010-5315-1074'        
        self.key    = []
        self.val    = {} 

    def add_result(self, key, val):
        self.key.append(key)
        self.val[key] = val

    def __len__(self):
        return len(self.key)

    def empty(self):
        self.key = []
        self.val = {}

    def save(self, filename='result.csv'):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            self.write_separator(wr)
            self.write_info(wr)
            self.write_separator(wr)
            self.write_result(wr)
           
    def write_separator(self, wr):
        wr.writerow(['******************************'])

    def write_info(self, wr):
        wr.writerow(['name', self.name])
        wr.writerow(['id', self.id])
        wr.writerow(['email', self.email])
        wr.writerow(['tel', self.tel])

    def write_result(self, wr):
        for i in range(len(self.key)):
            key = self.key[i]
            val = self.val[key]
            wr.writerow([key, val])

    def plot(self):
        self.plot_info()
        self.plot_result()

    def plot_info(self):
        print('******************************')
        print(f'name'.ljust(5), f': {self.name}')
        print(f'id'.ljust(5), f': {self.id}')
        print(f'email'.ljust(5), f': {self.email}')
        print(f'tel'.ljust(5), f': {self.tel}') 
        print('******************************')

    def plot_result(self):
        for i in range(len(self.key)):
            key = self.key[i]
            val = self.val[key]
            print(f'[{key}]'.ljust(5), f': {val}')


# ================================================
class MyPlot:
    def __init__(self):
        self.fig_size = (8, 8)

    def plot_data(self, data):
        fig = plt.figure(figsize=self.fig_size)
      
        for i in range(25):
            im = data[i]
            ax = fig.add_subplot(5, 5, i+1)
            ax.imshow(im, cmap='gray')
            ax.axis('off')
            ax.axis('equal')
        plt.tight_layout()
        plt.show()


    def plot_centroid(self, data, label, cluster):
        fig = plt.figure(figsize=self.fig_size)

        for i in cluster:
            index = (label == i)
            centroid = np.mean(data[index], axis=0)
            plt.subplot(2,2,i+1)
            plt.imshow(centroid, cmap='gray')
            plt.axis('off')
            plt.axis('equal')
            plt.title(f'cluster={i}') 
        plt.tight_layout()
        plt.show()


    def plot_curve(self, line1, ylabel1, xlabel, ylabel):
        fig = plt.figure(figsize=self.fig_size)
        plt.plot(line1, color='blue', label=ylabel1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ================================================
class MyData:
    def __init__(self):
        self.dir_data  = './data/'
        
    def get_data(self, split: int=1):
        file_data   = f'{self.dir_data}/data{split}.npy'
        data        = np.load(file_data) 
        return data

# ================================================
class MyModel:
    def __init__(self):
        self.cluster        = [] 
        self.num_cluster    = 0
        # =====================================================================
        self.num_iter       = 30
        # =====================================================================

    def set_cluster(self, cluster):
        self.cluster = cluster
        self.num_cluster = len(cluster)
   
    def get_num_cluster(self):
        return self.num_cluster
    
    def get_num_iter(self):
        return self.num_iter
  
    def compute_feature(self, data):
        feature_first_col = np.mean(data[: , 2, 2:-2], axis = (1)).reshape(-1, 1) 
        data_1 = data[:,2:10,2:10].reshape((4000,64))
        col_3 = np.mean(data[: , 6, 2:-2], axis = (1)).reshape(-1, 1) 
        col_4 = np.mean(data[: , 7 , 2:-2], axis = (1)).reshape(-1, 1)
        col_5 = np.mean(data[: , 5 , 2:-2], axis = (1)).reshape(-1, 1)
        col_6 = np.mean(data[: , 6 , 2:-2], axis = (1)).reshape(-1, 1)
        col_7 = np.mean(data[: , 7 , 2:-2], axis = (1)).reshape(-1, 1)
        col_8 = np.mean(data[: , 8 , 2:-2], axis = (1)).reshape(-1, 1)
        col_9 = np.mean(data[: , 9 , 2:-2], axis = (1)).reshape(-1, 1)
        col_10 = np.mean(data[:, 2:-2 , 2:-2], axis=(1,2)).reshape(-1,1)
        col_11 = np.var(data_1 , axis = (1)).reshape(-1,1)
        data_pad = np.pad(data, pad_width =((0,0), (1,1), (1,1)), mode = 'edge')
        df_dy2 = data_pad[:, 2:, 1:-1] - (2 * data_pad[:, 1:-1, 1:-1]) + data_pad[:, :-2, 1:-1]  
        df_dx2 = data_pad[:, 1:-1, 2:] - (2 * data_pad[:, 1:-1 , 1:-1]) + data_pad[:, 1:-1, :-2]
        lap = df_dx2 + df_dy2
        col_12 = np.mean(np.abs(lap), axis = (1,2)).reshape(-1, 1)
        col_13 = np.std(np.abs(lap), axis = (1, 2)).reshape(-1, 1) * 2
        col_14 = np.var(data[:, 2:-2, 2:-2], axis = (1, 2)).reshape(-1, 1) * 2
        feature_last_col= np.mean(data[: , 10 , 2:-2], axis = (1)).reshape(-1, 1)
        # ── 1. 중앙 10×10에서 x·y 방향 그래디언트 ───────────────────────────
# gx: 좌우 차분, gy: 상하 차분  → 둘 다 (4000, 10, 10)
        gx = data[:, 1:11, 2:12] - data[:, 1:11, 0:10]
        gy = data[:, 2:12, 1:11] - data[:, 0:10, 1:11]

# ── 2. 각 픽셀의 방향(angle) → 0‥7 bin 인덱스로 양자화 ─────────────
        angle   = np.mod(np.arctan2(gy, gx) + np.pi, 2*np.pi)      # 0~2π
        bin_idx = (angle / (np.pi / 4)).astype(np.int32)           # 0~7

# ── 3. 8-방향 히스토그램 계산 (정규화: 픽셀 수 = 100) ───────────────
        hog_hist = np.zeros((data.shape[0], 8), dtype=np.float32)  # (4000, 8)
        for b in range(8):
            hog_hist[:, b] = (bin_idx == b).sum(axis=(1, 2)) / 100.0
        feature = np.column_stack((feature_first_col,data_1, col_3 , col_4, col_5, col_6, col_7, col_8, col_9 ,col_10,col_11, col_12,col_13, col_14, hog_hist, feature_last_col))
        

        # =====================================================================
        return feature

    def feature_function_first(self, data_one_element):
        # =====================================================================
        f = np.mean(data_one_element[2, 2:-2])

        
        # =====================================================================
        return f 
    
    def feature_function_last(self, data_one_element):
        # =====================================================================
        f = np.mean(data_one_element[10, 2:-2])
        
        # =====================================================================
        return f 
    
    # =========================================================================
    # DO NOT MODIFY THIS FUNCTION
    # =========================================================================
    def compute_feature_first(self, data):
        num_data    = data.shape[0]
        feature     = np.zeros(num_data)
        for i in range(num_data):
            f = self.feature_function_first(data[i])
            feature[i] = f
        return feature
    
    # =========================================================================
    # DO NOT MODIFY THIS FUNCTION
    # =========================================================================
    def compute_feature_last(self, data):
        num_data    = data.shape[0]
        feature     = np.zeros(num_data)
        for i in range(num_data):
            f = self.feature_function_last(data[i])
            feature[i] = f
        return feature

    def compute_centroid(self, feature, label): 
        # =====================================================================
        centroid = []
        for i in range(4):
            num = feature[label == i]
            if len(num) == 0:
                centroid_i = feature[np.random.randint(0, feature.shape[0])]
            else:
                centroid_i = (1 / len(num)) * np.sum(feature[label == i], axis = 0)
            centroid.append(centroid_i)
        centroid = np.array(centroid)
        #print(centroid.shape)
        # =====================================================================
        return centroid
    
    def compute_distance(self, feature, centroid):
        # =====================================================================
        # distance = np.zeros((4000, 4))
        # for i in range(4000):
        #     for j in range(4):
        #         val = feature[i] - centroid[j]
        #         distance[i , j] = (np.sum(val ** 2)) ** (1 / 2)
        feature_reshape = feature.reshape(4000, 1, feature.shape[1])
        centroid_reshape = centroid.reshape(1, 4, feature.shape[1])
        distance = np.sum((feature_reshape - centroid_reshape) ** 2, axis = 2) ** (1/ 2)
                
        # =====================================================================
        return distance
    
    def compute_label(self, distance):
        # =====================================================================
        label = None 
        # =====================================================================
        label = np.argmin(distance, axis=1)
        return label
    
    def compute_loss(self, distance, label):
        # =====================================================================
        row = np.arange(4000)
        distance = distance[row, label]
        loss = (1 / len(label)) * np.sum(distance)
        # =====================================================================
        return loss

        
# ================================================
# DO NOT MODIFY THIS CLASS
# ================================================
class MyEval:
    def __init__(self):
        self.cluster                = []
        self.num_cluster            = 0
        self.num_data_per_cluster   = 0
        self.num_data               = 0

    def initialize(self, cluster=[0, 1, 2, 3], num_data_per_cluster=1000):
        self.cluster = cluster
        self.num_cluster = len(self.cluster)
        self.num_data_per_cluster = num_data_per_cluster 
        self.num_data = self.num_data_per_cluster * self.num_cluster
    
    def compute_accuracy(self, label):
        label = np.array(label)
        cluster_taken = []
        count_correct = 0 
        if set(label) == set(self.cluster):
            for (i, k) in enumerate(self.cluster):
                label_cluster = label[i * self.num_data_per_cluster : (i+1) * self.num_data_per_cluster]
                bin_count = np.bincount(label_cluster)
                max_count = np.max(bin_count)
                max_index = np.argmax(bin_count)
                if max_index not in cluster_taken:
                    count_correct += max_count
                    cluster_taken.append(max_index)
        accuracy = count_correct / self.num_data
        return accuracy