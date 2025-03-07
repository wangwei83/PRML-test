from sklearn import preprocessing
import numpy as np

x = np.array([[0,0],
             [0,0],
             [100,1],
             [1,1]])
print(x)

X_mean=x.mean(axis=0)
X_std=x.std(axis=0)
print(X_mean)   
print(X_std)
X1=(x-X_mean)/X_std
print(X1)

print("=====================================")

X_scale=preprocessing.scale(x)
print(X_scale)


from sklearn import datasets
iris = datasets.load_iris()
X_scale = preprocessing.scale(iris.data)
print(X_scale)
print("len(X_scale=")
print(len(X_scale))
# 打印特征数据的形状
print(iris.data.shape)


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

data = [[-1, 2], 
        [-0.5, 6], 
        [0, 10], 
        [1, 18]]

scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.transform(data))


from sklearn.preprocessing import MinMaxScaler
data = iris.data
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.transform(data))