# 如果你安装的是 Python 3.7 版的Anaconda，则需要声明新创建的环境使用 Python 3.6:
#conda create --name tf_gpu_env python=3.6 anaconda tensorflow-gpu
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成正弦波序列数据
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, num_samples, num_samples)
    y = np.sin(x)
    data = []
    for i in range(len(y) - seq_length):
        data.append(y[i:i + seq_length + 1])
    return np.array(data)

# 参数
seq_length = 50  # 序列长度
num_samples = 1000  # 总样本数

# 生成数据
data = generate_sine_wave(seq_length, num_samples)
X = data[:, :-1]  # 输入序列
Y = data[:, -1]   # 目标值（下一个值）

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 添加额外的维度以匹配 RNN 输入要求
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# 构建 RNN 模型
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型
test_loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss}')

# 预测
predictions = model.predict(X_test)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(Y_test, label='True Values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.title('RNN Predictions vs True Values')
plt.show()
