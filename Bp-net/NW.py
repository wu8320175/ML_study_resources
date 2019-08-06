import numpy as np
from pathlib import Path
import struct#用于数据类型的转换 bytes to str
import matplotlib.pyplot as plt
from tqdm import trange	# 替换range()可实现动态进度条，可忽略
import pickle

def Read_img_File(filepath):
    with open(filepath, 'rb') as f:
        struct.unpack('>4i', f.read(16))  # 4i代表4位整型，int：占4字节，4*8=32bit
        data = np.fromfile(f, dtype=np.uint8)
        return data.reshape(-1, 28 * 28)


def Read_label_File(filepath):
    with open(filepath, 'rb') as f:
        struct.unpack('>2i', f.read(8))
        data = np.fromfile(f, dtype=np.uint8)
        return data


def loadDataSets():
    train_img_path = 'handwrite/train-images.idx3-ubyte'
    train_label_path = 'handwrite/train-labels.idx1-ubyte'
    test_img_path = 'handwrite/t10k-images.idx3-ubyte'
    test_label_path = 'handwrite/t10k-labels.idx1-ubyte'
    train_img = Read_img_File(train_img_path)
    train_label = Read_label_File(train_label_path)
    test_img = Read_img_File(test_img_path)
    test_label = Read_label_File(test_label_path)
    return train_img, train_label, test_img, test_label


def sigmoid(x):  # 激活函数采用Sigmoid
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):  # Sigmoid的导数
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:  # 神经网络
    def __init__(self, layers):  # layers为神经元个数列表

        self.activation = sigmoid  # 激活函数
        self.activation_deriv = sigmoid_derivative  # 激活函数导数

        self.weights = []  # 权重列表
        self.bias = []  # 偏置列表

        # 初始化各层的参数
        for i in range(1, len(layers)):  # 正态分布初始化
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def fit(self, x, y, learning_rate=0.2, epochs=3):  # 反向传播算法
        x = np.atleast_2d(x)
        n = len(y)  # 样本数
        p = max(n, epochs)  # 样本过少时根据epochs减半学习率
        y = np.array(y)

        for k in trange(epochs * n):  # 带进度条的训练过程
            if (k + 1) % p == 0:
                learning_rate *= 0.5  # 每训练完一代样本减半学习率

            #取第k个样本进行训练
            a = [x[k % n]]  # 保存各层激活值的列表

            # 正向传播开始
            for lay in range(len(self.weights)):
                #a=g(a*theta+b)，生成每一层的激活项
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))

            # 反向传播开始
            label = np.zeros(a[-1].shape)
            label[y[k % n]] = 1
            # 根据标签，生成向量：[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

            #最后一层的误差 Y-Y_predict
            error = label - a[-1]  # 误差值
            deltas = [error * self.activation_deriv(a[-1])]  # 保存各层误差值的列表
            # err=err*g'

            #计算其它层的误差
            layer_num = len(a) - 2  # 导数第二层开始
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))  # 误差的反向传播，err(l)=theta(l).T*err(L+1).*g'[l]
            deltas.reverse()
            #deltas 生成theta的每一层的误差

            for i in range(len(self.weights)):  # 正向更新每一层权值
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                #w=w+alpha*d_theta
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]

    def predict(self, x):  # 预测
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):  # 正向传播
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a / sum(a))  # 改为百分比显示
        i = a.index(max(a))  # 预测值
        per = []  # 各类的置信程度
        for num in a:
            per.append(str(round(num, 2)) + '%')
        return i, per




def train(x,y,x_t,y_t):
    len_train = len(y)
    len_test = len(y_t)
    print('训练集大小%d，测试集大小%d' % (len_train, len_test))
    x = np.array(x)
    y = np.array(y)
    nn = NeuralNetwork([784, 784, 10])	# 神经网络各层神经元个数
    nn.fit(x, y)
    file = open('NN.txt', 'wb')
    pickle.dump(nn, file)
    count = 0
    for i in range(len_test):
        p, _ = nn.predict(x_t[i])
        if p == y_t[i]:
            count += 1
    print('模型识别正确率：', count/len_test)


def mini_test():	# 小型测试，验证神经网络能正常运行
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 2, 3]
    nn = NeuralNetwork([2, 4, 16, 4])
    nn.fit(x, y, epochs=10000)
    for i in x:
        print(nn.predict(i))


#mini_test()
train_img, train_label, test_img, test_label = loadDataSets()

x=train_img[0:2000]
y=train_label[0:2000]
x_t=test_img[0:1000]
y_t=test_label[0:1000]

train(x,y,x_t,y_t)
