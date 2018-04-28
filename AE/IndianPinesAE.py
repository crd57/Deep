# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time    : 2018/4/27 20:56
# @Author  : Crd
# @Email   : crd57@outlook.com
# @File    : IndianPinesAE.py
# @Software: PyCharm
-------------------------------------------------
"""
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as prep
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def xavier_init(fan_in, fan_out, constant=1):
    """
    定义一种参数初始化方法xavier
    :param fan_in:输入的节点数
    :param fan_out:输出的节点数
    :param constant:常数
    :return:均匀分布的Xaiver初始化器
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_out + fan_in))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden1, n_hidden2, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
        初始化，可以尝试添加多个隐藏层
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为SoftPlus
        :param optimizer: 优化器，默认为Adam
        """
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None,self.n_input])
        self.hidden1 = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.hidden2 = self.transfer(tf.add(tf.matmul(self.hidden1, self.weights['w2']), self.weights['b2']))
        self.reconstruction = tf.add(tf.matmul(self.hidden2,
                                               self.weights['w3']), self.weights['b3'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))  # 直接使用平方误差作为cost
        self.optimizer = optimizer.minimize(self.cost)  # 定义训练操作为优化器optimizer对损失self.cost进行优化
        init = tf.global_variables_initializer()  # 全局变量初始化
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        参数初始化函数
        :return: 初始化系数矩阵
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden1))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden1, self.n_hidden2], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_hidden2], dtype=tf.float32))
        all_weights['w3'] = tf.Variable(tf.zeros([self.n_hidden2, self.n_input], dtype=tf.float32))
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """
        定义计算损失cost及执行一步训练的函数partial_fit.
        :param X: 输入数据X，噪声系数scale
        :return:用一个batch数据进行训练，，返回当前的损失cost
        """
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """
        只求损失cost的函数，在对模型性能进行评测是可以用到。
        :param X:输入数据X，噪声系数scale
        :return: cost的值
        """
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale
                                                   })

    def transform(self, X):
        """
        提供一个接口获得抽象后的特征
        :param X:输入数据X，噪声系数scale
        :return: 自编码器隐含层的输出结果
        """
        return self.sess.run(self.hidden2, feed_dict={self.x: X,
                                                      self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
        :param hidden: 自编码器隐含层的输出结果
        :return: 重建层将提取到的高阶特征复原的结果
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b2'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden2: hidden})

    def reconstruct(self, X):
        """
        整体运行一遍复原过程，包括提取高阶特征和通过高阶特征福原数据，
        :param X: 原数据
        :return: 复原后的数据
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale
                                                             })

    def getWeights(self):
        """
        获取隐含层的权重w1
        :return: 隐含层的权重w1
        """
        return self.sess.run(self.weights['w2'])

    def getBiases(self):
        """
        获取隐含层的偏置系数b1
        :return: 隐含层的偏置系数b1
        """
        return self.sess.run(self.weights['b2'])


def standard_scale(X_train, X_test, Y_train, Y_test):
    """
    生成TensorFlow合适的样本。
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    """

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    x_train = X_train.reshape(-1, 200)
    x_test = X_test.reshape(-1, 200)
    y_train = OneHotEncoder().fit_transform(Y_train).todense()
    y_test = OneHotEncoder().fit_transform(Y_test).todense()
    return x_train, x_test, y_train, y_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def Pretreatment(XFileNames, YFileNames):
    data_corrected = sio.loadmat(XFileNames)
    data_gt = sio.loadmat(YFileNames)
    indian_pines_gt = data_gt['indian_pines_gt']
    indian_pines_corrected = data_corrected['indian_pines_corrected']

    data = np.empty((145 * 145, 200), dtype=float)
    label = np.empty((145 * 145, 1), dtype=int)
    i = 0
    for row in range(145):
        for col in range(145):
            data[i, :] = indian_pines_corrected[row, col, :]
            label[i] = indian_pines_gt[row, col]
            i += 1
    index = np.where(label != 0)
    x = np.empty((index[0].size, 200))
    i = 0
    for rows in index[0]:
        x[i, :] = data[rows, :]
        i += 1
    y = label[label != 0]
    y = y.reshape((y.size, 1))
    kf = KFold(n_splits=100)
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    x_train, x_test, y_train, y_test = standard_scale(X_train, X_test, y_train, y_test)
    return x_train, x_test, y_train, y_test,data,label, indian_pines_gt


if __name__ == "__main__":
    x_train, x_test, y_train, y_test,data,label = Pretreatment('Indian_pines_corrected.mat', 'Indian_pines_gt.mat')
    n_samples = x_train.shape[0]
    training_epochs = 1000
    batch_size = 500
    display_step = 1
    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=200,
                                                   n_hidden1=50,
                                                   n_hidden2=20,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(x_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Total cost :" + str(autoencoder.calc_total_cost(x_test)))
