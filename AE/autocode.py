# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   autocode.py
   Author:      crd
   date:        2018/4/25
-------------------------------------------------
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf


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
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
        初始化，可以尝试添加多个隐藏层
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为SoftPlus
        :param optimizer: 优化器，默认为Adam
        :param scale: 高斯噪声系数，默认为0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),  # 给x加上噪声
                                                     self.weights['w1']), self.weights['b1']))  # 加上噪声的x乘权重加上偏置
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']), self.weights['b2'])
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
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
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
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
        :param hidden: 自编码器隐含层的输出结果
        :return: 重建层将提取到的高阶特征复原的结果
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

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
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """
        获取隐含层的偏置系数b1
        :return: 隐含层的偏置系数b1
        """
        return self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
    """
    对训练、测试数据进行标准化的处理的函数。标准化即让数据变成0均值，且标准差为1的分布
    :param X_train: 训练样本
    :param X_test: 测试样本
    :return: 标准化后的训练集和测试集
    """
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)  # 训练样本的数量
    training_epochs = 20  # 最大的训练轮数
    batch_size = 128  # batch大小
    display_step = 1  # 每隔一轮就现实一次损失

    """
    创建一个AGN自编码器的实例
    """
    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=784,
                                                   n_hidden=200,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)
    """
    开始训练，在每一轮开始时我们将平均损失avg_cost设为0，并计算总共需要的batch数（样本总数/batch大小）
    这里使用的时不放回抽样，所以并不能保证每个样本都参与训练。然后在每个batch循环中，
    先使用get_random_block_from_data函数随机抽取一个block的数据，然后使用成员函数partial_fit训练这个batch的数据并计算当前的cost
    最后将当前的cost整合到avg_cost中。
    """
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    """
    对训练完的模型进行性能测试
    """
    print("Total cost :" + str(autoencoder.calc_total_cost(X_test)))
