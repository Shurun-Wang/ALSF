#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   "    
# -implement by S.R. Wang
import numpy
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Concatenate, Lambda, Multiply
from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
from sklearn.model_selection import train_test_split
import math
from sklearn.manifold import TSNE

def count_first200_accuracy(rt_list):
    dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
    dist7, dist8, dist9, dist10 = 0, 0, 0, 0
    for i in range(len(rt_list)):
        if 0 <= rt_list[i] < 0.5:
            dist1 = dist1 + 1
        if 0.5 <= rt_list[i] < 0.55:
            dist2 = dist2 + 1
        if 0.55 <= rt_list[i] < 0.6:
            dist3 = dist3 + 1
        if 0.6 <= rt_list[i] < 0.65:
            dist4 = dist4 + 1
        if 0.65 <= rt_list[i] < 0.7:
            dist5 = dist5 + 1
        if 0.7 <= rt_list[i] < 0.75:
            dist6 = dist6 + 1
        if 0.75 <= rt_list[i] < 0.8:
            dist7 = dist7 + 1
        if 0.8 <= rt_list[i] < 0.85:
            dist8 = dist8 + 1
        if 0.85 <= rt_list[i] < 0.9:
            dist9 = dist9 + 1
        if 0.9 <= rt_list[i] < 1:
            dist10 = dist10 + 1
    print('first200: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
          'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
          .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))

def count_last200_accuracy(rt_list):
    dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
    dist7, dist8, dist9, dist10 = 0, 0, 0, 0
    rt_list = rt_list[2799:2999]
    for i in range(len(rt_list)):
        if 0 <= rt_list[i] < 0.5:
            dist1 = dist1 + 1
        if 0.5 <= rt_list[i] < 0.55:
            dist2 = dist2 + 1
        if 0.55 <= rt_list[i] < 0.6:
            dist3 = dist3 + 1
        if 0.6 <= rt_list[i] < 0.65:
            dist4 = dist4 + 1
        if 0.65 <= rt_list[i] < 0.7:
            dist5 = dist5 + 1
        if 0.7 <= rt_list[i] < 0.75:
            dist6 = dist6 + 1
        if 0.75 <= rt_list[i] < 0.8:
            dist7 = dist7 + 1
        if 0.8 <= rt_list[i] < 0.85:
            dist8 = dist8 + 1
        if 0.85 <= rt_list[i] < 0.9:
            dist9 = dist9 + 1
        if 0.9 <= rt_list[i] < 1:
            dist10 = dist10 + 1
    print('last200: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
          'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
          .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))

def count_total_accuracy(rt_list):
    dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
    dist7, dist8, dist9, dist10 = 0, 0, 0, 0
    for i in range(len(rt_list)):
        if 0 <= rt_list[i] < 0.5:
            dist1 = dist1 + 1
        if 0.5 <= rt_list[i] < 0.55:
            dist2 = dist2 + 1
        if 0.55 <= rt_list[i] < 0.6:
            dist3 = dist3 + 1
        if 0.6 <= rt_list[i] < 0.65:
            dist4 = dist4 + 1
        if 0.65 <= rt_list[i] < 0.7:
            dist5 = dist5 + 1
        if 0.7 <= rt_list[i] < 0.75:
            dist6 = dist6 + 1
        if 0.75 <= rt_list[i] < 0.8:
            dist7 = dist7 + 1
        if 0.8 <= rt_list[i] < 0.85:
            dist8 = dist8 + 1
        if 0.85 <= rt_list[i] < 0.9:
            dist9 = dist9 + 1
        if 0.9 <= rt_list[i] < 1:
            dist10 = dist10 + 1
    print('total: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
          'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
          .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))

def rolling_reward(rt_list, name=None):
    matplotlib.use('Agg')
    cumsum_vec = np.nancumsum(np.insert(rt_list, 0, 0))
    ma_vec = (cumsum_vec[50:] - cumsum_vec[:-50]) / 50
    plt.plot(ma_vec)
    plt.savefig(name+'.png')


def get_data():
    s1_emg = np.load('data/s1_emg.npy', allow_pickle=True)
    s2_emg = np.load('data/s2_emg.npy', allow_pickle=True)
    s3_emg = np.load('data/s3_emg.npy', allow_pickle=True)
    s4_emg = np.load('data/s4_emg.npy', allow_pickle=True)
    s5_emg = np.load('data/s5_emg.npy', allow_pickle=True)
    s6_emg = np.load('data/s6_emg.npy', allow_pickle=True)
    s7_emg = np.load('data/s7_emg.npy', allow_pickle=True)
    s8_emg = np.load('data/s8_emg.npy', allow_pickle=True)
    s9_emg = np.load('data/s9_emg.npy', allow_pickle=True)
    s10_emg = np.load('data/s10_emg.npy', allow_pickle=True)

    s1_label = np.load('data/s1_label.npy', allow_pickle=True)
    s2_label = np.load('data/s2_label.npy', allow_pickle=True)
    s3_label = np.load('data/s3_label.npy', allow_pickle=True)
    s4_label = np.load('data/s4_label.npy', allow_pickle=True)
    s5_label = np.load('data/s5_label.npy', allow_pickle=True)
    s6_label = np.load('data/s6_label.npy', allow_pickle=True)
    s7_label = np.load('data/s7_label.npy', allow_pickle=True)
    s8_label = np.load('data/s8_label.npy', allow_pickle=True)
    s9_label = np.load('data/s9_label.npy', allow_pickle=True)
    s10_label = np.load('data/s10_label.npy', allow_pickle=True)

    x = np.concatenate((s1_emg, s2_emg, s3_emg, s4_emg, s5_emg, s6_emg, s7_emg, s8_emg, s9_emg, s10_emg), axis=0)
    y = np.concatenate((s1_label, s2_label, s3_label, s4_label, s5_label, s6_label, s7_label, s8_label, s9_label, s10_label), axis=0)
    y = y.reshape((-1, 1))
    y = keras.utils.to_categorical(y, num_classes=12)
    x = x.astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    # implement data augmentation for cifar10


class AttentionGate:
    def __init__(self, kernel):
        self.kernel_size = int(kernel)

    def forward(self, x):
        maxpool = Lambda(lambda x: tf.reduce_max(x, axis=3))(x)
        maxpool = Lambda(lambda x: tf.expand_dims(x, axis=3))(maxpool)
        avgpool = Lambda(lambda x: tf.reduce_mean(x, axis=3))(x)
        avgpool = Lambda(lambda x: tf.expand_dims(x, axis=3))(avgpool)
        Z_pool = Concatenate()([maxpool, avgpool])

        x_out = Conv2D(filters=1, kernel_size=(self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer=initializers.he_normal())(Z_pool)
        x_out = BatchNormalization()(x_out)
        x_out = Activation("relu")(x_out)
        x_out = Activation("sigmoid")(x_out)
        x_out = Multiply()([x, x_out])
        return x_out


def epsilon_greedy(step, eps_start=0.95, eps_end=0.05, eps_decay=500):
    eps = eps_end + (eps_start - eps_end) * \
              math.exp(-1. * step / eps_decay)
    return eps


def dict_argmax(d):
    """
    find the max integer element's corresponding key value in the dict
    :param d: the dic object on which to perform argmax operation
    :return: the max integer element's corresponding key
    """
    assert isinstance(d, dict)
    max_value = 0
    max_key = list(d.keys())[0]
    for key in d.keys():
        if d[key] > max_value:
            max_value = d[key]
            max_key = key
    if max_value == 0:  # still 0, random chose
        max_key = np.random.choice(list(d.keys()))
    return max_key

def tsne():
    p_test_1 = np.load('p_test_1.npy', allow_pickle=True)
    p_test_2 = np.load('p_test_2.npy', allow_pickle=True)
    p_test_3 = np.load('p_test_3.npy', allow_pickle=True)
    p_test_4 = np.load('p_test_4.npy', allow_pickle=True)
    p_test_5 = np.load('p_test_5.npy', allow_pickle=True)
    p_test_6 = np.load('p_test_6.npy', allow_pickle=True)
    p_test_7 = np.load('p_test_7.npy', allow_pickle=True)
    p_test_8 = np.load('p_test_8.npy', allow_pickle=True)
    p_test_9 = np.load('p_test_9.npy', allow_pickle=True)
    p_test_10 = np.load('p_test_10.npy', allow_pickle=True)

    o_test_1 = np.load('o_test_1.npy', allow_pickle=True)
    o_test_2 = np.load('o_test_2.npy', allow_pickle=True)
    o_test_3 = np.load('o_test_3.npy', allow_pickle=True)
    o_test_4 = np.load('o_test_4.npy', allow_pickle=True)
    o_test_5 = np.load('o_test_5.npy', allow_pickle=True)
    o_test_6 = np.load('o_test_6.npy', allow_pickle=True)
    o_test_7 = np.load('o_test_7.npy', allow_pickle=True)
    o_test_8 = np.load('o_test_8.npy', allow_pickle=True)
    o_test_9 = np.load('o_test_9.npy', allow_pickle=True)
    o_test_10 = np.load('o_test_10.npy', allow_pickle=True)

    l_test_1 = np.load('l_test_1.npy', allow_pickle=True)
    l_test_2 = np.load('l_test_2.npy', allow_pickle=True)
    l_test_3 = np.load('l_test_3.npy', allow_pickle=True)
    l_test_4 = np.load('l_test_4.npy', allow_pickle=True)
    l_test_5 = np.load('l_test_5.npy', allow_pickle=True)
    l_test_6 = np.load('l_test_6.npy', allow_pickle=True)
    l_test_7 = np.load('l_test_7.npy', allow_pickle=True)
    l_test_8 = np.load('l_test_8.npy', allow_pickle=True)
    l_test_9 = np.load('l_test_9.npy', allow_pickle=True)
    l_test_10 = np.load('l_test_10.npy', allow_pickle=True)

    o = np.concatenate((o_test_1, o_test_2, o_test_3, o_test_4, o_test_5,
                        o_test_6, o_test_7, o_test_8, o_test_9, o_test_10), axis=0)
    p = np.concatenate((p_test_1, p_test_2, p_test_3, p_test_4, p_test_5,
                        p_test_6, p_test_7, p_test_8, p_test_9, p_test_10), axis=0)
    l = np.concatenate((l_test_1, l_test_2, l_test_3, l_test_4, l_test_5,
                        l_test_6, l_test_7, l_test_8, l_test_9, l_test_10), axis=0)

    n_components = 2
    # fig = plt.figure(figsize=(12, 12))
    ts = TSNE(n_components=n_components, init='pca', random_state=0)
    p = ts.fit_transform(p)
    o = ts.fit_transform(o)
    print(l_test_1.shape[0], l_test_2.shape[0], l_test_3.shape[0], l_test_4.shape[0], l_test_5.shape[0],
          l_test_6.shape[0], l_test_7.shape[0], l_test_8.shape[0], l_test_9.shape[0], l_test_10.shape[0])
    np.savetxt('tsne.csv', np.concatenate((o, p, l), axis=1))

if __name__ == '__main__':
    tsne()
