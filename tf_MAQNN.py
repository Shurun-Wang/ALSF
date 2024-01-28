#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   "
# -implement by S.R. Wang

import time
from tf_MAQNN_agents_2 import CovBlockAgent, NetAgent, AttBlockAgent, searched_model
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras import optimizers
from tf_MAQNN_tools import rolling_reward, get_data, epsilon_greedy, count_first200_accuracy, count_last200_accuracy, count_total_accuracy
import random
import os
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split


def seed_tensorflow(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.random.set_seed(seed)


class MAQnn:
    def __init__(self,
                 name,
                 T_cov_block=10,
                 T_net=10,
                 sampleBlock_num=64,
                 batch_size=128 * 2,
                 epoch=3000,
                 update_mode='weighted_double_Q'):
        """
        updated rule :
        Q
        expected_s
        weighted_double_Q
        """
        self.epoch = epoch
        self.name = name
        self.update_mode = update_mode
        self.CovBlockAgent = CovBlockAgent(T_cov_block=T_cov_block)
        self.AttBlockAgent = AttBlockAgent()
        self.NetAgent = NetAgent(T_Net=T_net)
        self.searched_model = searched_model(T_cov_block, T_net)

        self.sampleBlock_num = sampleBlock_num
        self.batch_size = batch_size
        self.replay_memory = []
        self.sampled_network_memory = []
        self.T_cov_block = T_cov_block
        self.T_net = T_net

    def nas(self):
        x = np.load('data/X_train.npy')
        x_test = np.load('data/X_test.npy')
        y = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=123)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        train_dataset = train_dataset.shuffle(500).batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        TopThreeList = []
        rt_list = []

        strategy = tf.distribute.MirroredStrategy()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)
        test_dataset = test_dataset.with_options(options)

        for iteration_num in range(self.epoch):
            epsilon = epsilon_greedy(iteration_num, 0.95, 0.05, 3000)
            print('{0}th iteration'.format(iteration_num))
            S_cov_block, U_cov_block = self.CovBlockAgent.sample_new_cov_block(epsilon=epsilon, update_mode=self.update_mode)
            S_att_block, U_att_block = self.AttBlockAgent.sample_new_att_block(epsilon=epsilon, update_mode=self.update_mode)
            S_net, U_net = self.NetAgent.sample_new_network(epsilon=epsilon, update_mode=self.update_mode)
            S_cat_tuple = (S_cov_block, S_att_block, S_net)
            U_cat_tuple = (U_cov_block, U_att_block, U_net)

            tf.keras.backend.clear_session()
            with strategy.scope():
                network_model = self.searched_model.generate_network(S_Cov_block=S_cov_block, S_Att_block=S_att_block,
                                                                     S_Net=S_net)
                network_model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005),
                                      loss='categorical_crossentropy', metrics=['accuracy'])
            def lr_decay_schedule(epoch, lr):
                if epoch % 5 == 0 and epoch != 0:
                    lr *= 0.5
                return lr
            lr_decay = LearningRateScheduler(schedule=lr_decay_schedule, verbose=0)
            es = EarlyStopping(patience=4)
            mycalls = [lr_decay, es]
            history = network_model.fit(train_dataset, epochs=12, validation_data=val_dataset, verbose=0,
                                        workers=4, callbacks=mycalls)
            accuracy = history.history['val_accuracy'][-1]

            print(accuracy)
            rt_list.append(accuracy)
            self.replay_memory.append((S_cat_tuple, U_cat_tuple, accuracy))
            self.sampled_network_memory.append(S_cat_tuple)
            # Save the top three validation accuracy
            if len(TopThreeList) < 3:
                TopThreeList.append([network_model, S_cat_tuple, U_cat_tuple, accuracy])
            else:
                TopThreeList.append([network_model, S_cat_tuple, U_cat_tuple, accuracy])
                TopThreeList.sort(key=lambda ele: ele[3], reverse=True)
                TopThreeList.pop()

            # experience replay
            for memory in range(0, np.min((len(self.replay_memory), self.sampleBlock_num))):
                choiceIndex = np.random.choice(range(0, len(self.replay_memory)))
                s_sample, u_sample, accuracy_sample = self.replay_memory[choiceIndex]
                S_cov_block, S_att_block, S_net = s_sample[0], s_sample[1], s_sample[2]
                U_cov_block, U_att_block, U_net = u_sample[0], u_sample[1], u_sample[2]
                self.CovBlockAgent.update_cov_q_values(S=S_cov_block, U=U_cov_block, accuracy=accuracy_sample, update_mode=self.update_mode)
                self.AttBlockAgent.update_att_q_values(S=S_att_block, U=U_att_block, accuracy=accuracy_sample, update_mode=self.update_mode)
                self.NetAgent.update_net_q_values(S=S_net, U=U_net, accuracy=accuracy_sample, update_mode=self.update_mode)

            if iteration_num == 199:
                count_first200_accuracy(rt_list)

            if iteration_num == 2999:
                count_last200_accuracy(rt_list)
                count_total_accuracy(rt_list)
            if iteration_num % 199 == 0:
                rolling_reward(rt_list, self.name)
        np.save('rt_list'+self.name+'.npy', rt_list)
        rolling_reward(rt_list, self.name)
        for i in range(len(TopThreeList)):
            print('the accuracy for best model {} is %f'.format(i) % TopThreeList[i][3])
            print('the description for the best model: states:{0} actions:{1}'.
                  format(TopThreeList[i][1], TopThreeList[i][2]))


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    tf.keras.backend.clear_session()
    # tf.config.optimizer.set_jit(False)
    seed_tensorflow(67)
    warnings.filterwarnings("ignore")
    name = time.strftime("%Y%m%d-%H%M%S")
    timebegin = time.time()
    MAQnn = MAQnn(name=name)
    MAQnn.nas()
    print(time.time() - timebegin)
