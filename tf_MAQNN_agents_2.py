#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   "
# -implement by Shurun Wang

import numpy as np
from tf_MAQNN_tools import dict_argmax, AttentionGate
import os
from tensorflow.keras.layers import Input, Activation, Conv2D, BatchNormalization, \
    MaxPooling2D, AveragePooling2D, Concatenate, Add, Flatten, Dense, Permute, Lambda
from tensorflow.keras import initializers, regularizers, Model


class CovBlockAgent:
    def __init__(self, T_cov_block, q_lr=0.01, gamma=1.0):  # T is the total time steps
        self.T_cov_block = T_cov_block
        self.q_lr = q_lr  # the learning rate for Bellman equation
        self.gamma = gamma
        self.Cov_Q_table = self.initiate_cov_q_table()
        self.Cov_Q_table_new = self.initiate_cov_q_table()

    def initiate_cov_q_table(self):
        cov_block_q_table = {}
        """
        use dict to store the Q_table (Cov Block)
        Type == 1 : PCC_conv
        Type == 2 : max_pooling
        Type == 3 : Average_pooling
        Type == 4 : Add
        Type == 5 : Concat
        Type == 6 : Terminal
        """
        cov_block_q_table['0_input'] = {}
        for index in range(1, self.T_cov_block + 1):  # the right side is not included
            # initiate states
            for TYPE in range(1, 7):  # there are 6 kinds of Type
                if TYPE == 1:  # convolution
                    for kernel in (1, 3, 5):
                        for pred1 in range(0, index):
                            cov_block_q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, kernel, pred1, 0)] = {}
                elif TYPE in (2, 3):
                    for kernel in (2, 4):
                        for pred1 in range(0, index):
                            cov_block_q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, kernel, pred1, 0)] = {}
                elif TYPE in (4, 5):
                    if index < 2:  # add and concat operation need at least two inputs(input layer can be included)
                        continue
                    for pred1 in range(0, index):
                        for pred2 in range(0, index):
                            if pred1 == pred2:
                                continue
                            cov_block_q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, 0, pred1, pred2)] = {}
                elif TYPE == 6:
                    if index == 1:  # output = input, meaningless
                        continue
                    cov_block_q_table["{0}_{1}_{2}_{3}_{4}".format(
                        index, TYPE, 0, 0, 0)] = {}

        # initiate actions
        # original reward is 0.5 as random guessing accuracy
        for state in cov_block_q_table.keys():
            index = int(state.split('_')[0]) + 1  # the state's action can chose the state itself as predecessor
            for TYPE in range(1, 7):  # there are 6 kinds of Type
                if TYPE == 1:  # convolution
                    for kernel in (1, 3, 5):
                        for pred1 in range(0, index):
                            cov_block_q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, kernel, pred1, 0)] = 0.5
                elif TYPE in (2, 3):
                    for kernel in (2, 4):
                        for pred1 in range(0, index):
                            cov_block_q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, kernel, pred1, 0)] = 0.5
                elif TYPE in (4, 5):
                    if state == '0_input':  # add and concat operation need at least two inputs
                        continue
                    for pred1 in range(0, index):
                        for pred2 in range(0, index):
                            if pred1 == pred2:
                                continue
                            cov_block_q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, 0, pred1, pred2)] = 0.5
                elif TYPE == 6:
                    if state == '0_input':  # output = input, meaningless
                        continue
                    cov_block_q_table[state]["{0}_{1}_{2}_{3}".format(
                        TYPE, 0, 0, 0)] = 0.5
        return cov_block_q_table

    def sample_new_cov_block(self, epsilon, update_mode='Q'):
        """
        based on the algorithm posted in the metaQnn paper
        """
        if update_mode == 'weighted_double_Q':
            S = ['0_input']
            U = []
            index = 1
            while index <= self.T_cov_block:
                q_action_values, q_action_values_new = [], []
                for u in list(self.Cov_Q_table[S[-1]].keys()):
                    q_action_values.append(self.Cov_Q_table[S[-1]][u])
                for u in list(self.Cov_Q_table_new[S[-1]].keys()):
                    q_action_values_new.append(self.Cov_Q_table_new[S[-1]][u])
                q_sum = np.sum([q_action_values, q_action_values_new], axis=0)

                if np.random.uniform(0, 1) > epsilon:  # exploitation
                    keys = list(self.Cov_Q_table[S[-1]].keys())
                    u = keys[int(np.random.choice(np.reshape(np.where(q_sum == np.max(q_sum)), -1), 1))]
                    new_state = str(index) + '_' + u
                else:  # exploration
                    u = np.random.choice(list(self.Cov_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)                                                 
                if u != '6_0_0_0':  # u != terminate
                    S.append(new_state)
                else:
                    return S, U
                index += 1
            U.append('6_0_0_0')

        else:
            # initialize S->state sequence;U->action sequence
            S = ['0_input']
            U = []
            index = 1
            # not the terminate layer and not surpass the max index(can be infinite)
            while index <= self.T_cov_block:
                a = np.random.uniform(0, 1)
                if a > epsilon:  # exploitation
                    u = dict_argmax(self.Cov_Q_table[S[-1]])
                    new_state = str(index) + '_' + u
                else:  # exploration
                    u = np.random.choice(list(self.Cov_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)
                if u != '6_0_0_0':  # u != terminate
                    S.append(new_state)
                else:
                    return S, U
                index += 1
            U.append('6_0_0_0')
        return S, U

    def update_cov_q_values(self, S, U, accuracy, gamma=1.0, update_mode='Q'):
        """
        based on the algorithm posted in the metaQnn paper
        :param gamma: the discount factor which measures the importance of future rewards
        :param S: state sequence
        :param U: action sequence
        :param accuracy: the model accuracy on the validation set
        :return: None
        """
        self.Cov_Q_table[S[-1]][U[-1]] = (1 - self.q_lr) * self.Cov_Q_table[S[-1]][U[-1]] \
                                       + self.q_lr * accuracy
        if update_mode == 'weighted_double_Q':
            self.Cov_Q_table_new[S[-1]][U[-1]] = (1 - self.q_lr) * self.Cov_Q_table_new[S[-1]][U[-1]] \
                                             + self.q_lr * accuracy

        rt = accuracy/len(S) 

        if update_mode == 'Q':
            # find the max action reward for the next step
            i = len(S) - 2
            while i >= 0:
                max_action_reward = 0
                for action in list(self.Cov_Q_table[S[i + 1]].keys()):
                    if self.Cov_Q_table[S[i + 1]][action] > max_action_reward:
                        max_action_reward = self.Cov_Q_table[S[i + 1]][action]

                self.Cov_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Cov_Q_table[S[i]][U[i]] \
                                           + self.q_lr * (rt + gamma * max_action_reward)
                i -= 1

        elif update_mode == 'expected_s':
            i = len(S) - 2
            while i >= 0:
                sum_action_reward = 0
                count = 0
                for action in list(self.Cov_Q_table[S[i + 1]].keys()):
                    sum_action_reward += self.Cov_Q_table[S[i + 1]][action]
                    count += 1
                expected_action_reward = sum_action_reward / count
                self.Cov_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Cov_Q_table[S[i]][U[i]] \
                                                + self.q_lr * (accuracy + gamma * expected_action_reward)
                i -= 1

        elif update_mode == 'weighted_double_Q':
            i = len(S) - 2
            while i >= 0:
                if np.random.random() < 0.5:
                # update q1
                    a_max = max(self.Cov_Q_table[S[i + 1]], key=self.Cov_Q_table[S[i + 1]].get)
                    a_low = min(self.Cov_Q_table[S[i + 1]], key=self.Cov_Q_table[S[i + 1]].get)
                    betta_u = abs(self.Cov_Q_table_new[S[i + 1]][a_max]-self.Cov_Q_table_new[S[i + 1]][a_low])\
                              /1+abs(self.Cov_Q_table_new[S[i + 1]][a_max]-self.Cov_Q_table_new[S[i + 1]][a_low])
                    delta_u = rt + gamma*(betta_u*self.Cov_Q_table[S[i + 1]][a_max]+
                                        (1-betta_u)*self.Cov_Q_table_new[S[i + 1]][a_max])-self.Cov_Q_table[S[i]][U[i]]
                    self.Cov_Q_table[S[i]][U[i]] = self.Cov_Q_table[S[i]][U[i]] + self.q_lr * delta_u
                else:
                # update q2
                    a_max = max(self.Cov_Q_table_new[S[i + 1]], key=self.Cov_Q_table_new[S[i + 1]].get)
                    a_low = min(self.Cov_Q_table_new[S[i + 1]], key=self.Cov_Q_table_new[S[i + 1]].get)
                    betta_v = abs(self.Cov_Q_table[S[i + 1]][a_max]-self.Cov_Q_table[S[i + 1]][a_low])\
                              /1+abs(self.Cov_Q_table[S[i + 1]][a_max]-self.Cov_Q_table[S[i + 1]][a_low])
                    delta_v = rt + gamma*(betta_v*self.Cov_Q_table_new[S[i + 1]][a_max]+
                                        (1-betta_v)*self.Cov_Q_table[S[i + 1]][a_max])-self.Cov_Q_table_new[S[i]][U[i]]
                    self.Cov_Q_table_new[S[i]][U[i]] = self.Cov_Q_table_new[S[i]][U[i]] + self.q_lr * delta_v
                i -= 1

class AttBlockAgent:
    def __init__(self, q_lr=0.01, gamma=1.0):
        self.q_lr = q_lr  # the learning rate for Bellman equation
        self.gamma = gamma
        self.Att_Q_table = self.initiate_att_q_table()
        self.Att_Q_table_new = self.initiate_att_q_table()

    def initiate_att_q_table(self):
        att_q_table = {}
        att_q_table['0_input'] = {}
        for index in (1, 2, 3):
            for kernel in (3, 5, 7):
                att_q_table["{0}_{1}".format(index, kernel)] = {}

        for state in att_q_table.keys():
            for kernel in (3, 5, 7):
                att_q_table[state]["{0}".format(kernel)] = 0.5
                index = int(state.split('_')[0])
                if index == 3:
                    att_q_table[state]["end"] = 0.5
        return att_q_table

    def sample_new_att_block(self, epsilon, update_mode='Q'):
        if update_mode == 'weighted_double_Q':
            S = ['0_input']
            U = []
            index = 1
            while index <= 3:
                q_action_values, q_action_values_new = [], []
                for u in list(self.Att_Q_table[S[-1]].keys()):
                    q_action_values.append(self.Att_Q_table[S[-1]][u])
                for u in list(self.Att_Q_table_new[S[-1]].keys()):
                    q_action_values_new.append(self.Att_Q_table_new[S[-1]][u])
                q_sum = np.sum([q_action_values, q_action_values_new], axis=0)

                if np.random.uniform(0, 1) > epsilon:  # exploitation
                    keys = list(self.Att_Q_table[S[-1]].keys())
                    u = keys[int(np.random.choice(np.reshape(np.where(q_sum == np.max(q_sum)), -1), 1))]
                    new_state = str(index) + '_' + u
                else:  # exploration
                    u = np.random.choice(list(self.Att_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)
                S.append(new_state)
                index += 1
            U.append('end')
        else:
            index = 1
            # not the terminate layer and not surpass the max index(can be infinite)
            S = ['0_input']
            U = []
            while index <= 3:
                a = np.random.uniform(0, 1)
                if a > epsilon:  # exploration
                    u = dict_argmax(self.Att_Q_table[S[-1]])
                    new_state = str(index) + '_' + u
                else:  # exploitation
                    u = np.random.choice(list(self.Att_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)
                S.append(new_state)
                index += 1
            U.append('end')
        return S, U

    def update_att_q_values(self, S, U, accuracy, gamma=1.0, update_mode='Q'):
        self.Att_Q_table[S[-1]][U[-1]] = (1 - self.q_lr) * self.Att_Q_table[S[-1]][U[-1]]+ self.q_lr * accuracy

        if update_mode == 'weighted_double_Q':
            self.Att_Q_table_new[S[-1]][U[-1]] = (1 - self.q_lr) * self.Att_Q_table_new[S[-1]][U[-1]] \
                                             + self.q_lr * accuracy
        accuracy = accuracy/len(S) 
                                             
        
        if update_mode == 'Q':
            i = len(S) - 2
            while i >= 0:
                max_action_reward = 0
                for action in list(self.Att_Q_table[S[i + 1]].keys()):
                    if self.Att_Q_table[S[i + 1]][action] > max_action_reward:
                        max_action_reward = self.Att_Q_table[S[i + 1]][action]

                self.Att_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Att_Q_table[S[i]][U[i]] \
                                           + self.q_lr * (accuracy + gamma * max_action_reward)
                i -= 1

        elif update_mode == 'expected_s':
            i = len(S) - 1
            while i >= 0:
                sum_action_reward = 0
                count = 0
                for action in list(self.Att_Q_table[S[i + 1]].keys()):
                    sum_action_reward += self.Att_Q_table[S[i + 1]][action]
                    count += 1
                expected_action_reward = sum_action_reward / count
                self.Att_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Att_Q_table[S[i]][U[i]] \
                                                + self.q_lr * (accuracy + gamma * expected_action_reward)
                i -= 1

        elif update_mode == 'weighted_double_Q':
            i = len(S) - 2
            while i >= 0:
                if np.random.random() < 0.5:
                # update q1
                    a_max = max(self.Att_Q_table[S[i + 1]], key=self.Att_Q_table[S[i + 1]].get)
                    a_low = min(self.Att_Q_table[S[i + 1]], key=self.Att_Q_table[S[i + 1]].get)
                    betta_u = abs(self.Att_Q_table_new[S[i + 1]][a_max]-self.Att_Q_table_new[S[i + 1]][a_low])\
                              /1+abs(self.Att_Q_table_new[S[i + 1]][a_max]-self.Att_Q_table_new[S[i + 1]][a_low])
                    delta_u = accuracy + gamma*(betta_u*self.Att_Q_table[S[i + 1]][a_max]+
                                        (1-betta_u)*self.Att_Q_table_new[S[i + 1]][a_max])-self.Att_Q_table[S[i]][U[i]]
                    self.Att_Q_table[S[i]][U[i]] = self.Att_Q_table[S[i]][U[i]] + self.q_lr * delta_u
                else:
                # update q2
                    a_max = max(self.Att_Q_table_new[S[i + 1]], key=self.Att_Q_table_new[S[i + 1]].get)
                    a_low = min(self.Att_Q_table_new[S[i + 1]], key=self.Att_Q_table_new[S[i + 1]].get)
                    betta_v = abs(self.Att_Q_table[S[i + 1]][a_max]-self.Att_Q_table[S[i + 1]][a_low])\
                              /1+abs(self.Att_Q_table[S[i + 1]][a_max]-self.Att_Q_table[S[i + 1]][a_low])
                    delta_v = accuracy + gamma*(betta_v*self.Att_Q_table_new[S[i + 1]][a_max]+
                                        (1-betta_v)*self.Att_Q_table[S[i + 1]][a_max])-self.Att_Q_table_new[S[i]][U[i]]
                    self.Att_Q_table_new[S[i]][U[i]] = self.Att_Q_table_new[S[i]][U[i]] + self.q_lr * delta_v
                i -= 1

class NetAgent:
    """
    each operation is a dic item like:
    'Index_Type_Kernel size_Pred1_Pred2'
    """

    def __init__(self, T_Net=8, q_lr=0.01, gamma=1.0):
        self.T_Net = T_Net
        self.q_lr = q_lr  # the learning rate for Bellman equation
        self.gamma = gamma
        self.Net_Q_table = self.initiate_net_q_table()
        self.Net_Q_table_New = self.initiate_net_q_table()

    def initiate_net_q_table(self):
        net_q_table = {}
        """  
        use dict to store the Q_table 
        Type == 1 : integrated block cov+att
        Type == 2 : maxpool  kernel 2, stride 2
        Type == 3 : avgpool  kernel 2, stride 2
        Type == 4 : Add
        Type == 5 : Terminal
        """
        net_q_table['0_input'] = {}
        for index in range(1, self.T_Net + 1):  # the right side is not included
            # initiate states
            for TYPE in range(1, 6):  # there are 5 kinds of Type
                if TYPE == 1:  # integrated block
                    for channel in (16, 32, 64):
                        net_q_table["{0}_{1}_{2}_{3}_{4}".format(index, TYPE, channel, index-1, 0)] = {}
                elif TYPE in (2, 3):
                        net_q_table["{0}_{1}_{2}_{3}_{4}".format(index, TYPE, 0, index-1, 0)] = {}
                elif TYPE == 4:
                    if index < 2:  # add and concat operation need at least two inputs(input layer can be included)
                        continue
                    for pred2 in range(0, index):
                        if index-1 == pred2:
                            continue
                        net_q_table["{0}_{1}_{2}_{3}_{4}".format(
                            index, TYPE, 0, index-1, pred2)] = {}
                elif TYPE == 5:
                    if index == 1:  # output = input, meaningless
                        continue
                    net_q_table["{0}_{1}_{2}_{3}_{4}".format(
                        index, TYPE, 0, index-1, 0)] = {}

        # initiate actions
        # original reward is 0.5 as random guessing accuracy
        for state in net_q_table.keys():
            index = int(state.split('_')[0]) + 1  # the state's action can chose the state itself as predecessor
            for TYPE in range(1, 6):  # there are 6 kinds of Type
                if TYPE == 1:
                    for channel in (16, 32, 64):
                        net_q_table[state]["{0}_{1}_{2}_{3}".format(TYPE, channel, index-1, 0)] = 0.5
                elif TYPE in (2, 3):
                        net_q_table[state]["{0}_{1}_{2}_{3}".format(TYPE, 0, index-1, 0)] = 0.5
                elif TYPE == 4:
                    if state == '0_input':  # add and concat operation need at least two inputs
                        continue
                    for pred2 in range(0, index):
                        if index-1 == pred2:
                            continue
                        net_q_table[state]["{0}_{1}_{2}_{3}".format(TYPE, 0, index-1, pred2)] = 0.5
                elif TYPE == 5:
                    if state == '0_input':  # output = input, meaningless
                        continue
                    net_q_table[state]["{0}_{1}_{2}_{3}".format(TYPE, 0, index-1, 0)] = 0.5
        return net_q_table

    def sample_new_network(self, epsilon, update_mode='Q'):
        iblock_count = 0
        pool_count = 0

        if update_mode == 'weighted_double_Q':
            S = ['0_input']
            U = []
            index = 1
            while index <= self.T_Net:
                q_action_values, q_action_values_new = [], []
                for u in list(self.Net_Q_table[S[-1]].keys()):
                    q_action_values.append(self.Net_Q_table[S[-1]][u])
                for u in list(self.Net_Q_table_New[S[-1]].keys()):
                    q_action_values_new.append(self.Net_Q_table_New[S[-1]][u])
                q_sum = np.sum([q_action_values, q_action_values_new], axis=0)

                if np.random.uniform(0, 1) > epsilon:  # exploitation
                    keys = list(self.Net_Q_table[S[-1]].keys())
                    u = keys[int(np.random.choice(np.reshape(np.where(q_sum == np.max(q_sum)), -1), 1))]
                    new_state = str(index) + '_' + u
                else:  # exploration
                    u = np.random.choice(list(self.Net_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)
                if u != '5_0_'+str(index-1)+'_0':  # u != terminate
                    S.append(new_state)
                else:
                    return S, U
                index += 1
            U.append('5_0_'+str(index-1)+'_0')
        else:
            # initialize S->state sequence;U->action sequence
            S = ['0_input']
            U = []
            index = 1
            # not the terminate layer and not surpass the max index(can be infinite)
            while index <= self.T_Net:
                a = np.random.uniform(0, 1)
                if a > epsilon:  # exploration
                    u = dict_argmax(self.Net_Q_table[S[-1]])
                    new_state = str(index) + '_' + u
                else:  # exploitation
                    u = np.random.choice(list(self.Net_Q_table[S[-1]].keys()))
                    new_state = str(index) + '_' + u
                U.append(u)
                if u != '5_0_'+str(index-1)+'_0':  # u != terminate
                    S.append(new_state)
                else:
                    return S, U
                index += 1
            U.append('5_0_'+str(index-1)+'_0')

        for i in range(len(U)):
            if int(U[i].split('_')[0]) == 1:
                iblock_count += 1
            if int(U[i].split('_')[0]) == 2 or int(U[i].split('_')[0]) == 3:
                pool_count += 1
        if iblock_count > 5 or pool_count > 3:
            return self.sample_new_network(epsilon)
        return S, U

    def update_net_q_values(self, S, U, accuracy, gamma=1.0, update_mode='Q'):
        self.Net_Q_table[S[-1]][U[-1]] = (1 - self.q_lr) * self.Net_Q_table[S[-1]][U[-1]] \
                                         + self.q_lr * accuracy

        if update_mode == 'weighted_double_Q':
            self.Net_Q_table_New[S[-1]][U[-1]] = (1 - self.q_lr) * self.Net_Q_table_New[S[-1]][U[-1]] \
                                             + self.q_lr * accuracy
        rt = accuracy/len(S)

        # find the max action reward for the next step
        if update_mode == 'Q':
            i = len(S) - 2
            while i >= 0:
                max_action_reward = 0
                for action in list(self.Net_Q_table[S[i + 1]].keys()):
                    if self.Net_Q_table[S[i + 1]][action] > max_action_reward:
                        max_action_reward = self.Net_Q_table[S[i + 1]][action]
                self.Net_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Net_Q_table[S[i]][U[i]] \
                                               + self.q_lr * (rt + gamma * max_action_reward)
                i -= 1
        elif update_mode == 'expected_s':
            i = len(S) - 2
            while i >= 0:
                sum_action_reward = 0
                count = 0
                for action in list(self.Net_Q_table[S[i + 1]].keys()):
                    sum_action_reward += self.Net_Q_table[S[i + 1]][action]
                    count += 1
                expected_action_reward = sum_action_reward / count
                self.Net_Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Net_Q_table[S[i]][U[i]] \
                                                + self.q_lr * (rt + gamma * expected_action_reward)
                i -= 1

        elif update_mode == 'weighted_double_Q':
            i = len(S) - 2
            while i >= 0:
                if np.random.random() < 0.5:
                # update q1
                    a_max = max(self.Net_Q_table[S[i + 1]], key=self.Net_Q_table[S[i + 1]].get)
                    a_low = min(self.Net_Q_table[S[i + 1]], key=self.Net_Q_table[S[i + 1]].get)
                    betta_u = abs(self.Net_Q_table_New[S[i + 1]][a_max]-self.Net_Q_table_New[S[i + 1]][a_low])\
                              /1+abs(self.Net_Q_table_New[S[i + 1]][a_max]-self.Net_Q_table_New[S[i + 1]][a_low])
                    delta_u = rt + gamma*(betta_u*self.Net_Q_table[S[i + 1]][a_max]+
                                        (1-betta_u)*self.Net_Q_table_New[S[i + 1]][a_max])-self.Net_Q_table[S[i]][U[i]]
                    self.Net_Q_table[S[i]][U[i]] = self.Net_Q_table[S[i]][U[i]] + self.q_lr * delta_u
                else:
                # update q2
                    a_max = max(self.Net_Q_table_New[S[i + 1]], key=self.Net_Q_table_New[S[i + 1]].get)
                    a_low = min(self.Net_Q_table_New[S[i + 1]], key=self.Net_Q_table_New[S[i + 1]].get)
                    betta_v = abs(self.Net_Q_table[S[i + 1]][a_max]-self.Net_Q_table[S[i + 1]][a_low])\
                              /1+abs(self.Net_Q_table[S[i + 1]][a_max]-self.Net_Q_table[S[i + 1]][a_low])
                    delta_v = rt + gamma*(betta_v*self.Net_Q_table_New[S[i + 1]][a_max]+
                                        (1-betta_v)*self.Net_Q_table[S[i + 1]][a_max])-self.Net_Q_table_New[S[i]][U[i]]
                    self.Net_Q_table_New[S[i]][U[i]] = self.Net_Q_table_New[S[i]][U[i]] + self.q_lr * delta_v
                i -= 1

class searched_model:
    def __init__(self,T_cov_block, T_net,):
        self.T_cov_block = T_cov_block
        self.T_net = T_net

    def generate_cov_block(self, S_cov_block, current_filters, inputs):
        print("Cov:", S_cov_block)
        follow = []
        for i in range(0, self.T_cov_block + 1):
            follow.append(0)
        follow[0] = 1  # input layer must feed as input of at least one other layers
        # initiate a dict to store the output of each layer
        layer_outputs = {} # input is the data feautre

        for s in S_cov_block:
            if s.split('_')[0] == str(0):
                layer_outputs={'0': inputs}
                continue
            Index, Type, KernelSize, Pred1, Pred2 = s.split('_')
            KernelSize = int(KernelSize)
            follow[int(Pred1)] = 1
            follow[int(Pred2)] = 1

            if Type == "1": # conv
                x = Conv2D(filters=current_filters, kernel_size=(KernelSize, KernelSize), strides=(1, 1), padding='same',
                           kernel_initializer=initializers.he_normal())(layer_outputs[Pred1])
                x = BatchNormalization()(x)
                layer_outputs[Index] = Activation("relu")(x)

            elif Type == "2": # MaxPooling2D
                layer_outputs[Index] = MaxPooling2D(pool_size=(KernelSize, KernelSize), padding='same',
                                                    strides=(1, 1))(layer_outputs[Pred1])
            elif Type == "3": # AveragePooling2D
                layer_outputs[Index] = AveragePooling2D(pool_size=(KernelSize, KernelSize), padding='same',
                                                        strides=(1, 1))(layer_outputs[Pred1])
            elif Type == "4": # add
                shape_x1 = layer_outputs[Pred1].get_shape().as_list()
                shape_x2 = layer_outputs[Pred2].get_shape().as_list()
                if shape_x1[-1] > shape_x2[-1]:
                    currentFilters = shape_x2[-1]
                    layer_outputs[Pred1] = Conv2D(filters=currentFilters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                               kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(layer_outputs[Pred1])

                if shape_x1[-1] < shape_x2[-1]:
                    currentFilters = shape_x1[-1]
                    layer_outputs[Pred2] = Conv2D(filters=currentFilters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                               kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(layer_outputs[Pred2])

                layer_outputs[Index] = Add()([layer_outputs[Pred1], layer_outputs[Pred2]])
            elif Type == "5": # concat
                x = Concatenate()([layer_outputs[Pred1], layer_outputs[Pred2]])
                layer_outputs[Index] = Conv2D(filters=current_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                           kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(x)

        # some layers that never feed as other layers' inputs are concatenated together to generate the final output
        concat_layers = []
        concat_layers_string = ""  # use as name prefix
        for i in range(1, len(S_cov_block)):
            if follow[i] == 0:
                concat_layers_string += str(i) + '_'
                concat_layers.append(layer_outputs[str(i)])
        if len(concat_layers) == 1:
            return concat_layers[0]

        block_output = Concatenate()(concat_layers)
        block_output = Conv2D(filters=current_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                              kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(block_output)
        block_output = BatchNormalization()(block_output)
        block_output = Activation('relu')(block_output)
        return block_output

    def generate_att_block(self, S_Att_block, inputs):
        print("att:", S_Att_block)
        kernel_list = []
        for s in S_Att_block:
            if s.split('_')[0] == str(0):
                continue
            kernel_list.append(s.split('_')[1])
        hw = AttentionGate(kernel=kernel_list[0])
        cw = AttentionGate(kernel=kernel_list[1])
        hc = AttentionGate(kernel=kernel_list[2])

        x_perm1 = Permute((2, 1, 3))(inputs)
        x_out1 = cw.forward(x_perm1)
        x_out11 = Permute((2, 1, 3))(x_out1)

        x_perm2 = Permute((3, 2, 1))(inputs)
        x_out2 = hc.forward(x_perm2)
        x_out21 = Permute((3, 2, 1))(x_out2)

        x_out = hw.forward(inputs)
        att_out = Add()([x_out, x_out11, x_out21])
        att_out = Lambda((lambda x: x/3))(att_out)

        return att_out

    def generate_network(self, S_Cov_block, S_Att_block, S_Net, need_attention=True):
        print('Net:', S_Net)

        net_outputs = {}
        for s in S_Net:
            if s.split('_')[0] == str(0):
                net_outputs['0'] = Input(shape=(32, 32, 3))
                continue
            Index, Type, Channel, Pred1, Pred2 = s.split('_')
            Channel = int(Channel)

            if Type == "1":  # block
                net_outputs[Index] = self.generate_cov_block(S_Cov_block, Channel, net_outputs[Pred1])
                if need_attention == True:
                    net_outputs[Index] = self.generate_att_block(S_Att_block, net_outputs[Index])
            if Type == "2":  # max pool
                net_outputs[Index] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(net_outputs[Pred1])
            if Type == "3":  # avg pool
                net_outputs[Index] = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(net_outputs[Pred1])

            if Type == "4":  # add
                channel_1 = net_outputs[Pred1].get_shape().as_list()[-1]
                channel_2 = net_outputs[Pred2].get_shape().as_list()[-1]
                feature_1 = net_outputs[Pred1].get_shape().as_list()[-2]
                feature_2 = net_outputs[Pred2].get_shape().as_list()[-2]
                min_channel = min(channel_1, channel_2)
                min_feature = min(feature_1, feature_2)
                if channel_1 == min_channel and feature_1 == min_feature:
                    x1 = net_outputs[Pred1]
                else:
                    ratio = int(feature_1 / min_feature)
                    x1 = Conv2D(filters=min_channel, kernel_size=(1, 1), strides=(ratio, ratio),
                                kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(net_outputs[Pred1])
                if channel_2 == min_channel and feature_2 == min_feature:
                    x2 = net_outputs[Pred2]
                else:
                    ratio = int(feature_2 / min_feature)
                    x2 = Conv2D(filters=min_channel, kernel_size=(1, 1), strides=(ratio, ratio),
                                kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(net_outputs[Pred2])
                net_outputs[Index] = Add()([x1, x2])

        model_output = Flatten()(net_outputs[str(len(S_Net)-1)])
        model_output = Dense(12, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2())(model_output)
        model_output = Activation("softmax")(model_output)
        return Model(inputs=net_outputs['0'], outputs=model_output)


