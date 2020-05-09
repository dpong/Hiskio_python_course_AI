from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, GRU, Dropout, concatenate
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import os, pickle
from matplotlib import style
from api import Rest_api


class Timeseries_tf():
    def __init__(self):
        self.df = pd.DataFrame()
        self.past_history = 20      #訓練用的過去天數
        self.future_target = 7     #預測未來天數
        self.col = 5
        self.checkpoint_path = 'model_weights\weights'
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.check_index = self.checkpoint_path + '.index'
        self.model = self.build_model()
        self.epochs = 30
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.optimizer = tf.optimizers.RMSprop(learning_rate=0.0001, epsilon=0.000065)
        self.loss_function = tf.keras.losses.MSE
        self.bar = Progbar(self.epochs)

    def get_data(self):
        ra = Rest_api()
        df = ra.get_history_data("BTC-PERP", 100)
        self.df = pd.DataFrame(df,dtype=np.float)
        
    def _multivariate_data(self, dataset, target, start_index, end_index, single_step=False):
        #整理資料
        data = []
        labels = []
        start_index = start_index + self.past_history
        if end_index is None:
            end_index = len(dataset) - self.future_target
        for i in range(start_index, end_index):
            indices = range(i-self.past_history, i)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i+self.future_target])
            else:
                labels.append(target[i:i+self.future_target])
        return np.array(data), np.array(labels)

    def handle_data(self):
        TRAIN_SPLIT = int(len(self.df) * 0.8)   #80%來train
        features = self.df
        self.close_mean = features['close'].mean()    #標準化們
        self.close_std = features['close'].std()
        dataset = features.values
        dataset_mean = dataset.mean(axis=0)
        dataset_std = dataset.std(axis=0)
        dataset = (dataset -dataset_mean) / dataset_std
        #target是Close, 就是dataset的第0個
        x_train, y_train = self._multivariate_data(dataset, dataset[:,0], 0, TRAIN_SPLIT)
        x_val, y_val = self._multivariate_data(dataset, dataset[:,0], TRAIN_SPLIT, None)
        self.train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
        self.val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(10000).batch(128)


    def build_GRU_model(self):
        tf.keras.backend.set_floatx('float64')
        data_input = Input(shape=(self.past_history, self.col), name='input')
        gru1 = GRU(64, return_sequences=True)(data_input)
        drop1 = Dropout(0.2)(gru1)
        gru2 = GRU(64, return_sequences=False)(drop1)
        drop2 = Dropout(0.2)(gru2)
        d1 = Dense(self.future_target, activation='relu')(drop2)
        model = Model(inputs=data_input, outputs=d1)
        if os.path.exists(self.check_index):
            print('-'*52+'  Weights loaded!!'+'-'*52)
            model.load_weights(self.checkpoint_path)
        else:
            print('-'*53+'Create new model!!'+'-'*53)
        return model


    def build_model(self):
        tf.keras.backend.set_floatx('float64')
        data_input = Input(shape=(self.past_history, self.col), name='input')
        drop0 = Dropout(0.2)(data_input)
        con1 = Conv1D(25 , 3, padding='causal')(drop0)
        con1_norm = BatchNormalization()(con1)
        con1_norm_act = Activation('relu')(con1_norm)
        drop1 = Dropout(0.5)(con1_norm_act)
        con2 = Conv1D(25 , 3, padding='causal')(drop1)
        con2_norm = BatchNormalization()(con2)
        con2_norm_act = Activation('relu')(con2_norm)
        drop2 = Dropout(0.5)(con2_norm_act)
        pool_max = MaxPooling1D(pool_size=5, strides=1)(drop2)
        flat1 = Flatten()(pool_max)
        d1 = Dense(self.future_target, activation='relu')(flat1)
        model = Model(inputs=data_input, outputs=d1)
        if os.path.exists(self.check_index):
            print('-'*52+'  Weights loaded!!'+'-'*52)
            model.load_weights(self.checkpoint_path)
        else:
            print('-'*53+'Create new model!!'+'-'*53)
        return model
    
    # loss
    def _loss(self, model, x, y):
        y_ = self.model(x)
        return self.loss_function(y_true=y, y_pred=y_)

    # gradient
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(self.model, inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def training(self):
        for i in range(self.epochs):
            for x, y in self.train_data:
                loss_value, grads = self._grad(self.model, x, y)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables),get_or_create_global_step())
                self.epoch_loss_avg(loss_value)
            self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])
        self.model.save_weights(self.checkpoint_path, save_format='tf')

    def prediction_test(self):
        for x, y in self.val_data.take(10):
            raw_predict = self.model(x)
            predict = raw_predict.numpy() # * self.close_std + self.close_mean
            x = x.numpy()
            y = y.numpy()
            x_p = x[10][:,0] #* self.close_std + self.close_mean
            y_p = y[10] #* self.close_std + self.close_mean
            plot = v.show_plot(x_p, y_p, predict[0])
        

class Visualize():
    def __init__(self):
        mpl.rcParams['figure.figsize'] = (8, 6)
        mpl.rcParams['axes.grid'] = False
        style.use('ggplot')

    def show_plot(self, history, true_future, prediction):
        plt.figure(figsize=(12, 6))            
        num_in = Visualize.create_time_steps(len(history))
        num_out = len(true_future)
        plt.plot(num_in, np.array(history),c='b', label='History')
        plt.plot(np.arange(num_out), np.array(true_future), c='g',ls='--', label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out), np.array(prediction), c='r',ls='-.', label='Predicted Future')
        plt.title('Model')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.show()

    def create_time_steps(length):
        time_steps = []
        for i in range(-length, 0, 1):
          time_steps.append(i)
        return time_steps

if __name__=='__main__':
    t = Timeseries_tf()
    v = Visualize()
    t.get_data()
    t.handle_data()
    t.training()
    #t.prediction_test()



