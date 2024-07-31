import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "W_regularizer": tf.keras.regularizers.serialize(self.W_regularizer),
            "b_regularizer": tf.keras.regularizers.serialize(self.b_regularizer),
            "W_constraint": tf.keras.constraints.serialize(self.W_constraint),
            "b_constraint": tf.keras.constraints.serialize(self.b_constraint),
            "bias": self.bias,
            "step_dim": self.step_dim,
            "features_dim": self.features_dim
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight(shape=(self.features_dim,),
                                 initializer='glorot_uniform',
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(self.step_dim,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        e = K.reshape(K.dot(K.reshape(x, (-1, self.features_dim)), K.reshape(self.W, (self.features_dim, 1))), (-1, self.step_dim))
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a, axis=-1)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
    

def cnn1d_module(inputs, cnn, fc):
    x = tf.keras.layers.Conv1D(cnn, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(fc, activation='relu')(x)
    return x


def gru_module(inputs, gru, fc):
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru, return_sequences=True))(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(gru/2), return_sequences=True))(x)
    x = Attention(41)(x)
    x = tf.keras.layers.Dense(fc, activation='relu')(x)
    return x


def cnn1d_gru_model(list_shape_input1, list_shape_input2, params, fuse=''):
    cnn = params['cnn']
    gru = params['gru']
    fc1 = params['fc1']
    fc2 = params['fc2']
    fc3 = params['fc3']
    inputs1_ls = []
    for i in range(len(list_shape_input1)):
        inputs1 = tf.keras.layers.Input(shape=list_shape_input1[i])
        inputs1_ls.append(inputs1)
        if i == 0:
            x1 = cnn1d_module(inputs1, cnn, fc1)
        else:
            temp1 = cnn1d_module(inputs1, cnn, fc1)
            x1 = tf.keras.layers.Concatenate(axis=-1)([x1, temp1])
    
    inputs2_ls = []
    for j in range(len(list_shape_input2)):
        inputs2 = tf.keras.layers.Input(shape=list_shape_input2[j])
        inputs2_ls.append(inputs2)
        if j == 0:
            x2 = gru_module(inputs2, gru, fc2)
        else:
            temp2 = gru_module(inputs2, gru, fc2)
            x2 = tf.keras.layers.Concatenate(axis=-1)([x2, temp2])
    if fuse == 'cross':
        x1_f3 = tf.keras.layers.Dense(2*fc3)(x1)
        x1_f3 = tf.expand_dims(x1_f3, axis=2)
        x2_f3 = tf.keras.layers.Dense(2*fc3)(x2)
        x2_f3 = tf.expand_dims(x2_f3, axis=2)
        x_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=2*fc3,
            
        )(x1_f3, x2_f3, x2_f3)
        x_1 = tf.squeeze(x_1, axis=2)
        x_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=2*fc3,
            
        )(x2_f3, x1_f3, x1_f3)
        x_2 = tf.squeeze(x_2, axis=2)
        x = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2])

    x = tf.keras.layers.Dense(fc3, activation='relu')(x) 
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[inputs1_ls, inputs2_ls], outputs=outputs)

    return model