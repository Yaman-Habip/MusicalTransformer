import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy  as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import requests
from artists import rock_artists # List of rock artists whose catalogues we want to scrape
import csv

# Define the Encoder
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()


# Define the Transformer model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # inputs.shape = (batch_size, seq_len, d_model)
        x = self.encoder(inputs, training = training)
        x = self.final_layer(x)
        return x


# Define the encoder layer
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation = "relu"), layers.Dense(d_model)]
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # inputs.shape = (batch_size, seq_len, d_model)
        attn_output = self.mha(inputs, inputs, attention_mask = None, training = training)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layer_norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


# Define the encoder
class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.PositionalEncoding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training):
        # inputs.shape = (batch_size, seq_len)
        seq_len = tf.shape(inputs)[1]

        # Add embedding and positional encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x, seq_len = seq_len)

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x


# Define the positional encoding layer
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype = tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype = tf.float32)[tf.newaxis, :],
            d_model = d_model,
        )

        # Apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = tf.concat([sines, cosines], axis = -1)

        return angle_rads

    def call(self, inputs, seq_len):
        return inputs + self.positional_encoding[:, :seq_len, :]