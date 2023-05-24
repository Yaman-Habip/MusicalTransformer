import tensorflow as tf
from tensorflow import keras
from keras import layers
from data_processing import in_model, out_model, max_val

# Define the Transformer model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = layers.Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
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


# Define hyperparameters and create the model
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000  # Change this according to your data
target_vocab_size = 10000  # Change this according to your data
dropout_rate = 0.1

model = TransformerModel(
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    dff = dff,
    input_vocab_size = input_vocab_size,
    target_vocab_size = target_vocab_size,
    dropout_rate = dropout_rate,
)

# Compile the model
model.compile(optimizer = keras.optimizers.Adam(),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True))

# Define your training data and labels

"""Training Data looks like in: [[a,b,c]
                [a,b,c,d]
                [a,b,c,d,e]]
                out: [d]
                     [e]
                     [f]
    Out is represented as: [[[0,0,0,1]]
                            [[0,0,0,0,1]]
                            [[0,0,0,0,0,1]]]"""
                
train_data = in_model
train_labels = out_model

# Pad the input chord sequences
padding_token_index = 1000 # Unique value not present in data set
train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data, maxlen=max_val, padding="post", value=padding_token_index
)

# Create the input mask
input_mask = tf.math.not_equal(train_data, padding_token_index)

# Train the model
model.fit(train_data, train_labels, batch_size=64, epochs=10, sample_weight=input_mask)