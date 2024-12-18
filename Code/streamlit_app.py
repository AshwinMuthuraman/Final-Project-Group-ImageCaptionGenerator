import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Encoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, features):
        features = self.dense(features)
        features = self.relu(features)
        return features

class Attention_model(tf.keras.Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

@st.cache_resource
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((299, 299))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def generate_caption(image, encoder, decoder, tokenizer, max_length, image_features_extract_model):
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    features = image_features_extract_model.predict(image)
    features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
    features = encoder(features)

    hidden = decoder.init_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index.get('<start>', 0)], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()

        if tokenizer.index_word.get(predicted_id) is None:
            break

        word = tokenizer.index_word[predicted_id]
        if word == '<end>':
            break

        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)

    caption = ' '.join(result)
    return caption

tokenizer_path = "tokenizer.pkl"  
try:
    tokenizer = load_tokenizer(tokenizer_path)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
max_length = 31  

encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)

image_model = InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = Model(new_input, hidden_layer)

checkpoint_path = "model_weights/checkpoint1"  
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    except Exception as e:
        st.error(f"Error restoring checkpoint: {e}")
        st.stop()
else:
    st.error(f"No checkpoint found at {checkpoint_path}")
    st.stop()

st.title("Image Caption Generator")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        caption = generate_caption(image, encoder, decoder, tokenizer, max_length, image_features_extract_model)
        st.write(f"{caption}")
    except Exception as e:
        st.error(f"Error generating caption: {e}")
else:
    st.write("Please upload an image to generate a caption.")

