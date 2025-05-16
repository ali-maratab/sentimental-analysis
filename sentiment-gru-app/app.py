
import numpy as np
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('sentiment_gru_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 200  # Use the same value as used during training

st.title("🎬 IMDB Sentiment Analysis (GRU Model)")

review = st.text_area("Enter a movie review:", "")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        prediction = model.predict(padded)[0][0]
        label = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
        st.success(f"**Prediction: {label}** (Confidence: {prediction:.2f})")
