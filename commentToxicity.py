import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import gradio as gr
import matplotlib.pyplot as plt

# Install dependencies (if needed)
# !pip install tensorflow pandas matplotlib gradio

# Load dataset
df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]].values

# Vectorization
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)
train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

# Build the model
model = Sequential([
    Embedding(MAX_FEATURES + 1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()

# Train the model
history = model.fit(train, epochs=1, validation_data=val)

# Plot training history
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot()
plt.show()

# Save model
model.save('toxicity1.h5')

# Load model
model = tf.keras.models.load_model('toxicity.h5')

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += f'{col}: {results[0][idx] > 0.5}\n'
    return text

# Gradio Interface
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
                         outputs='text')
interface.launch(share=True)