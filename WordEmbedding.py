import numpy as np
import tensorflow 
import keras

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sentences = [
    'I am sabita rajbanshi',
    'I am a software engineer',
    'I am currently learning Machine Learning',
    'Artificial intelligence is the future of humankind',
    '2021 Learnings AI ML DL DS'
]

### initialize the vocabulary size=10000
vocab_size = 10000

### one hot representation of word
onehot_repre = [one_hot(words, vocab_size) for words in sentences]

### word embedding representation
length_of_sentence = 8
embedded_docs = pad_sequences(onehot_repre, padding='pre', maxlen=length_of_sentence)

# Model Sequential
dim=8
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=length_of_sentence))

embedded_docs[0]
print(model.predict(embedded_docs[0]))