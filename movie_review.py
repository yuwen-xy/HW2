#!/usr/bin/env python
# coding:utf-8

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import numpy as np

pos_list = []
with open('./aclImdb/train/pos_all.txt', 'r')as f:
    line = f.readlines()
    pos_list.extend(line)
neg_list = []
with open('./aclImdb/train/neg_all.txt', 'r')as f:
    line = f.readlines()
    neg_list.extend(line)

label = [1 for i in range(12500)]
label.extend([0 for i in range(12499)])

content = pos_list.extend(neg_list)
content = pos_list

seq = []
seqtence = []
stop_words = set(stopwords.words('english'))
for con in content:
    words = nltk.word_tokenize(con)
    line = []
    for word in words:
        if word.isalpha() and word not in stop_words:
            line.append(word)
    seq.append(line)
    seqtence.extend(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(content)
one_hot_results = tokenizer.texts_to_matrix(content, mode='binary')
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(seq)
final_sequences = sequence.pad_sequences(sequences, maxlen=800)

label = np.array(label)
indices = np.random.permutation(len(final_sequences))
X = final_sequences[indices]
y = label[indices]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Embedding(89483, 256, input_length=800))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, batch_size=32, epochs=10, validation_data=(Xtest, ytest))
