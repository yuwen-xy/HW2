#!/usr/bin/env python
# coding:utf-8

# 导入需要用的库
import os
import tarfile
# 软件包的解压
import urllib.request
# 网络下载的请求
import tensorflow as tf
import numpy as np
import codecs

import re
# 正则化
import string
from random import randint
# 数据地址
url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
filepath = 'data/aclImdb_v1.tar.gz'

# 如果当前目录下不存在data文件夹，则建立
if not os.path.exists('data'):
    os.makedirs('data')
# 下载数据，80兆左右
if not os.path.isfile(filepath):
    print('downloading...')
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)
else:
    print(filepath, 'is existed')

# 解压数据
if not os.path.exists('data/aclImdb'):
    tfile = tarfile.open(filepath, 'r:gz')
    print('extracting...')
    result = tfile.extractall('data/')
    print('extraction completed')
else:
    print('data/aclImdb is existed!')

def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    pos_location = './aclImdb/train/pos'
    pos_files = os.listdir(pos_location)
    neg_location = './aclImdb/train/neg'
    neg_files = os.listdir(neg_location)
    pos_all = codecs.open('./aclImdb/train/pos_all.txt', 'a', encoding='utf8')
    neg_all = codecs.open('./aclImdb/train/neg_all.txt', 'a', encoding='utf8')

    all = []
    for file in pos_files:
        whole_location = os.path.join(pos_location, file)
        with open(whole_location, 'r') as f:
            line = f.readlines()
            all.extend(line)
    for file in all:
        pos_all.write(file)
        pos_all.write('\n')
    pos_files_num = len()

    alls = []
    for file in neg_files:
        whole_location = os.path.join(neg_location, file)
        with open(whole_location, 'r') as f:
            try:
                line = f.readlines()
                alls.extend(line)
            except:
                continue
    for file in alls:
        neg_all.write(file)
        neg_all.write('\n')

    path = './data/aclImdb/'
    file_list = []
    # 读取正面评价的文件路径，存到file_list列表里
    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
    pos_files_num = len(file_list)
    # 读取负面评价的文件的路径，存到file_list列表里
    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
    neg_files_num = len(file_list) - pos_files_num

    print('read', filetype, 'files:', len(file_list))
    print(pos_files_num, 'pos files in', filetype, 'files')
    print(neg_files_num, 'neg files in', filetype, 'files')
    # 得到所有标签。标签用one——hot编码，正面{1,0}负面[0,1]
    all_labels = ([[1, 0]] * pos_files_num + [[0, 1]] * neg_files_num)

    # 得到所有文本
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            # 文本中有<br />这类html标签，将文本传入remove_tags函数
            # 函数里用正则表达式将标签去除
            all_texts += [remove_tags(''.join(file_input.readlines()))]
    return all_labels, all_texts


train_labels, train_texts = read_files("train")
test_labels, test_texts = read_files('test')

token = tf.keras.preprocessing.text.Tokenizer(num_words=4000)
# 分词器，把出现率最高的4000个词纳入分词器

token.fit_on_texts(train_texts)
# print(token.word_index)# 出现频率的排名

token.word_docs
# 将单词映射为他们在训练器出现的文档或文本的数量

train_sequences = token.texts_to_sequences(train_texts)
test_sequences = token.texts_to_sequences(test_texts)

print(train_texts[0])
print(train_sequences[0])

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post', truncating='post', maxlen=400)
x_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post', truncating='post', maxlen=400)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(output_dim=32,
                                    input_dim=4000,
                                    input_length=400))
# 用RNN牛拍卖行把词嵌入平坦化
# model.add(keras.layers.SimpleRNN(units=16))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=8)))
# 双相LSTM
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=128,
                    verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Test Accuracy', test_acc)
predictions = model.predict(x_test)
predictions[0]

sentiment_dict = {0: 'pos', 1: 'neg'}


def display_test_sentiment(i):
    print(test_texts[i])
    print('label values', sentiment_dict[np.argmax(y_test[i])],
          'predict value:', sentiment_dict[np.argmax()])

    display_test_sentiment(0)
