#!/usr/bin/env python
# coding:utf-8

import os
import codecs

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
