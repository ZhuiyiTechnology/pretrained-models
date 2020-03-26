#! -*- coding:utf-8 -*-
# 评估脚本
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)

import json
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import *

set_gelu('tanh')  # 切换tanh版本


num_classes = 119
maxlen = 128
batch_size = 32

# RoBERTa small
config_path = '/root/kg/bert/chinese_roberta_L-6_H-384_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_L-6_H-384_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_L-6_H-384_A-12/vocab.txt'
model_type = 'bert'

"""
# albert small
config_path = '/root/kg/bert/albert_small_zh_google/albert_config.json'
checkpoint_path = '/root/kg/bert/albert_small_zh_google/albert_model.ckpt'
dict_path = '/root/kg/bert/albert_small_zh_google/vocab.txt'
model_type = 'albert'

# RoBERTa tiny
config_path = '/root/kg/bert/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_L-4_H-312_A-12/vocab.txt'
model_type = 'bert'

# albert tiny
config_path = '/root/kg/bert/albert_tiny_zh_google/albert_config.json'
checkpoint_path = '/root/kg/bert/albert_tiny_zh_google/albert_model.ckpt'
dict_path = '/root/kg/bert/albert_tiny_zh_google/vocab.txt'
model_type = 'albert'
"""


def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('/root/CLUE-master/baselines/CLUEdataset/iflytek/train.json')
valid_data = load_data('/root/CLUE-master/baselines/CLUEdataset/iflytek/dev.json')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=model_type,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(units=num_classes,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(5e-5),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=50,
                    callbacks=[evaluator])
