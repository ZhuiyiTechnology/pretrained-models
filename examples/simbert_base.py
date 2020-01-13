#! -*- coding: utf-8 -*-
# SimBERT base 基本例子

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.layers import *


maxlen = 32

# bert配置
config_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_bert_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='seq2seq',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


def random_generate(s, n=5):
    """通过随机采样来seq2seq生成
    """
    token_ids, segment_ids = tokenizer.encode(s)
    token_count = Counter(token_ids)
    target_ids = [[] for _ in range(n)]
    R = []
    for i in range(maxlen):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        # 下面直接忽略[PAD], [UNK], [CLS]
        _probas = seq2seq.predict([_target_ids, _segment_ids])[:, -1, 3:]
        for i, p in enumerate(_probas):
            target_count = Counter(target_ids[i])
            for j, k in target_count.items():
                if token_count.get(j, 2) - k == 0:
                    p[j - 3] *= 0
            p /= sum(p)
            target_ids[i].append(np.random.choice(len(p), p=p) + 3)
        for t in target_ids:
            if t[-1] == 3:
                R.append(tokenizer.decode(t))
        target_ids = [t for t in target_ids if t[-1] != 3]
        if len(target_ids) == 0:
            break
    return R


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = [i for i in set(random_generate(text, n)) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


"""
gen_synonyms(u'微信和支付宝哪个好？')

[
    u'微信和支付宝，哪个好?',
    u'微信和支付宝哪个好',
    u'支付宝和微信哪个好',
    u'支付宝和微信哪个好啊',
    u'微信和支付宝那个好用？',
    u'微信和支付宝哪个好用',
    u'支付宝和微信那个更好',
    u'支付宝和微信哪个好用',
    u'微信和支付宝用起来哪个好？',
    u'微信和支付宝选哪个好',
    u'微信好还是支付宝比较用',
    u'微信与支付宝哪个',
    u'支付宝和微信哪个好用一点？',
    u'支付宝好还是微信',
    u'微信支付宝究竟哪个好',
    u'支付宝和微信哪个实用性更好',
    u'好，支付宝和微信哪个更安全？',
    u'微信支付宝哪个好用？有什么区别',
    u'微信和支付宝有什么区别？谁比较好用',
    u'支付宝和微信哪个好玩'
]
 """
