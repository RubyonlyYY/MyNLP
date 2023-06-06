'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import (WordEmbeddings,
                              FlairEmbeddings, DocumentLSTMEmbeddings, RoBERTaEmbeddings,
                              BertEmbeddings, DocumentPoolEmbeddings)
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# Avoid gpu bug
import flair, torch

flair.device = torch.device('cpu')


def create_dataset():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    labels = np.array(newsgroups_test.target_names)

    df_train = pd.DataFrame(dict(label=labels[newsgroups_train.target], text=newsgroups_train.data))
    df_train['label'] = '__label__' + df_train['label'].astype('str')

    df_test = pd.DataFrame(dict(label=labels[newsgroups_test.target], text=newsgroups_test.data))
    df_test['label'] = '__label__' + df_test['label'].astype('str')

    df_test = df_test.iloc[0:int(len(df_test) * 0.5)]
    df_dev = df_test.iloc[int(len(df_test) * 0.5):]

    for df, name in zip([df_train, df_test, df_dev], ['train', 'test', 'dev']):
        df.to_csv(name + '.csv', sep=' ', index=False, header=False, quotechar=' ')

    corpus = NLPTaskDataFetcher.load_classification_corpus('./', train_file='train.csv',
                                                           test_file='test.csv', dev_file='dev.csv')

    return corpus


def train_flair():
    corpus = create_dataset()

    word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'),
                       FlairEmbeddings('news-backward-fast')]

    document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=128, reproject_words=True)

    dictionary = corpus.make_label_dictionary()
    classifier = TextClassifier(document_embeddings, label_dictionary=dictionary, multi_label=False)

    trainer = ModelTrainer(classifier, corpus, use_tensorboard=True)
    trainer.train('resources/taggers/classy',
                  max_epochs=25, patience=1, learning_rate=0.2,
                  mini_batch_size=32, anneal_factor=0.5, embeddings_storage_mode='gpu')


create_dataset()
train_flair()