'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR

from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

import rubrix as rb

def train() -> None:
    # 1. Load the dataset from Rubrix
    limit_num = 2048
    train_dataset = rb.load("tweet_eval_emojis", limit=limit_num).to_pandas()

    # 2. Pre-processing training pandas dataframe
    train_df = pd.DataFrame()
    train_df['text'] = train_dataset['text']
    train_df['label'] = train_dataset['annotation']

    # 3. Save as csv with tab delimiter
    train_df.to_csv('train.csv', sep='\t')
    # 4. Read the with CSVClassificationCorpus
    data_folder = './'

    # column format indicating which columns hold the text and label(s)
    label_type = "label"
    column_name_map = {1: "text", 2: "label"}

    corpus = CSVClassificationCorpus(
        data_folder, column_name_map, skip_header=True, delimiter='\t', label_type=label_type)

    # 5. create the label dictionary
    label_dict = corpus.make_label_dictionary(label_type=label_type)


    # 6. initialize transformer document embeddings (many models are available)
    document_embeddings = TransformerDocumentEmbeddings(
        'distilbert-base-uncased', fine_tune=True)


    # 7. create the text classifier
    classifier = TextClassifier(
        document_embeddings, label_dictionary=label_dict, label_type=label_type)

    # 8. initialize trainer with AdamW optimizer
    trainer = ModelTrainer(classifier, corpus, optimizer=torch.optim.AdamW)


    # 9. run training with fine-tuning
    trainer.train('./emojis-classification',
                  learning_rate=5.0e-5,
                  mini_batch_size=4,
                  max_epochs=4,
                  scheduler=OneCycleLR,
                  embeddings_storage_mode='none',
                  weight_decay=0.,
                  )
def predict_example() -> None:
    from flair.data import Sentence
    from flair.models import TextClassifier

    classifier = TextClassifier.load('./emojis-classification/best-model.pt')

    # create example sentence
    sentence = Sentence('Farewell, Charleston! The memories are sweet #mimosa #dontwannago @ Virginia on King')

    # predict class and print
    classifier.predict(sentence)

    print(sentence.labels)

if __name__ == '__main__':
    #train()
    predict_example()