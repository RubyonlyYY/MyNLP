'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
from flair.data import Corpus
from flair.datasets import TREC_6 # 1670 in document_classification.py
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
import torch
import flair
import matplotlib.pyplot as plt

def load_dataset():
    data = pd.read_csv("./dataset/class/IMDBDataset.csv", encoding='latin-1')

    data.sample(frac=1).drop_duplicates()
    data = data[['review', 'sentiment']].rename(columns={"review": "text", "sentiment": "label"})
    data['label'] = '__label__' + data['label'].astype(str)
    data = data[['label', 'text']]
    border_1 = int(len(data) * 0.8)
    border_2 = int(len(data) * 0.9)

    data.iloc[0:border_1].to_csv('./dataset/class/train.csv', sep='\t', index=False, header=False)
    data.iloc[border_1:border_2].to_csv('./dataset/class/test.csv', sep='\t', index=False, header=False)
    data.iloc[border_2:].to_csv('./dataset/class/dev.csv', sep='\t', index=False, header=False)

def plot():
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves('resources/taggers/ner-english-large/loss20230420conll.tsv')
    plotter.plot_weights('resources/taggers/question-classification-with-transformer/weights.txt')

    data = pd.read_csv('resources/taggers/question-classification-with-transformer/loss.tsv', sep='\t')
    #data.plot()
    plt.figure('result')
    plt.xlabel("x")
    plt.ylabel("y")
    print(data["EPOCH"])
    plt.plot(data["EPOCH"], data['DEV_F1'])
    plt.plot(data["EPOCH"], data["DEV_LOSS"])
    plt.plot(data["EPOCH"], data["DEV_PRECISION"])
    plt.show()

def NERplot():
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    #plotter.plot_training_curves('resources/taggers/ner-english-large/loss20230420conll.tsv')
    #plotter.plot_weights('resources/taggers/ner-english-large/weights.txt')

    #data = pd.read_csv('resources/taggers/ner-english-large/loss20230420conll.tsv', sep='\t')
    data = pd.read_csv('resources/taggers/question-classification-with-transformer/123.csv')
    #data.plot()
    #print( data['DEV_F1'] )
    plt.figure('result')
    plt.xlabel("x")
    plt.ylabel("y")
    print(data["EPOCH"])
    #plt.plot(data["EPOCH"], data['DEV_F1'], color="red", linewidth=1.0, marker='s', linestyle="--")
    #plt.plot(data["EPOCH"], data["DEV_LOSS"], color="blue", linewidth=1.0, marker='s', linestyle="--")
    #plt.plot(data["EPOCH"], data["DEV_PRECISION"], color="green", linewidth=1.0, marker='s', linestyle="--")
    #plt.plot(data["EPOCH"], data["DEV_RECALL"], color="orange", linewidth=1.0, marker='s', linestyle="--")
    #plt.plot(data["EPOCH"], data["DEV_ACCURACY"], color="yellow", linewidth=1.0, marker='s', linestyle="--")
    plt.plot(data["EPOCH"], data['DEV_F1'], color="red", label='F1')
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.show()
    plt.plot(data["EPOCH"], data["DEV_LOSS"], color="blue", label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(data["EPOCH"], data["DEV_PRECISION"], color="green", label='Precision')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    plt.plot(data["EPOCH"], data["DEV_RECALL"], color="orange", label='Recall')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.show()
    plt.plot(data["EPOCH"], data["DEV_ACCURACY"], color="yellow", label='Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
def train() -> None:
    flair.device = torch.device('cpu')


    '''
    # this is the folder in which train, test and dev files reside
    data_folder = 'tests/resources/tasks/ag_news'

    # load corpus containing training, test and dev data
    corpus: Corpus = ClassificationCorpus(data_folder,
                                          test_file='test.txt',
                                          dev_file='dev.txt',
                                          train_file='train.txt',
                                          label_type='topic',  # here you give a name to the label
                                          )
    corpus: Corpus = CSVClassificationCorpus(
    Path('./flair_spam/'),
    test_file='test.csv', 
    dev_file='dev.csv', 
    train_file='train.csv',
    delimiter=';',
    column_name_map={0: 'text', 1: 'label'},
    label_type='topic')
    
    print(corpus)

    dictionary = corpus.make_label_dictionary(label_type='topic')  # use the name again to create dictionary

    print(dictionary)
    '''

    # 1. get the corpus
    #corpus: Corpus = TREC_6()
    #label_type = 'topic'
    label_type = 'sentiment'
    data_folder = './dataset/class/'
    corpus: Corpus = ClassificationCorpus(data_folder, test_file='test.csv', dev_file='dev.csv',
                                          train_file='train.csv', label_type=label_type)

    # 2. what label do we want to predict?


    # 3. create the label dictionary
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    # 4. initialize transformer document embeddings (many models are available),微调分类器True
    document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

    # 6. initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    # 7. run training with fine-tuning
    trainer.fine_tune('resources/taggers/question-classification-with-transformer',
                      learning_rate=5.0e-5,
                      mini_batch_size=4,
                      max_epochs=1,
                      )


def predict_example() -> None:
    '''
    ABBR - Abbreviation
    DESC - Description and abstract concepts
    ENTY - Entities
    HUM - Human beings
    LOC - Locations
    NYM - Numeric values
    '''
    classifier = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')

    # create example sentence
    #sentence = Sentence('Who built the Eiffel Tower ?')
    sentence = Sentence('Nice hair dryer, dual voltage comes in handy')
    sentence = Sentence('This is a piece of junk.')
    sentence = Sentence('Set it up on the high watt setting, but  found that it would not operate after my wife dried her hair.')
    # predict class and print
    classifier.predict(sentence)

    print(sentence.labels)

if __name__ == '__main__':
    #load_dataset()
    #train()
    #predict_example()
    #plot()
    NERplot()