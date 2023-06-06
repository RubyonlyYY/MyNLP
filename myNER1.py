'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

'''
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
#tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
tagger = SequenceTagger.load("D:/myDev/myNER/Lib/flair/ner-english-ontonotes-fast")
# make example sentence
sentence = Sentence("On September 1st George Washington won 1 dollar.")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)
'''
import torch
from torch.optim import Optimizer
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence

def training():
    # 1. get the corpus
    corpus = CONLL_03('C:/Users/10068/myNLP/myproject') # add a path by yuan after see this defintion/line:1223 in sequence_labeling.py[/datasets/]
    '''
   //token pos_tags chunk_tags  ner_tags
     单词    词性      语法块       实体标签
    EU       NNP     I-NP      I-ORG
    rejects  VBZ     I-VP       O
    German   JJ      I-NP      I-MISC
    call     NN      I-NP       O
    to       TO      I-VP       O
    boycott  VB      I-VP       O
    British  JJ      I-NP      I - MISC
    lamb     NN      I-NP       O
    ..O  O
    '''
    print(corpus)
    print('[myLog]:label_dict ',len(corpus.train))
    print('[myLog]:label_dict --tag ',corpus.train[0].to_tagged_string('ner'))
    print('[myLog]:label_dict --tag ',corpus.train[1].to_tagged_string('ner'))
    print('[myLog]:label_dict --tag ',corpus.train[2].to_tagged_string('ner'))

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type) # delete add_unk=False by yuan see this defintion/line:1432 in data.py(flair ver0.11)
    print('[myLog]:label_dict ',label_dict)
    '''
    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
        TransformerWordEmbeddings(model='roberta-base', cache_dir='D:/myDev/myNER/Lib/huggingface_hub/')
        #TransformerWordEmbeddings('bert-base-uncased',cache_dir='D:/myDev/myNER/Lib/huggingface_hub/', layers='-1', layer_mean=False)
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)
    '''
    embeddings = TransformerWordEmbeddings(
        model='xlm-roberta-large',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        #fine_tune=False,
        use_context=True,
    )

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,  #change 256 into 128 by yuan
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            rnn_type='LSTM',
                            use_rnn=True,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus) #,optimizer=torch.optim.AdamW)

    # 7. start training
    trainer.train('resources/taggers/sota-ner-flair', # base_path=D:\myDev\myNER\Lib,myNER1 in it
                  learning_rate=0.1,
                  #learning_rate=5.0e-6,
                  train_with_dev=True,
                  mini_batch_size=16, #change 32 into 2 by yuan
                  max_epochs=100) #change 500 into 2 by yuan

def NerPredict():
    # load tagger
    tagger = SequenceTagger.load("resources/taggers/sota-ner-flair/best-model.pt")

    sentence = Sentence('yuan say : France is the current world cup winner')
    #sentence = Sentence('Hi. Yes mum, I will...')
    tagger.predict(sentence)
    print(sentence)
    # print predicted NER spans
    print('The following NER tags are found:')
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)

print('[myLog]:start training......... ')
training()
print('[myLog]:start predict......... ')
#NerPredict()

