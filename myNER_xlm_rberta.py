'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

#The following Flair script was used to train this model:

import torch

# 1. get the corpus
from flair.datasets import CONLL_03

corpus = CONLL_03(base_path='./')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
#tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
label_dictionary = corpus.make_label_dictionary(label_type=tag_type)

# 4. initialize fine-tuneable transformer embeddings WITH document context
from flair.embeddings import TransformerWordEmbeddings

embeddings = TransformerWordEmbeddings(
    model='distilbert-base-uncased',#'distilbert-base-uncased'
    layers="-1",
    subtoken_pooling="first",
    fine_tune=False,#True
    use_context=False,#True
)

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
from flair.models import SequenceTagger

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dictionary,
    tag_type='ner',
    use_crf=True,
    use_rnn=False,
    reproject_embeddings=False,
)

# 6. initialize trainer with AdamW optimizer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus)

# 7. run training with XLM parameters (20 epochs, small LR)
from torch.optim.lr_scheduler import OneCycleLR

trainer.train('resources/taggers/ner-english',train_with_dev=True,max_epochs=20)