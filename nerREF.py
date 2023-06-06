'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

from flair.data import Corpus
from flair.datasets import CONLL_2000
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus CoNLL-2000 Noun Phrase Chunking (English)
corpus: Corpus = CONLL_2000()

# 2. what tag do we want to predict?
tag_type = 'np'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('extvec'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-chunk',
              train_with_dev=True,
              max_epochs=150)

##############################################################3
from flair.data import Corpus
from flair.datasets import ONTONOTES
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List

# 1. get the corpus The Ontonotes corpus is one of the best resources for different types of NLP and contains rich NER annotation
corpus: Corpus = ONTONOTES()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('crawl'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              train_with_dev=True,
              # it's a big dataset so maybe set embeddings_storage_mode to 'none' (embeddings are not kept in memory)
              embeddings_storage_mode='none')