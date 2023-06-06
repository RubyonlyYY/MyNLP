'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask

# load your corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS)
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter

# define your search space
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ WordEmbeddings('en') ],
    [ CharLMEmbeddings('news-forward'), CharLMEmbeddings('news-backward') ]
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ CharLMEmbeddings('news-forward'), CharLMEmbeddings('news-backward') ]
])
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue

# create the parameter selector
param_selector = TextClassifierParamSelector(
    corpus,
    False,
    'resources/results',
    'lstm',
    max_epochs=50,
    training_runs=3,
    optimization_value=OptimizationValue.DEV_SCORE
)

# start the optimization
param_selector.optimize(search_space, max_evals=100)