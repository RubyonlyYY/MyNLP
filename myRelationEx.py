'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

'''
from flair.data import Sentence
from flair.models import RelationExtractor, SequenceTagger

# 1. make example sentence
sentence = Sentence("George was born in Washington")

# 2. load entity tagger and predict entities
tagger = SequenceTagger.load('ner-fast')
tagger.predict(sentence)

# check which entities have been found in the sentence
entities = sentence.get_labels('ner')
for entity in entities:
    print(entity)

# 3. load relation extractor
extractor: RelationExtractor = RelationExtractor.load('relations-fast')

# predict relations
extractor.predict(sentence)

# check which relations have been found
relations = sentence.get_labels('relation')
for relation in relations:
    print(relation)
'''

from flair.data import Sentence
from flair.models import RelationExtractor, SequenceTagger

# 1. make example sentence
sentence = Sentence("George was born in Washington")
sentence = Sentence("We have prepared data files with train/dev/test split in our another project")
#sentence = Sentence("I found everything goes well except the plug")

# 2. load entity tagger and predict entities
tagger = SequenceTagger.load('ner-fast')
tagger.predict(sentence)

# check which entities have been found in the sentence
print("predict entities in sentence")
entities = sentence.get_labels('ner')
for entity in entities:
    print(entity)

# 3. load relation extractor
extractor: RelationExtractor = RelationExtractor.load('relations')

# predict relations
print("predict relations")
extractor.predict(sentence)

# check which relations have been found
relations = sentence.get_labels('relation')
for relation in relations:
    print(relation)