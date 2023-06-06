'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

from flair.data import Sentence
from flair.datasets import RE_ENGLISH_CONLL04
from flair.embeddings import TransformerWordEmbeddings
from flair.models import RelationExtractor
from flair.trainers import ModelTrainer


def train() -> None:
    # Hyperparameters
    transformer: str = 'xlm-roberta-large'
    learning_rate: float = 5e-5
    mini_batch_size: int = 8

    # Step 1: Create the training data

    # The relation extractor is *not* trained end-to-end.
    # A corpus for training the relation extractor requires annotated entities and relations.
    corpus: RE_ENGLISH_CONLL04 = RE_ENGLISH_CONLL04()

    # Print examples
    sentence: Sentence = corpus.test[0]
    print(sentence)
    print(sentence.get_spans('ner'))  # 'ner' is the entity label type
    print(sentence.get_relations('relation'))  # 'relation' is the relation label type

    # Step 2: Make the label dictionary from the corpus
    label_dictionary = corpus.make_label_dictionary('relation')
    label_dictionary.add_item('O')
    print(label_dictionary)

    # Step 3: Initialize fine-tunable transformer embeddings
    embeddings = TransformerWordEmbeddings(
        model=transformer,
        layers='-1',
        subtoken_pooling='first',
        fine_tune=True
    )

    # Step 4: Initialize relation classifier
    model: RelationExtractor = RelationExtractor(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type='relation',
        entity_label_type='ner',
        entity_pair_filters=[  # Define valid entity pair combinations, used as relation candidates
            ('Loc', 'Loc'),
            ('Peop', 'Loc'),
            ('Peop', 'Org'),
            ('Org', 'Loc'),
            ('Peop', 'Peop')
        ]
    )

    # Step 5: Initialize trainer
    trainer: ModelTrainer = ModelTrainer(model, corpus)

    # Step 7: Run fine-tuning
    trainer.fine_tune(
        'conll04',
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        main_evaluation_metric=('macro avg', 'f1-score')
    )


def predict_example() -> None:
    # Step 1: Load trained relation extraction model
    model: RelationExtractor = RelationExtractor.load('conll04/final-model.pt')

    # Step 2: Create sentences with entity annotations (as these are required by the relation extraction model)
    # In production, use another sequence tagger model to tag the relevant entities.
    sentence: Sentence = Sentence('On April 14, while attending a play at the Ford Theatre in Washington, '
                                  'Lincoln was shot in the head by actor John Wilkes Booth.')
    sentence[15:16].add_label(typename='ner', value='Peop', score=1.0)  # Lincoln -> Peop
    sentence[23:26].add_label(typename='ner', value='Peop', score=1.0)  # John Wilkes Booth -> Peop

    # Step 3: Predict
    model.predict(sentence)
    print(sentence.get_relations('relation'))


if __name__ == '__main__':
    train()
    predict_example()