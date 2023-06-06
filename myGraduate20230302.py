'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
import numpy as np
import re
from flair.datasets.sequence_labeling import (
    ONTONOTES,
    JsonlCorpus,
    JsonlDataset,
    MultiFileJsonlCorpus,
)
from typing import Union

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|@[^\s]+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile(r'<[^>]+>')
    return html_pattern.sub(r'', text)
def remove_other(text):
    other_pattern = re.compile(r'&#34;')
    return other_pattern.sub(r'', text)
def fetch_reviews_from_rawfile():
    '''
    Amazon Reviews file in .tsv [fields]: 1.marketplace 2.customer_id 3.review_id 4.product_id  5.product_parent
                              6.product_title 7.product_category 8.star_rating 9.review_body 10.review_date
    '''
    dataset=pd.read_csv('D:/myDev/myNER/dataset/hair_dryer.tsv', sep='\t')
    #reviews = dataset["review_body"].apply(lambda text: remove_other(remove_html(remove_urls(text))) ).head(2000)
    #reviews.to_csv(r'D:/myDev/myNER/dataset/hair_dryer_reviews_train.txt', sep='\t', header=None, index=None)

    reviews = dataset["review_body"].apply(lambda text: remove_other(remove_html(remove_urls(text))) ).iloc[2000:3000]
    reviews.to_csv(r'D:/myDev/myNER/dataset/hair_dryer_reviews_test.txt', sep='\t', header=None, index=None)

    reviews = dataset["review_body"].apply(lambda text: remove_other(remove_html(remove_urls(text))) ).iloc[3000:4000]
    reviews.to_csv(r'D:/myDev/myNER/dataset/hair_dryer_reviews_dev.txt', sep='\t', header=None, index=None)


def load(tasks_base_path):
    data = pd.read_csv(".\\data\\spam.csv", encoding='latin-1')

    data.sample(frac=1).drop_duplicates()
    data = data[['v1', 'v2']].rename(columns={"v1": "label", "v2": "text"})
    data['label'] = '__label__' + data['label'].astype(str)
    border_1 = int(len(data) * 0.8)
    border_2 = int(len(data) * 0.9)

    data.iloc[0:border_1].to_csv('.\\data\\train.csv', sep='\t', index=False, header=False)
    data.iloc[border_1:border_2].to_csv('.\\data\\test.csv', sep='\t', index=False, header=False)
    data.iloc[border_2:].to_csv('.\\data\\dev.csv', sep='\t', index=False, header=False)

    corpus: Corpus = ClassificationCorpus(Path('.\\data\\'), test_file='test.csv', dev_file='dev.csv',
                                          train_file='train.csv', label_type='topic')
    word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'),
                       FlairEmbeddings('news-backward-fast')]
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                reproject_words_dimension=256)
    classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(label_type='topic'),
                                multi_label=False, label_type='topic')
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('.\\model', max_epochs=20)

def train():
    from flair.data import Corpus
    from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings
    from typing import List
    from flair.datasets import ColumnCorpus
    from flair.models import SequenceTagger
    from flair.trainers import ModelTrainer



    columns = {0: 'text', 1: 'ner'}
    data_folder = './dataset/'
    corpus: Corpus = ColumnCorpus(data_folder, columns,  train_file='train.txt', dev_file='dev.txt',
                                  test_file='test.txt') #document_separator_token="-DOCSTART-"
    #corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_column_corpus("./", columns,
    #                                                              train_file="train.txt",
    #                                                              test_file="test.txt",
    #                                                              dev_file="dev.txt")
    print('[myLog]:label_dict ', corpus.train)
    print('[myLog]:label_dict ',len(corpus.train))
    print('[myLog]:label_dict ',len(corpus.test))
    print('[myLog]:label_dict ',len(corpus.dev))

    print('[myLog]:label_dict ',corpus.train[0])
    print('[myLog]:label_dict ', corpus.test[0])
    print('[myLog]:label_dict ', corpus.dev[0])
    print('[myLog]:label_dict --tag ',corpus.train[0].to_tagged_string('ner'))
    print('[myLog]:label_dict --test tag ',corpus.test[0].to_tagged_string('ner'))

    tag_type = 'ner'
    #tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    print('[myLog]:label_dict ', tag_dictionary)
    #label_type = 'ner'
    #label_dict = corpus.make_label_dictionary(label_type=label_type)

    embedding_types: List[TokenEmbeddings] = [#WordEmbeddings('crawl'),
                                              CharacterEmbeddings(),
                                              FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward'), ]
    #embedding_types: List[TokenEmbeddings] = [WordEmbeddings('glove'), ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    #tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type,
    #                       tag_format='BIO', use_crf=True).load('ner')
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
    )
    trainer = ModelTrainer(tagger, corpus)
    trainer.train('resources/taggers/sota-ner-flair', learning_rate=0.01, mini_batch_size=16, max_epochs=5)

def NerPredict():
    from flair.models import SequenceTagger
    from flair.data import Sentence

    model = SequenceTagger.load('resources/taggers/sota-ner-flair/best-model.pt')

    #sentence = Sentence('Not as good as the original Conair 1875. Makeovers and upgrades don\'t always yield the best results.For instance, this 1875 replica. Lacks the basic functions a blow dryer should have, such as temperature adjustment. There is only a setting for high and low fan speed. Also, in order to use the air only feature with no heat, the cooling button has to be pressed and held down to use this basic function.' )
    sentence = Sentence('This thing has lots of power - -- works so fast, and seems to do so without damaging the hair...I was wavering between this dryer and another that was a bit cheaper.But was glad that I chose this  one, as it\'s the most satisfying hair dryer I\' ve used at home thus far.My only issue with it is that the sliding switches are annoying to use, as they are very stiff.Wish that this awesome product came with easier to maneuver switches....Maybe next time, in the redesign, it will?')
    #sentence = Sentence('Great! Everything I expected from Conair.MORE than enough heat for my thick curly hair.Love it!')
    model.predict(sentence)
    print(sentence)
    # print predicted NER spans
    print('The following NER tags are found:')
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)

def NERevaluate():
    from flair.models import SequenceTagger
    from flair.data import Sentence

    model: SequenceTagger = SequenceTagger.load('resources/taggers/sota-ner-flair/best-model.pt')

    result, _ = model.evaluate(corpus.test)
    print(result.detailed_results)

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

def biluo_tags_from_offsets(doc, entities, missing="O"):
    """Encode labelled spans into per-token tags, using the
    Begin/In/Last/Unit/Out scheme (BILUO).
    doc (Doc): The document that the entity offsets refer to. The output tags
        will refer to the token boundaries within the document.
    entities (iterable): A sequence of `(start, end, label)` triples. `start`
        and `end` should be character-offset integers denoting the slice into
        the original string.
    RETURNS (list): A list of unicode strings, describing the tags. Each tag
        string will be of the form either "", "O" or "{action}-{label}", where
        action is one of "B", "I", "L", "U". The string "-" is used where the
        entity offsets don't align with the tokenization in the `Doc` object.
        The training algorithm will view these as missing values. "O" denotes a
        non-entity token. "B" denotes the beginning of a multi-token entity,
        "I" the inside of an entity of three or more tokens, and "L" the end
        of an entity of two or more tokens. "U" denotes a single-token entity.
    EXAMPLE:
        text = 'I like London.'
        entities = [(len('I like '), len('I like London'), 'LOC')]
        doc = nlp.tokenizer(text)
        tags = biluo_tags_from_offsets(doc, entities)
        assert tags == ["O", "O", 'U-LOC', "O"]
    """
    from spacy.errors import Errors
    # Ensure no overlapping entity labels exist
    tokens_in_ents = {}

    starts = {token.idx: token.i for token in doc}
    ends = {token.idx + len(token): token.i for token in doc}
    biluo = ["-" for _ in doc]
    # Handle entity cases
    for start_char, end_char, label in entities:
        for token_index in range(start_char, end_char):
            if token_index in tokens_in_ents.keys():
                raise ValueError(Errors.E103.format(
                    span1=(tokens_in_ents[token_index][0],
                            tokens_in_ents[token_index][1],
                            tokens_in_ents[token_index][2]),
                    span2=(start_char, end_char, label)))
            tokens_in_ents[token_index] = (start_char, end_char, label)

        start_token = starts.get(start_char)
        end_token = ends.get(end_char)
        # Only interested if the tokenization is correct
        if start_token is not None and end_token is not None:
            if start_token == end_token:
                biluo[start_token] = "U-%s" % label
            else:
                biluo[start_token] = "B-%s" % label
                for i in range(start_token+1, end_token):
                    biluo[i] = "I-%s" % label
                biluo[end_token] = "L-%s" % label
    # Now distinguish the O cases from ones where we miss the tokenization
    entity_chars = set()
    for start_char, end_char, label in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            if i in entity_chars:
                break
        else:
            biluo[token.i] = missing
    if "-" in biluo:
        ent_str = str(entities)
        #warnings.warn(Warnings.W030.format(
        #    text=doc.text[:50] + "..." if len(doc.text) > 50 else doc.text,
        #    entities=ent_str[:50] + "..." if len(ent_str) > 50 else ent_str
        #))
    return biluo


def parse():
    import jsonlines

    with open("../dataset/reviews/yaoyuan.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            print(item['entities'])
def convert_jsonl_to_FlairBIO1(tasks_base_path):
    import jsonlines
    with open("../dataset/reviews/yaoyuan.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            sent = item['text']
            tags = []
            for element in item['label'] :
                tags.append( tuple(element) )

    import spacy

    #from spacy.gold import biluo_tags_from_offsets
    nlp = spacy.load("en_core_web_sm")

    with open("flair_ner.txt", "w") as f:

        doc = nlp(sent) # token
        print(doc)
        print(tags)
        biluo = biluo_tags_from_offsets(doc, tags)
        print(biluo)
        for word, tag in zip(doc, biluo):  # zip生成两个迭代类型的新元组
            f.write(f"{word} {tag}\n")
        f.write("\n")
def convert_jsonl_to_FlairBIO2(tasks_base_path):
    import spacy
    import jsonlines
    nlp = spacy.load("en_core_web_sm")

    #with open("./dataset/train.jsonl", "r+", encoding="utf8") as f1, open("./dataset/train.txt", "w", encoding="utf8") as f2:
    #with open("./dataset/dev.jsonl", "r+", encoding="utf8") as f1, open("./dataset/dev.txt", "w", encoding="utf8") as f2:
    with open("./dataset/test.jsonl", "r+", encoding="utf8") as f1, open("./dataset/test.txt", "w", encoding="utf8") as f2:
        for item in jsonlines.Reader(f1):
            sent = item['text']
            tags = []
            for element in item['label'] :
                tags.append( tuple(element) )

            doc = nlp(sent) # token
            print(doc)
            print(tags)
            biluo = biluo_tags_from_offsets(doc, tags)
            print(biluo)
            for word, tag in zip(doc, biluo):  # zip生成两个迭代类型的新元组
                f2.write(f"{word} {tag}\n")
            f2.write("\n")


def convert_jsonl_to_FlairBIO(tasks_base_path):
    import spacy
    #from spacy.gold import biluo_tags_from_offsets
    nlp = spacy.load("en_core_web_sm")

    ents = [("George Washington went to Washington", {'entities': [(0, 6, 'PER'), (7, 17, 'PER'), (26, 36, 'LOC')]}),
            ("Uber blew through $1 million a week", {'entities': [(0, 4, 'ORG')]}),
            ]

    with open("flair_ner.txt", "w") as f:
        for sent, tags in ents:
            doc = nlp(sent)
            print(doc)
            print(tags['entities'])
            biluo = biluo_tags_from_offsets(doc, tags['entities'])
            for word, tag in zip(doc, biluo):  # zip生成两个迭代类型的新元组
                f.write(f"{word} {tag}\n")
            f.write("\n")

def corpus(data_folder):
    from flair.data import Corpus
    from flair.datasets import ColumnCorpus

    # define columns
    columns = {0: 'text', 1: 'pos', 2: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = '/path/to/data/folder'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')
    print(corpus.train[0].to_tagged_string('ner'))
    print(corpus.train[1].to_tagged_string('pos'))

def convert_jsonl_to_flair(tasks_base_path):
    import spacy
    from spacy.training import JsonlCorpus

    #nlp = spacy.load("./resources/en_core_web_sm-2.3.0/en_core_web_sm")
    #下载一个small的基于web上预训练English模型管道：python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    #corpus = JsonlCorpus("../dataset/reviews/yaoyuan.jsonl")
    nlp.max_length = 1030000  # or higher
    corpus = JsonlCorpus("../dataset/reviews/yaoyuan.jsonl")
    print(type(corpus))
    data = corpus(nlp)

    # Flair supports BIO and BIOES, see https://github.com/flairNLP/flair/issues/875
    def rename_biluo_to_bioes(old_tag):
        new_tag = ""
        try:
            if old_tag.startswith("L"):
                new_tag = "E" + old_tag[1:]
            elif old_tag.startswith("U"):
                new_tag = "S" + old_tag[1:]
            else:
                new_tag = old_tag
        except:
            pass
        return new_tag


    def generate_corpus():
        corpus = []
        n_ex = 0
        for example in data:
            n_ex += 1
            print(type(example))
            print(example)

            text = example.text
            doc = nlp(text)
            #tags = example.get_aligned_ner()
            tags = example['doc_annotation']['entities']
            print(tags)
            print(n_ex)
            # Check if it's an empty list of NER tags.
            if None in tags:
                pass
            else:
                new_tags = [rename_biluo_to_bioes(tag) for tag in tags]
                for token, tag in zip(doc, new_tags):
                    row = token.text + ' ' + token.pos_ + ' ' + tag + '\n'
                    corpus.append(row)
                corpus.append('\n')
        return corpus


    def write_file(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            corpus = generate_corpus()
            f.writelines(corpus)

    write_file('./dataset/train0322.txt')

def test_simple_folder_jsonl_corpus_should_load(tasks_base_path):
    corpus = JsonlCorpus(tasks_base_path , "yaoyuan.jsonl" )
    for sentence in corpus.get_all_sentences():
        print(sentence.has_label("Noise"))




def detectUTF8(file_name):
    import sys
    state = 0
    line_num = 0
    file_obj = open(file_name)
    all_lines = file_obj.readlines()
    file_obj.close()
    for line in all_lines:
        line_num += 1
        line_len = len(line)
        for index in range(line_len):
            if state == 0:
                if ord(line[index])&0x80 == 0x00:#上表中的第一种情况
                    state = 0
                elif ord(line[index])&0xE0 == 0xC0:#上表中的第二种情况
                    state = 1
                elif ord(line[index])&0xF0 == 0xE0:#上表中的第第三种
                    state = 2
                elif ord(line[index])&0xF8 == 0xF0:#上表中的第第四种
                    state = 3
                else:
                    print("%s isn't a utf8 file,line:\t"%file_name+str(line_num))
                    sys.exit(1)
            else:
                if not ord(line[index])&0xC0 == 0x80:
                    print("%s isn't a utf8 file in line:\t"%file_name+str(line_num))
                    sys.exit(1)
                state -= 1

if __name__ == '__main__':
    #fetch_reviews_from_rawfile()
    #test_simple_folder_jsonl_corpus_should_load("D:/myDev/myNER/dataset/reviews")
    #convert_jsonl_to_flair("D:/myDev/myNER/dataset/reviews")
    #convert_jsonl_to_FlairBIO("D:/myDev/myNER/dataset/reviews")
    #convert_jsonl_to_FlairBIO2("D:/myDev/myNER/dataset/reviews")
    train()
    #NerPredict()
    #fetch_reviews_from_rawfile()
    #parse()
    #detectUTF8('train.txt')