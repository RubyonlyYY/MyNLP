'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence, Corpus
from flair.embeddings import TransformerDocumentEmbeddings

from myGraduate import remove_other, remove_html, remove_urls
import re
from matplotlib import pyplot as plt
import numpy as np

import pymysql

def InsertData(TableName, dic):

    try:
        conn=pymysql.connect(host='localhost',password='1234',port=3306,user='root',charset='utf8')
        cur = conn.cursor()
        COLstr = ''
        ROWstr = ''

        ColumnStyle = ' VARCHAR(20)'
        for key in dic.keys():
            COLstr = COLstr + ' ' + key + ColumnStyle + ','
            ROWstr = (ROWstr + '"%s"' + ',') % (dic[key])

        try:
            cur.execute("SELECT * FROM  %s" % (TableName))
            cur.execute("INSERT INTO %s VALUES (%s)" % (TableName, ROWstr[:-1]))

        except pymysql.Error:
            cur.execute("CREATE TABLE %s (%s)" % (TableName, COLstr[:-1]))
            cur.execute("INSERT INTO %s VALUES (%s)" % (TableName, ROWstr[:-1]))
        conn.commit()
        cur.close()
        conn.close()
    except:

        print("未知异常")


def plot(pos):
    # 定义饼的标签，
    labels = ['Location', 'Amenity', 'Rating', 'Cuisine', 'Restaurant_Name', 'Hours', 'Dish', 'Price']

    # 每个标签所占的数量
    x = np.array([pos['Location'], pos['Amenity'], pos['Rating'], pos['Cuisine'], pos['Restaurant_Name'], pos['Hours'], pos['Dish'], pos['Price']])

    # 饼图分离
    explode = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)

    # 设置阴影效果
    # plt.pie(x,labels=labels,autopct='%3.2f%%',explode=explode,shadow=True)

    plt.pie(x, labels=labels, autopct='%3.2f%%', explode=explode)
    plt.legend()

    plt.show()



def plotStatistic(pos,neg):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 中文负号
    plt.rcParams['axes.unicode_minus'] = False

    # 设置分别率 为100
    plt.rcParams['figure.dpi'] = 100

    # 设置大小
    plt.rcParams['figure.figsize'] = (5, 3)
    # ================确定距离左侧========

    # 设置线条高度
    height = 0.2

    # 绘制图形:
    movie=['Location', 'Amenity', 'Rating', 'Cuisine', 'Restaurant_Name', 'Hours', 'Dish', 'Price']
    col1 = np.array([pos['Location'], pos['Amenity'], pos['Rating'], pos['Cuisine'], pos['Restaurant_Name'], pos['Hours'], pos['Dish'], pos['Price']])
    plt.barh(movie, col1, height=height, color='blue', label='正面')  # 第一天图形
    plt.legend()

    col2 = np.array([neg['Location'], neg['Amenity'], neg['Rating'], neg['Cuisine'], neg['Restaurant_Name'], neg['Hours'], neg['Dish'], neg['Price']])
    plt.barh(movie, col2, left=col1, height=height, color='red', label='负面')  # 第二天图形
    plt.legend()
    plt.show()
from matplotlib import pyplot as plt
import numpy as np

def training():
    # 1. get the corpus
    '''
    columns = {0: 'text', 2: 'ner'}
    data_folder = './dataset/semeval4'
    corpus: Corpus = ColumnCorpus(data_folder, columns,  train_file='Laptop_Train_v2.iob', dev_file='Laptops_Test_Data_phaseB.iob',
                                  test_file='test.txt')
    '''
    columns = {1: 'text', 0: 'ner'}
    data_folder = '.'
    corpus: Corpus = ColumnCorpus(data_folder, columns,  train_file='restauranttrain.bio', test_file='restauranttest.bio')
    print(corpus)
    print('[myLog]:label_dict ',len(corpus.train))
    print('[myLog]:label_dict --tag ',corpus.train[0].to_tagged_string('ner'))
    print('[myLog]:label_dict --tag ',corpus.train[1].to_tagged_string('ner'))
    print('[myLog]:label_dict --tag ',corpus.train[2].to_tagged_string('ner'))

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type) # delete add_unk=False by  see this defintion/line:1432 in data.py(flair ver0.11)
    print('[myLog]:label_dict ',label_dict)

    '''
    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'), #cache_dir='D:/myDev/myNER/huggingface_hub/news-forward-0.4.1.pt'),
        FlairEmbeddings('news-backward'), #cache_dir='D:/myDev/myNER/huggingface_hub/news-backward-0.4.1.pt'),
        #TransformerWordEmbeddings(model='roberta-base', cache_dir='D:/myDev/myNER/Lib/huggingface_hub/')
        #TransformerWordEmbeddings('bert-base-uncased',cache_dir='D:/myDev/myNER/Lib/huggingface_hub/', layers='-1', layer_mean=False)
    ]
    '''
    # initialize embeddings
    '''
    embedding_types: List[TokenEmbeddings] = [

        # GloVe embeddings
        WordEmbeddings('glove'),

        # contextual string embeddings, forward
        PooledFlairEmbeddings('news-forward', pooling='min'),

        # contextual string embeddings, backward
        PooledFlairEmbeddings('news-backward', pooling='min'),
    ]
    '''
    #embeddings = StackedEmbeddings(embeddings=embedding_types)

    #embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True) #, num_labels=408)

    # initialize embeddings
    embeddings = TransformerWordEmbeddings(
        model='distilbert-base-uncased',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,
        respect_document_boundaries=False,
    )

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=128,  #change 256 into 128 by yaoyuan
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            rnn_type='LSTM',
                            use_rnn=True,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/sota-ner-flair', # base_path=D:\myDev\myNER\Lib,myNER1 in it
                  learning_rate=0.1,
                  mini_batch_size=16, #change 32 into 2 by Yaoyuan
                  max_epochs=30) #change 500 into 2 by Yaoyuan

'''
    sentence1 = Sentence('Amaazing food! ')
    sentence5 = Sentence('The whole experience from start to finish is great')
    sentence2 = Sentence(' waitress is always so friendly and kind.')
    sentence3 = Sentence(' The food can’t get better and the prices are fair for the portion size.')
    sentence4 = Sentence(' Always a great spot to get great food.')
    #sentence2 = Sentence('The fajitas were great to taste, but not to see')
    tagger.predict(sentence1)
    print(sentence1)
    tagger.predict(sentence2)
    print(sentence2)
    tagger.predict(sentence3)
    print(sentence3)
    tagger.predict(sentence4)
    print(sentence5)
    tagger.predict(sentence5)
    #tagger.predict(sentence2)
    print(sentence4)

    classifierClass.predict(sentence1)
    print(sentence1.labels)
    classifierClass.predict(sentence2)
    print(sentence2.labels)
    classifierClass.predict(sentence3)
    print(sentence3.labels)
    classifierClass.predict(sentence4)
    print(sentence4.labels)
'''
def sentimentResult():
    # load tagger
    tagger = SequenceTagger.load("resources/taggers/sota-ner-flair/log5/best-model.pt")
    classifierClass = TextClassifier.load('resources/taggers/question-classification-with-transformer/final-model.pt')

    negWordCount= {'Location': 0, 'Amenity': 0, 'Rating': 0, 'Cuisine': 0, 'Restaurant_Name': 0, 'Hours': 0, 'Dish': 0, 'Price': 0}
    posWordCount = {'Location': 0, 'Amenity': 0, 'Rating': 0, 'Cuisine': 0, 'Restaurant_Name': 0, 'Hours': 0, 'Dish': 0, 'Price': 0}

    dataset = pd.read_csv('C:/Users/10068/myNLP/myproject/resources/taggers/sota-ner-flair/Restaurant_Reviews.csv')
    reviews = dataset["Review"]
    print("yaoyuan : begin")
    i=0
    for text in reviews:
        sentence = Sentence(text)

        tagger.predict(sentence)
        classifierClass.predict(sentence)
        print(sentence)
        l = re.findall('positive', str(sentence.labels))

        if len(l) == 0 :
            for entity in sentence.get_spans('ner'):
                # print(entity.tag)
                negWordCount[entity.tag] = negWordCount[entity.tag] + 1
        else:
            for entity in sentence.get_spans('ner'):
                # print(entity.tag)
                posWordCount[entity.tag] = posWordCount[entity.tag] + 1
        print(sentence)
        i=i+1
        if i%20 == 0:
            input()

    print(posWordCount)
    print(negWordCount)
    #plot(posWordCount)
    #plotStatistic(posWordCount,negWordCount)

    df_list1 = [posWordCount]
    df1 = pd.DataFrame.from_dict(df_list1)

    #print(df)

    df_list2 = [posWordCount]
    df2= pd.DataFrame.from_dict(df_list2)

    result = pd.concat([df1, df2])

    # 将数据保存到 CSV 文件中
    result.to_csv('sentimentResult.csv', index=True, encoding='utf-8')



    # print predicted NER spans
    #print('The following NER tags are found:')
    #print('The fajitas were great to taste, but not to see')
    # iterate over entities and print
    #for entity in sentence1.get_spans('ner'):
    #    WordCount[entity.tag]= WordCount[entity.tag]+ 1
    #    print(WordCount[entity.tag])
    #print('2')
    #for entity in sentence2.get_spans('ner'):
        #print(entity)
    #print(sentence1)



print('[myLog]:start training......... ')
#training()
#print('[myLog]:start predict......... ')
sentimentResult()


