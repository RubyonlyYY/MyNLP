'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

# importing all the required Libraries
#import glob
#import json
#import csv
import pandas as pd
import numpy as np
import re
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from wordcloud import WordCloud
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import sent_tokenize, word_tokenize
#from textblob import TextBlob
#from textblob.sentiments import NaiveBayesAnalyzer
#import string
#import matplotlib.pyplot as plt
#from nltk.stem import PorterStemmer


'''
def clean_data(df):

    pd.options.mode.chained_assignment = None

    print("******Cleaning Started*****")

    print(f'Shape of df before cleaning : {df.shape}')
    df['review_date'] = pd.to_datetime(df['review_date'])
    df = df[df['review_body'].notna()]
    df['review_body'] = df['review_body'].str.replace("<br />", " ")
    df['review_body'] = df['review_body'].str.replace("\[?\[.+?\]?\]", " ")
    df['review_body'] = df['review_body'].str.replace("\/{3,}", " ")
    df['review_body'] = df['review_body'].str.replace("\&\#.+\&\#\d+?;", " ")
    df['review_body'] = df['review_body'].str.replace("\d+\&\#\d+?;", " ")
    df['review_body'] = df['review_body'].str.replace("\&\#\d+?;", " ")

    #facial expressions
    df['review_body'] = df['review_body'].str.replace("\:\|", "")
    df['review_body'] = df['review_body'].str.replace("\:\)", "")
    df['review_body'] = df['review_body'].str.replace("\:\(", "")
    df['review_body'] = df['review_body'].str.replace("\:\/", "")

    #replace multiple spaces with single space
    df['review_body'] = df['review_body'].str.replace("\s{2,}", " ")

    df['review_body'] = df['review_body'].str.lower()
    print(f'Shape of df after cleaning : {df.shape}')
    print("******Cleaning Ended*****")


    return(df)
'''
# VADER sentiment analysis tool for getting pos, neg and neu.
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'

def create_Word_Corpus(df):
    words_corpus = ''
    for val in df["Summary"]:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = stemming(tokens)
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus
def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordclouds.png', facecolor='k', bbox_inches='tight')

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|@[^\s]+')
    return url_pattern.sub(r'', text)

def fetch_reviews_to_file():
    '''
    把test_data目录下的文件绝对路径保存到csv文件中，同时把文件名中的label也保存下来
    保存两列  filename， label
    '''
    dataset=pd.read_csv('D:/myDev/myNER/dataset/hair_dryer.tsv', sep='\t')
    #reviews = dataset[['review_body']] #读取某列
    reviews = dataset["review_body"].apply(lambda text: remove_urls(text))
    reviews.to_csv(r'D:/myDev/myNER/dataset/reviews.tsv',sep="\t")

fetch_reviews_to_file()


#dataset=pd.DataFrame(reviews,columns=['Reviewer_ID','Asin','Reviewer_Name','helpful_UpVote','Total_Votes','Review_Text','Rating','Summary','Unix_Review_Time','Review_Time'])
#Selected_Rows=dataset.head(100000)
#Selected_Rows['Sentiment_Score']=Selected_Rows['Review_Text'].apply(lambda x: sentimental_Score(x))
#pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))
#neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))
#plot_Cloud(pos_wordcloud)
#plot_Cloud(neg_wordcloud)


