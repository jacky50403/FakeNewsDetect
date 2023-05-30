import os
import pandas as pd
import re
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

pd.set_option('display.width', 5000)


TRAINING_PATH = './Dataset/Kaggle_Fake_News/train.csv'
TESTING_PATH = './Dataset/Kaggle_Fake_News/test.csv'

column_name =["id", "title", "author", "text"]


trainFile = pd.read_csv(TRAINING_PATH)

trainX = trainFile.loc[:,["id", "title", "author", "text"]]

trainX.text = trainX.text.str.lower()


stop = stopwords.words('english')

pat = r'\b(?:{})\b'.format('|'.join(stop))
#print(pat)
trainX['text'] = trainX['text'].str.replace(pat, '')
#trainX['text'] = trainX['text'].str.replace(r'\s+', ' ')
trainX['text'] = trainX['text'].str.replace(r'[^\w\s]+', '')

ps = PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

trainX['text'] = trainX['text'].apply(stemSentence)   # fucking fail


trainY = trainFile.loc[:,["label"]]

trainX_df = pd.DataFrame(trainX, columns=["id", "title", "author", "text"])

trainX_df.to_pickle('./someSaveThing/trainX.pkl')

trainY_df = pd.DataFrame(trainY, columns=["id", "title", "author", "text"])

trainY_df.to_pickle('./someSaveThing/trainY.pkl')


tXdf = pd.read_pickle("./someSaveThing/trainX.pkl")
tYdf = pd.read_pickle("./someSaveThing/trainY.pkl")

corpus = tXdf.text#.sample(frac=1)
print(corpus)