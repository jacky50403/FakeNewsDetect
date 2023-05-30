#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string


# In[2]:


column_name = ['ID' , 'Trust_value' , 'Statement' , 'Subject' , 'Speaker',
               'Speaker_Job' , 'State_Info' , 'Party_Affiliation' , 
               'Barely_True_Count' , 'False_Count' , 'Half_True_Count' , 'Mostly_True_Count' ,
               'Pants_On_Fire_Count' , 'Context']


# In[3]:


train_tsv = pd.read_csv('./Dataset/Liar/train.tsv' , sep = '\t' , names = column_name)
test_tsv = pd.read_csv('./Dataset/Liar/test.tsv' , sep = '\t' , names = column_name)


# In[4]:


ps = PorterStemmer()


# In[5]:


#train data
train_tsv['Statement'] = train_tsv['Statement'].str.lower() # turn into lower
train_tsv['Statement'] = train_tsv['Statement'].str.replace('[^\w\s]' ,'') #remove punctuation
train_tsv['Statement'] = train_tsv['Statement'].str.replace('\d+' , '') #remove numbers

#test data
test_tsv['Statement'] = test_tsv['Statement'].str.lower() # turn into lower
test_tsv['Statement'] = test_tsv['Statement'].str.replace('[^\w\s]' ,'') #remove punctuation
test_tsv['Statement'] = test_tsv['Statement'].str.replace('\d+' , '') #remove numbers


# In[6]:


ps = PorterStemmer()
#porter stemming
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

train_tsv['Stemmed_Statement'] = train_tsv['Statement'].apply(stem_sentences)
test_tsv['Stemmed_Statement'] = test_tsv['Statement'].apply(stem_sentences)


# In[7]:


train_tsv['Statement_Tokenize'] = train_tsv.apply(lambda row : word_tokenize(row['Stemmed_Statement']), axis = 1)
test_tsv['Statement_Tokenize'] = test_tsv.apply(lambda row : word_tokenize(row['Stemmed_Statement']), axis = 1)


# In[12]:


Trust_Score= {'true' : 1 , 'mostly-true' : 0.8 , 'half-true' : 0.5 , 'barely-true':0.3 , 'false' : 0.1 , 'pants-fire': 0}
train_tsv['Trust_Score'] = train_tsv['Trust_value'].map(Trust_Score)
test_tsv['Trust_Score'] = test_tsv['Trust_value'].map(Trust_Score)


# In[13]:


train_tsv.to_csv('./Dataset/Liar/Preprocessed_train.csv')
test_tsv.to_csv('./Dataset/Liar/Preprocessed_test.csv')

