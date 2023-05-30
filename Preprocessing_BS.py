
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string


bs_fake = pd.read_csv("./FakeNewsData/fake-news bad/fake.csv")


bs_fake["text"] = bs_fake["text"].str.lower()
bs_fake["text"] = bs_fake["text"].str.replace("\d+" , "")
bs_fake["text"] = bs_fake["text"].str.replace("[^\w\s]" , "")


#stop = set(stopwords.words('english'))


bs_fake['text'] = bs_fake['text'].fillna("")

#bs_fake['text'].isnull().sum()


stop = stopwords.words('english')

pat = r'\b(?:{})\b'.format('|'.join(stop))
#print(pat)
bs_fake['text'] = bs_fake['text'].str.replace(pat, '')

#bs_fake["text"].apply(lambda x: [item for item in x if item not in stop])

ps = PorterStemmer()

def stem_sentence(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

bs_fake["text"].apply(stem_sentence).head()

bs_fake["text"] = bs_fake.apply(lambda row : word_tokenize(row["text"]) , axis = 1)


#(bs_fake["type"] == "True").sum()


bs_dictionary = {"bias" : 0 , 
                 "conspiracy" : 1 , 
                 "fake" : 2 , 
                 "hate" : 3 , 
                 "junksci" : 4 , 
                 "satire" : 5 , 
                 "state" : 6}


bs_fake.groupby("type").size()


bs_test = bs_fake[bs_fake["type"] == "bs"]


bs_train = bs_fake[bs_fake["type"] != "bs"]


bs_train["type"] = bs_train["type"].map(bs_dictionary)
bs_fake["type"] = bs_fake["type"].map(bs_dictionary)

bs_fake.to_csv("bs_fake.csv")
bs_train.to_csv("bs_train.csv")
bs_test.to_csv("bs_test.csv")

