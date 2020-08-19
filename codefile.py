import pandas as pd
import numpy as np
import nltk
#from os import getcwd
#from utils import build_freqs

path = '/home/sandynote/Desktop/Hackathon/JantaHack4/train_2kmZucJ.csv'#sample_submission_LnhVWA4.csv
data = pd.read_csv(path)

from sklearn.model_selection import train_test_split
#nltk.download('stopwords')

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


# removing digits from tweets
for i in range(len(data['tweet'])):
    data['tweet'][i] = ''.join([j for j in data['tweet'][i] if not j.isdigit()])

# removing punctuation
for i in range(len(data['tweet'])):
    data['tweet'][i] = ''.join([j for j in data['tweet'][i] if j not in string.punctuation])


# removing url/hyperlinks form tweets
for i in range(len(data['tweet'])):
    data['tweet'][i] =  re.sub(r"http\S+", "", data['tweet'][i])
    
# removing # from tweets
for i in range(len(data['tweet'])):
    data['tweet'][i] =  re.sub(r'#', "", data['tweet'][i])

# instantiate tokenizer class   
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
# tokenize tweets
for i in range(len(data['tweet'])):
    data['tweet'][i] =  tokenizer.tokenize(data['tweet'][i])
    

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

data['tweet1']= ""
for i in range(len(data['tweet'])):
    data['tweet1'][i] = []
    for word in data['tweet'][i]:
        if (word not in stopwords_english and word not in string.punctuation):
           data['tweet1'][i].append(word)


# Instantiate stemming class
stemmer = PorterStemmer() 
data['tweet2'] = ""
for i in range(len(data['tweet1'])):
    data['tweet2'][i] = []
    for word in data['tweet1'][i]:
        stem_word = stemmer.stem(word)  # stemming word
        data['tweet2'][i].append(stem_word)  # append to the list


# removing digits from tweets
for i in range(len(data['tweet2'])):
    for word in data['tweet2'][i]:
        if word.isdigit():
            data['tweet2'][i].remove(word)
    
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    
    for y, tweet in zip(yslist, tweets):
        for word in tweet:
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
                
    return freqs

freqs = build_freqs(data['tweet2'], data['label'])


