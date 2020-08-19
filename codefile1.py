import pandas as pd
import numpy as np
import nltk
import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

#from sklearn.model_selection import train_test_split

path = '/home/sandynote/Desktop/Hackathon/JantaHack4/train_2kmZucJ.csv'#sample_submission_LnhVWA4.csv
data = pd.read_csv(path)

x_train = data[:5500]
x_test = data[5500:]

## prepossing on x_train dataset

# removing digits from tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] = ''.join([j for j in x_train['tweet'][i] if not j.isdigit()])
    
# removing punctuation
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] = ''.join([j for j in x_train['tweet'][i] if j not in string.punctuation])


# removing url/hyperlinks form tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  re.sub(r"http\S+", "", x_train['tweet'][i])
    
# removing # from tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  re.sub(r'#', "", x_train['tweet'][i])

# instantiate tokenizer class   
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
# tokenize tweets
for i in range(len(x_train['tweet'])):
    x_train['tweet'][i] =  tokenizer.tokenize(x_train['tweet'][i])
    

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

x_train['tweet1']= ""
for i in range(len(x_train['tweet'])):
    x_train['tweet1'][i] = []
    for word in x_train['tweet'][i]:
        if (word not in stopwords_english and word not in string.punctuation):
           x_train['tweet1'][i].append(word)


# Instantiate stemming class
stemmer = PorterStemmer() 
x_train['tweet2'] = ""
for i in range(len(x_train['tweet1'])):
    x_train['tweet2'][i] = []
    for word in x_train['tweet1'][i]:
        stem_word = stemmer.stem(word)  # stemming word
        x_train['tweet2'][i].append(stem_word)  # append to the list


## prepossing on x_test dataset
x_test.reset_index(inplace = True)
# removing digits from tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] = ''.join([j for j in x_test['tweet'][i] if not j.isdigit()])
    
# removing punctuation
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] = ''.join([j for j in x_test['tweet'][i] if j not in string.punctuation])


# removing url/hyperlinks form tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  re.sub(r"http\S+", "", x_test['tweet'][i])
    
# removing # from tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  re.sub(r'#', "", x_test['tweet'][i])

# instantiate tokenizer class   
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
# tokenize tweets
for i in range(len(x_test['tweet'])):
    x_test['tweet'][i] =  tokenizer.tokenize(x_test['tweet'][i])
    

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

x_test['tweet1']= ""
for i in range(len(x_test['tweet'])):
    x_test['tweet1'][i] = []
    for word in x_test['tweet'][i]:
        if (word not in stopwords_english and word not in string.punctuation):
           x_test['tweet1'][i].append(word)


# Instantiate stemming class
stemmer = PorterStemmer() 
x_test['tweet2'] = ""
for i in range(len(x_test['tweet1'])):
    x_test['tweet2'][i] = []
    for word in x_test['tweet1'][i]:
        stem_word = stemmer.stem(word)  # stemming word
        x_test['tweet2'][i].append(stem_word)  # append to the list



    
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

freqs_train = build_freqs(x_train['tweet2'], x_train['label'])
freqs_test = build_freqs(x_test['tweet2'], x_test['label'])




def extract_features(tweet, freqs):
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    #bias term is set to 1
    x[0,0] = 1 
    # loop through each word in the list of words
    for word in tweet:
        # increment the word count for the positive label 1
        if (word, 1) in freqs.keys():
            x[0,1] += freqs[(word,1)]
        # increment the word count for the negative label 0
        if (word, 0) in freqs.keys():
            x[0,2] += freqs[(word,0)]
        
    assert(x.shape == (1, 3))
    return x

# collect the features 'x' and stack them into a matrix 'X'
X_train = np.zeros((len(x_train['tweet2']), 3))
for i in range(len(x_train['tweet2'])):
    X_train[i, :]= extract_features(x_train['tweet2'][i], freqs_train)

Y_train = np.array(x_train['label']).reshape((len(x_train['label']), 1))

#X_test = np.zeros((len(x_test['tweet2']), 3))
#for i in range(len(x_test['tweet2'])):
#    X_test[i, :]= extract_features(x_test['tweet2'][i], freqs_train)

Y_test = np.array(x_test['label']).reshape((len(x_test['label']), 1))






def sigmoid(z): 
    h = 1 / (1 + np.exp(-z))
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)
    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)
        # get the sigmoid of z
        h = sigmoid(z)
        # calculate the cost function
        J = (-1/m)*(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1 - h)))
#        J = sum(J)
#        J = float(np.squeeze(J))                                    
#        assert(isinstance(J, float))
        # update the weights theta
        theta = theta - (alpha / m) * (np.dot(x.T, (h-y)))
        
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, J))
        
    J = float(J)
    return J, theta

theta = np.zeros((3, 1))#np.random.randn(3, 1)*10
alpha = 1e-8
iterations = 50000 

tmp_J, tmp_theta = gradientDescent(X_train, Y_train, theta, alpha, iterations)




def predict_tweet(tweet, freqs, theta):
    # extract the features of the tweet and store it into x
#    T = np.zeros((len(tweet), 3))
#    for i in range(len(tweet)):
#        T[i, :]= extract_features(tweet[i], freqs)
    
    T = extract_features(tweet, freqs)
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(T, theta))
    
    return y_pred


def test_logistic_regression(test_x, test_y, freqs, theta):
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # append 1.0 to th list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = sum(sum(np.array(y_hat) == np.array(test_y.T)))/len(y_hat)

    
    return accuracy

acc_train = test_logistic_regression(x_train['tweet2'], Y_train, freqs_train, tmp_theta)
acc_test = test_logistic_regression(x_test['tweet2'], Y_test, freqs_test, tmp_theta)

print(acc_train)
print(acc_test)