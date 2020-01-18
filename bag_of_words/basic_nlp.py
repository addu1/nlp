# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 19:56:56 2020

@author: adyal
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#nltk.download()

##read labeled training dataset
training = pd.read_csv('labeledTrainData.tsv', header=0, delimiter ="\t", quoting=3)
##shape of dataset
#print(training.shape)


def words_from_review(inputs):
    ##beautifulsoup up
    #print(training["review"][0])
    soup1 = BeautifulSoup(inputs)
    #print(soup1.get_text())
    
    ##remove punctuatuions, numbers, special chars, etc. using Regex
    chars_only = re.sub("[^a-zA-Z]",
                    " ",
                    soup1.get_text())
    #print(chars_only)
    
    ##convert to lower case
    chars_only = chars_only.lower()
    #print(chars_only)
    
    ##tokenize
    words = chars_only.split()
    #print(words)
    
    ##remove stop words using nltk stopwords list
    stop_words = set(stopwords.words("English"))
    words = [w for w in words if not w in stop_words]
    #print(words)
    
    #join words into one string
    return " ".join(words)

#testing our function
print(words_from_review(training["review"][0]))

#get cleaned reviews from dataset
review_count = training["review"].size
clean_list = []

for i in range(0,review_count):
    if ((i+100)%100 == 0):
        print("review", i, "in process")
    clean_list.append(words_from_review(training["review"][i]))
    if (i == review_count-1):
        print('Completed Successfully!')

#bag of words, get features using sklearn vectorizer
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
training_features = vectorizer.fit_transform(clean_list)
training_features= training_features.toarray()

#words in training features
vocab = vectorizer.get_feature_names()

#count of each word from vocab list
noc = np.sum(training_features, axis=0)

for tag,count in zip(vocab, noc):
    print(tag, count)

randomForestClassifier = RandomForestClassifier(n_estimators = 100)
randomForestClassifier.fit(training_features, training["sentiment"])

#applying on test data
testing = pd.read_csv('testData.tsv', header=0, delimiter ="\t", quoting=3)
test_count = testing["review"].size
clean_test_list = []

for i in range(0, test_count):
    if(i%100 == 0):
        print(i,"review under process")
    clean_test_list.append(words_from_review(testing["review"][i]))

test_data_features = vectorizer.transform(clean_test_list)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = randomForestClassifier.predict(test_data_features)

output = pd.DataFrame( data={"id":testing["id"], "sentiment":result} )



