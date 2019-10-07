#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:23:07 2019
@author: mashagubenko
"""

import pandas as pd 
import textstat
from scipy import mean
import nltk
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
import re 
import numpy as np

import seaborn as sns

#Read in data
data = pd.read_csv("INPUT_2.txt", sep='\t', header=None, 
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data.drop(['id', 'target'], axis=1, inplace=True)

#Total number of tweets
print("Total number of tweets is ", len(data))

#Total number of characters
char = 0
i = 0
while i < len(data):
    char = char + textstat.char_count(data['tweet'][i])
    i += 1
print("Total number of characters is ", char)

#Total number of distinct words
twt = TweetTokenizer(strip_handles=True, reduce_len=True)
words = []
i = 0
while i < len(data):
    for word in twt.tokenize(data['tweet'][i]):
        words.append(word)
    i += 1

print("Total number of distinct words is ", len(set(words)))

#Avg number of of characters in each tweet
char = []
i = 0
while i < len(data):
    char.append(textstat.char_count(data['tweet'][i]))
    i += 1
print("Total number of characters is ", mean(char))

###### Norman's Tokenizer

# Does not ignore emojis
# A somewhat better way to write a tokenizer with multiple
# regular expressions is in this snippet below (shorter version of 
# http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py)
# The order is important (match from first to last)
#### Figure out how to not match urls and remove the # from hashtags. 
regexes=(
# Keep usernames together (any token starting with @, followed by A-Z, a-z, 0-9)        
r"(?:@[\w_]+)",

# Keep hashtags together (any token starting with #, followed by A-Z, a-z, 0-9, _, or -)
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",

# abbreviations, e.g. U.S.A.
r'(?:[A-Z]\.)+',
r'[A-Za-z]\.(?:[A-Za-z0-9]\.)+',
r'[A-Z][bcdfghj-np-tvxz]+\.',

# URL, e.g. https://google.com
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9]+\.[^\s]{2,}',
r'www\.[a-zA-Z0-9]+\.[^\s]{2,}',

# currency and percentages, e.g. $12.40, 82%
r'\$?\d+(?:\.\d+)?%?',

# Numbers i.e. 123,56.34
r'(?:[0-9]+[,]?)+(?:[.][0-9]+)?',

# Keep words with apostrophes, hyphens and underscores together
r"(?:[a-z][a-z’'\-_]+[a-z])",

# Keep all other sequences of A-Z, a-z, 0-9, _ together
r"(?:[\w_]+)",

# Match words at the end of a sentence.  e.g. tree. or tree!
r'(?:[a-z]+(?=[.!\?]))',

# Everything else that's not whitespace
# It seems like this captures punctuations and emojis and emoticons.  
#r"(?:\S)"
)

big_regex="|".join(regexes)

# Note re.I for performing Perform case-insensitive matching; 
# expressions like [A-Z] will match lowercase letters, too. 
my_extensible_tokenizer = re.compile(big_regex, re.VERBOSE | re.I | re.UNICODE)


#####After tokenization

#Avg number and sd of characters per token

chartok = []
i = 0
while i < len(data):
    for word in my_extensible_tokenizer.findall(data['tweet'][i]):
        chartok.append(textstat.char_count(word))
    i += 1
    
print("Avg number of characters per token is ", mean(chartok))
print("Sd of characters per token is ", np.std(chartok))
#sns.distplot(chartok) #Hist showing the spread of the data

#The total number of tokens corresponding to the top 10 most frequent words (types) in the vocabulary

#Tagging all of the words
tagword = []
i = 0
while i < len(data):
    for word in nltk.pos_tag(my_extensible_tokenizer.findall(data['tweet'][i])):
        tagword.append(word)
    i += 1

#Saving the tags separately 
tags = []
while i < len(tagword):
    tags.append(tagword[i][1])
    i +=1

#Counting the number of words associated with each tag
i = 0
j = 0
wordtagcount = []
while j < len(set(tags)):
    count = 0
    i = 0
    while i < len(tagword):
        if tagword[i][1] == tags[j]:
            count +=1        
        i += 1
        
    wordtagcount.append(count)
    j += 1

a = list(set(tags))
tagd = {'Tags':a, 'Word Count': wordtagcount}
tagdf = pd.DataFrame(tagd)
tag_by_wrd = tagdf.sort_values('Word Count',ascending=False) #sorting by the number of words      

a = sum(tag_by_wrd['Word Count'][0:10])  #adding up the number of words associated with top 10tags
print("Total number of words corresponding to top 10 tags is ", a)

#The token/type ratio in the dataset

#The total number of distinct n-grams (of words) that appear in the dataset for n=2,3,4,5.
def GramCount(data,n):
    i = 0
    ngramlist = [] 
    while i < len(data):
        tokens = my_extensible_tokenizer.findall(data['tweet'][i])
        grams = list(ngrams(tokens, n)) 
        for gram in grams:
            ngramlist.append(gram)
        i += 1
    return(len(set(ngramlist)))

for i in range(2,6):    
      print("There are ", GramCount(data,i), " of distinct ", i,"-grams.")
      
#The total number of distinct n-grams of characters that appear for n=2,3,4,5,6,7

#If spaces do not count as characters
n = 2
def CharGramCount(data,n):
    i = 0
    wordgram = []
    while i < len(data):
        for word in my_extensible_tokenizer.findall(data['tweet'][i]):
            for j in range(0,len(word)-(n-1)):
                word = word.lower()
                wordgram.append(word[i:i+n])
        i += 1
    return(len(set(wordgram)))

for i in range(2,7):    
      print("There are ", CharGramCount(data,i), " of distinct ", i,"-grams of characters.")

#If spaces count as characters
def SpCharGramCount(data,n):
    i = 0
    ngramlist = [] 
    while i < len(data):
        tokens = data['tweet'][i].lower()
        grams = list(ngrams(tokens, n)) 
        for gram in grams:
            ngramlist.append(gram)
        i += 1
    return(len(set(ngramlist)))

#####Gold 

#Read in data
gold = pd.read_csv("train.txt", sep='\t', header=None, 
                   names=['id', 'class', 'tweet'], encoding='utf-8')
#gold.drop(['id', 'target'], axis=1, inplace=True)

#Number of types in Gold but not in Dev
 
#Takes in a dataset and returns the set of tags that is used in the set
def TagSet(data):
    tagword = []
    
    for tweet in data['tweet']:
        for word in nltk.pos_tag(my_extensible_tokenizer.findall(tweet)):
            tagword.append(word)
    
    
    #Saving the tags separately 
    tags = []
    i = 0
    while i < len(tagword):
        tags.append(tagword[i][1])
        i +=1
    
    return(set(tags))

a = TagSet(data) #set of unique tags in the dev set
b = TagSet(gold) #set of unique tags in the gold set

num = []
count = 0 #counting the number of tags in gold but not in dev

for tag in b:
    for tag2 in a:
        if tag2 == tag:
            num.append(1)
    if sum(num) == 0:
        count += 1

print(count)

#Vocab growth?

#Class distribution of training set

uniqgold = set(gold['class']) #unique values of the class column 

numuniq = []
for value in uniqgold:
    count = 0
    for value2 in gold['class']:
        if value2 == value:
            count += 1
    numuniq.append(count)

uniqlist = {'Class':list(uniqgold), 'Number of Tweets': numuniq}
uniqdf = pd.DataFrame(uniqlist)
print(uniqdf)       

#Distribution of word types by class
for value in uniqgold: 
    print(value)
    print(TagSet(gold[gold['class'] == value]))

        
            