import re 
import pandas as pd 
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

data_dev = pd.read_csv("Gold/dev.txt", sep='\t', header=None, 
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_dev.drop(['id',], axis=1, inplace=True)

data_train = pd.read_csv("Gold/train.txt", sep='\t', header=None, 
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_train.drop(['id',], axis=1, inplace=True)



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
big_regex = "|".join(regexes)
my_extensible_tokenizer = re.compile(big_regex, re.VERBOSE | re.I | re.UNICODE)


url_pattern = (
# URL, e.g. https://google.com
# This pattern will match any url.  
r'(https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9]+\.[^\s]{2,}',
r'www\.[a-zA-Z0-9]+\.[^\s]{2,})',
)
big_url_pattern = "|".join(url_pattern)
url_tokenizer = re.compile(big_url_pattern, re.VERBOSE | re.I | re.UNICODE)


dev_tokens = []
for text in data_dev.values:
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            dev_tokens.append(({'contains(' + matches + ')': True}, text[0]))

train_tokens = []
for text in data_train.values:
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            train_tokens.append(({'contains(' + matches + ')':True}, text[0]))
            
training_features = train_tokens + dev_tokens

sentiment_analyzer = SentimentAnalyzer()
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_features)









            