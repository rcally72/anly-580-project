import re 
import pandas as pd 
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer


data_dev = pd.read_csv("Gold/dev.txt", sep='\t', header=None, index_col=False,
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_dev.drop(['id'], axis=1, inplace=True)

data_train = pd.read_csv("Gold/train.txt", sep='\t', header=None, index_col=False,
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_train.drop(['id'], axis=1, inplace=True)

data_devtest = pd.read_csv("Gold/devtest.txt", sep='\t', header=None, index_col=False,
                           names=['id', 'target', 'tweet'], encoding='utf-8')
data_devtest.drop(['id'], axis=1, inplace=True)

data_test = pd.read_csv("Gold/test.txt", sep='\t', header=None, index_col=False,
                           names=['id', 'target', 'tweet'], encoding='utf-8')
data_test.drop(['id'], axis=1, inplace=True)


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


dev_tokens = [] # stores all features for dev dataset.  
for text in data_dev.values: # for each tweet.  
    temp = []  # stores the matches for a single tweet
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            temp.append(('contains(' + matches + ')', True))
        dev_tokens.append((dict(temp), text[0]))

train_tokens = [] # stores all features for train dataset.  
for text in data_train.values: # for each tweet.  
    temp = [] # stores tokens for a given tweet.  
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            temp.append(('contains(' + matches + ')', True))
        train_tokens.append((dict(temp), text[0]))


training_features = train_tokens + dev_tokens

sentiment_analyzer = SentimentAnalyzer()
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer=trainer, training_set=training_features)
    
# Evaluating model on training data.
sentiment_analyzer.evaluate(training_features, classifier)


devtest_tokens = [] # stores all features for devtest dataset.  
for text in data_devtest.values: # for each tweet.  
    temp = []  # stores the matches for a single tweet
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            temp.append(('contains(' + matches + ')', True))
        devtest_tokens.append((dict(temp), text[0]))

test_tokens = [] # stores all features for train dataset.  
for text in data_test.values: # for each tweet.  
    temp = [] # stores tokens for a given tweet.  
    for matches in my_extensible_tokenizer.findall(text[1]):
        # determine if matches is a url.  
        url_matches = url_tokenizer.findall(matches)
        # if the match is a url, then won't add to temp.  
        if not url_matches:
            temp.append(('contains(' + matches + ')', True))
        test_tokens.append((dict(temp), text[0]))

test_final = devtest_tokens + test_tokens

sentiment_analyzer.evaluate(test_final, classifier)












            