import re 
import pandas as pd 
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize


data = pd.read_csv("Dev/INPUT.txt", sep='\t', header=None, 
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data.drop(['id', 'target'], axis=1, inplace=True)


# 1. Total number of tweets.
# 11906
print(data.shape[0])

# 2. Total number of characters.
# 1354375
num = 0
for tweet in data.values:
    num += len(tweet[0])
print(num)

# 3. Total number of distinct words (vocabulary)
# NLTK has a tweet tokenizer
# corpus includes hashtags, handles, punctuations (:, ..., ., :), and urls.
twtker = TweetTokenizer()
temp = []
for text in data.values:    
    for words in twtker.tokenize(text[0]):
        temp.append(words)

# NLTK word_tokenize api
# Doesnt remove punctuation, or urls.  
temp = []
for text in data.values:
    for words in word_tokenize(text[0]):
        temp.append(words)

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

temp = []
for text in data.values:
    for matches in my_extensible_tokenizer.findall(text[0]):
        if matches != '': # just in case get empty matches
            temp.append(matches)




