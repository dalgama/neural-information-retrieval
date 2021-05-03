# Main imports.
import string
import re
import json
import nltk
import math

# Import specific packages.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download packages if not installed locally.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stem.porter')

# Initialize stemmer
ps = PorterStemmer()

# Remove stop words
with open('../assets/stop_words.txt', 'r') as f:
	stopWords = [line.strip() for line in f]

def isNumeric(subj):
    '''
    Check if a string contains numerical values.

    :param str subj: the string to be converted
    :return: True if a subject string is a number, False otherwise.
    :rtype: boolean
    '''
    try:
        return float(subj)
    except Exception:
        return False

def importTweets(verbose = False):
    '''
    Import tweets from collection.

    :param boolean verbose: [Optional] Provide printed output of tokens for testing.
    :return: the tokenized list of queries.
    :rtype: list
    '''
    tweet_list = dict()
    # Splits tweet list at newline character.
    tweets = (line.strip('\n') for line in open('../assets/tweet_list.txt', 'r', encoding='utf-8-sig'))

    # Build the dictionary.
    for tweet in tweets:
        key, value = tweet.split('\t')
        # Tokenize each tweet, and put back in list.
        tweet_list[key] = filterSentence(value)

    return tweet_list


def importQuery(verbose = False):
    '''
    Import query from collection.

    :param boolean verbose: [Optional] Provide printed output of tokens for testing.
    :return: the tokenized list of queries.
    :rtype: list
    '''
    query_list = dict()

    with open('../assets/test_queries.txt', 'r') as file:
        fileContents = file.read()

    queryCheck = fileContents.strip('\n').split('\n\n')

    current_tweet = 1
    for x in queryCheck:
        save = x[x.index('<title>'): x.index('</title>')].strip('<title> ')
        query_list[current_tweet] = filterSentence(save)
        current_tweet+=1

    return query_list

def filterSentence(sen):
	'''
	:param list of sentences: list of sentences from the queries or documents.
	:return: the the input list with filtered sentences.
	:rt string
	'''

    # Removing html tags
	TAG_RE = re.compile(r'<[^>]+>')
	sentence = TAG_RE.sub('', sen)

	sentWords = sentence.split()
	nonStopwords  = [word for word in sentWords if word.lower() not in stopWords]
	sentence = ' '.join(nonStopwords)

	# Remove links
	sentence = re.sub(r'http\S+', '', sentence)

    # Remove numbers
	sentence = re.sub(r'[0-9]', '', sentence)

	# Remove punctuation
	sentence = re.sub(r'[^\w\s]', '', sentence)

    # Single character removal
	sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
	sentence = re.sub(r'\s+', ' ', sentence)

	# Create tokens and Stem the words.
	tokens = [ps.stem(word.lower()) for word in word_tokenize(sentence)]
	sent = ""

	# traverse in the string
	for token in tokens:
		sent += token + " "

	# return string
	return sent
