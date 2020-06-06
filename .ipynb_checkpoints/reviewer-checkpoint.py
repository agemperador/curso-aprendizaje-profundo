import re

def load( filename):
    with open( filename, 'r') as review_file:
        review_list = review_file.readlines()
    reviews = []
    for review in review_list:
        text, label = review.split('\t')
        reviews.append( ( re.findall( r'\w+', text.lower()) + ['.'], int(label) ) )
    return reviews


def vocabulary( reviews):
    words = set([ word for word_list, label in reviews for word in word_list ])
    wordlist = list(words)
    worddict = { word:index for index,word in enumerate(wordlist) }
    return wordlist, worddict


def sequence( reviews, worddict):
    indexed = []
    for words, label in reviews:
        seq = [ worddict[word] for word in words if word in worddict ]
        indexed.append( (seq, [label]) )
    return indexed
