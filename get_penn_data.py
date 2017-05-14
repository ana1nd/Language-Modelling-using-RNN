# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:49:01 2017

@author: anand
"""

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import matplotlib.pyplot as plt

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "START"
sentence_end_token = "END"
vocabulary_size = 8000

def function_get_data():
    print "Reading CSV file..."
    with open('data/ptb.train.txt', 'rb') as f:
        f = open('data/ptb.train.txt','rb')
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
        
    #replace <unk> token with unknown_token
    temp = [w.replace('<unk>',unknown_token) for w in sentences]        
    sentences= temp
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())
    
    vocab = word_freq.most_common(vocabulary_size-1)
    
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
    
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
    
#    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
#    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
#
#    # Print an training data example
#    x_example, y_example = X_train[17], y_train[17]
#    print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
#    print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)
    
    return index_to_word,word_to_index,tokenized_sentences
    
    
if __name__ == '__main__':
    main()