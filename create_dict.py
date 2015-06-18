import bz2
import logging
import os
import pickle
import numpy

from collections import Counter
from operator import add
from numpy.lib.stride_tricks import as_strided
import twitter

def create_dictionary():
    # Part I: Counting the words
    global_counter = Counter()
    CONSUMER_KEY = 'aSktbHDRI2kwmGzpnNgpEeTdW'
    CONSUMER_SECRET = 'Db3CN2J0rsH07E1Z9iDwvWPO5CJhsXstH601zfwAGpv1d9JN8Z'
    ACCESS_TOKEN_KEY = '3247038151-AWcCCGmjGQ0hfWNJZzpyBebJee1LQhgrURuS70j'
    ACCESS_TOKEN_SECRET = 'IG7bN7kU2EOS3MzqrgP3BG6mZ0mVMLkQuRhR0DTFybRWj'
    api = twitter.Api(consumer_key=CONSUMER_KEY,consumer_secret=CONSUMER_SECRET,access_token_key=ACCESS_TOKEN_KEY,access_token_secret=ACCESS_TOKEN_SECRET)

    users = load_obj('users')
    users = users[0:10]
    for j in range(len(users)):
        statuses = api.GetUserTimeline(user_id = users[j].id,count=200)
        for i in range(len(statuses)):
            line = statuses[i].text
            line = line.lower()
            words = None
            words = line.strip().split(' ')
            global_counter.update(words)

    # Part II: Combining the counts
    combined_counter = global_counter

    # Part III: Creating the dictionary
    num_words = min(5003,len(combined_counter)+3)
    vocab_count = combined_counter.most_common(num_words - 3)
    vocab = {'UNK': 1, '<s>': 0, '</s>': 0,'LINK': 2}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 3
    print len(vocab)
    return vocab


def binarize(sentence):
    words = list(sentence.lower().strip().split(' '))
    print words
    binarized_sentence = [vocab.get(word, 1) for word in words]
    for i in range(len(binarized_sentence)):
        if(binarized_sentence[i]==1):
            t = words[i]
            print t[0:7]
            if(t[0:7]=='http://'):
                binarized_sentence[i]=2

    binarized_sentence.append(0)
    return binarized_sentence

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    #vocab = create_dictionary()
    #save_obj(vocab,'vocab')
    users = load_obj('users')
    vocab = load_obj('vocab')
    print binarize('Hi how are you http://')
    #print combined_counter
    #print vocab
