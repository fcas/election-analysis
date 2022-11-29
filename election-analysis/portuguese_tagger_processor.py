import os.path
import pickle

import nltk.test
from nltk.corpus import floresta
from nltk.tag import UnigramTagger
from nltk.tokenize import TweetTokenizer

__tagger_path = "static_files/portuguese_tagger.pickle"
__tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


def __create_tagger():
    train_floresta_sents = floresta.tagged_sents()
    mac_morpho_sents = nltk.corpus.mac_morpho.tagged_sents()
    tagger = UnigramTagger(mac_morpho_sents)
    bigram_tagger = nltk.BigramTagger(train_floresta_sents, backoff=tagger)
    return bigram_tagger


def __save_tagger(tagger):
    file = open('static_files/portuguese_tagger.pickle', 'wb')
    pickle.dump(tagger, file)
    file.close()


def get_tagger():
    if os.path.isfile(__tagger_path) is False:
        __save_tagger(__create_tagger())
    file = open('static_files/portuguese_tagger.pickle', 'rb')
    return pickle.load(file)


def get_tweet_tokenizer():
    return __tweet_tokenizer
