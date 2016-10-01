# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
import itertools
from nltk import pos_tag, sent_tokenize, word_tokenize, FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def get_sentences(content):
    sentences = sent_tokenize(content)
    sentences = [word_tokenize(sent) for sent in sentences]
    sentences = [pos_tag(sent) for sent in sentences]

    return sentences


def get_nouns(sentences):
    nouns = list(filter(lambda w: w[1][:2] == 'NN', itertools.chain.from_iterable(sentences)))
    nouns = list(zip(*nouns))[0]

    return nouns


def get_bigram_keywords(nouns):
    finder = BigramCollocationFinder.from_words(nouns)
    scored = finder.score_ngrams(BigramAssocMeasures.raw_freq)
    lowest_score = min([score for bigram, score in scored])
    bigram_keywords = [' '.join(bigram) for bigram, score in scored if score > lowest_score]

    return bigram_keywords


def get_noun_keywords(nouns):
    frequency_distribution = FreqDist(nouns)
    frequency_limit = int(round(len(frequency_distribution) / 10))
    noun_keywords = [nk for nk in sorted(frequency_distribution, key=frequency_distribution.get, reverse=True)][:frequency_limit]

    return noun_keywords


def main(argv):
    corpus = open(argv[0], 'r').read()

    sentences = get_sentences(corpus)
    nouns = get_nouns(sentences)

    bigram_keywords = get_bigram_keywords(nouns)
    noun_keywords = get_noun_keywords(nouns)

    print('\n'.join(bigram_keywords + noun_keywords))

if __name__ == "__main__":
    main(sys.argv[1:])
