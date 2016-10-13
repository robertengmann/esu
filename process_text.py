# -*- coding: utf-8 -*-
#!/usr/bin/env python

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

import sys
import itertools
from nltk import pos_tag, sent_tokenize, word_tokenize, FreqDist, ne_chunk_sents
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


def get_named_entities(sentences):
    def traverse(tree):
        entity_names = []

        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entity_names.extend(traverse(child))

        return entity_names

    named_entities = []

    for chunk in ne_chunk_sents(sentences, binary=True):
        for entity in sorted(list(set([word for tree in chunk for word in traverse(tree)]))):
            named_entities.append(entity)

    return named_entities


def get_bigram_keywords(nouns):
    finder = BigramCollocationFinder.from_words(nouns)
    scored = finder.score_ngrams(BigramAssocMeasures.raw_freq)
    lowest_score = min([score for bigram, score in scored])
    bigram_keywords = [' '.join(bigram) for bigram, score in scored if score > lowest_score]

    return bigram_keywords


def get_noun_keywords(nouns):
    frequency_distribution = FreqDist(nouns)
    frequency_limit = int(round(len(frequency_distribution) / 5))
    noun_keywords = [nk for nk in sorted(frequency_distribution, key=frequency_distribution.get, reverse=True)][:frequency_limit]

    return noun_keywords


def get_ings(sentences):
    ings = list(filter(lambda w: w[1][:3] == 'VBG', itertools.chain.from_iterable(sentences)))
    ings = list(filter(lambda w: w[len(w)-3:] == 'ing', itertools.chain.from_iterable(ings)))
    
    ings = list(ings)

    return ings
    

def clean_keyword(keyword):
    return keyword.title().replace("’S", "")


def main(argv):
    corpus = open(argv[0], 'r').read()

    sentences = get_sentences(corpus)
    print(sentences)
    ings = get_ings(sentences)
    nouns = get_nouns(sentences)

    named_entities = get_named_entities(sentences)
    bigram_keywords = get_bigram_keywords(nouns)
    noun_keywords = get_noun_keywords(nouns)
    result_set = set([clean_keyword(k) for k in list(ings + named_entities + bigram_keywords + noun_keywords)])

    print('\n'.join(result_set))

if __name__ == "__main__":
    main(sys.argv[1:])
