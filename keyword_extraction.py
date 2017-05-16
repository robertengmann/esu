# -*- coding: utf-8 -*-
#!/usr/bin/env python
import re
import itertools
from nltk import pos_tag, sent_tokenize, word_tokenize, FreqDist, ne_chunk_sents
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


class KeywordExtractor(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def get_sentences(self, content):
        sentences = sent_tokenize(content)
        sentences = [word_tokenize(sent) for sent in sentences]
        sentences = [pos_tag(sent) for sent in sentences]

        return sentences


    def get_nouns(self, sentences):
        nouns = list(filter(lambda w: w[1][:2] == 'NN', itertools.chain.from_iterable(sentences)))
        
        if nouns:
            nouns = list(zip(*nouns))[0]

        return nouns


    def get_adjectives(self, sentences):
        adjectives = list(filter(lambda w: w[1][:2] == 'JJ', itertools.chain.from_iterable(sentences)))
        
        if adjectives:
            adjectives = list(zip(*adjectives))[0]

        return adjectives


    def get_named_entities(self, sentences):
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


    def get_bigram_keywords(self, nouns):
        bigram_keywords = []
        
        finder = BigramCollocationFinder.from_words(nouns)
        scored = finder.score_ngrams(BigramAssocMeasures.raw_freq)

        if scored:
            lowest_score = min([score for bigram, score in scored])
            bigram_keywords = [' '.join(bigram) for bigram, score in scored if score > lowest_score]

        return bigram_keywords


    def get_noun_keywords(self, nouns):
        frequency_distribution = FreqDist(nouns)
        frequency_limit = int(round(len(frequency_distribution) / 5))
        noun_keywords = [nk for nk in sorted(frequency_distribution, key=frequency_distribution.get, reverse=True)][:frequency_limit]

        return noun_keywords


    def get_ings(self, sentences):
        ings = list(filter(lambda w: w[1][:3] == 'VBG', itertools.chain.from_iterable(sentences)))
        ings = list(filter(lambda w: w[len(w)-3:] == 'ing', itertools.chain.from_iterable(ings)))
        ings = list(ings)

        return ings
        

    def clean_keyword(self, keyword):
        ck = keyword.lower().replace("’S", "")
        ck = ck.replace("'", "")
        ck = ck.replace("`", "")
        ck = ck.replace("-", "")
        ck = ck.replace("\\", "")
        
        return ck


    def extract_keywords(self):
        self.corpus = re.sub('<[^<]+?>', '', self.corpus)
        self.corpus = self.corpus.replace('&nbsp;', ' ')

        if not self.corpus:
            return []

        sentences = self.get_sentences(self.corpus)
        ings = self.get_ings(sentences)
        nouns = self.get_nouns(sentences)
        adjectives = self.get_adjectives(sentences)

        named_entities = self.get_named_entities(sentences)
        bigram_keywords = self.get_bigram_keywords(nouns)
        noun_keywords = self.get_noun_keywords(nouns)

        keywords = list(ings) + list(adjectives) + list(named_entities) + list(bigram_keywords) + list(noun_keywords) + list(named_entities)

        result_set = set([self.clean_keyword(k) for k in keywords])

        return result_set
