from __future__ import division
import nltk, re, pprint
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import bigrams
import operator
import math

class SentimentAnalysisTest:
    def __init__(self):
        self.percent_training = 3800 / 5331
        self.train_lines = []
        self.test_lines = []
        self.stemmer = nltk.stem.lancaster.LancasterStemmer()
        self.training = []
        self.testing = []
        self.stops = []
        self.pos_count = 0
        self.neg_count = 0

    def load(self, filedata, score):
        lines = filedata.splitlines()
        train_size = int(len(lines) * self.percent_training)
        train = lines[train_size:]
        test = lines[:train_size]
        self.training = self.training + [(x, score) for x in train]
        self.testing = self.testing + [(x, score) for x in test]

    def load_negatives(self, filedata ):
        self.load(filedata, -1)

    def load_positives(self, filedata ):
        self.load(filedata, 1)

    def load_stops(self, filedata ):
        self.stops = filedata.splitlines()

    def proc_line(self, line):
        raw_line, pos_neg = line
        message = self.get_ngram_tokens( raw_line )
        new_line = {}
        for word in message:
            new_line[word] = True
        return (new_line, pos_neg)

                

    def get_ngram_tokens(self, line):
        tokens = nltk.wordpunct_tokenize(line)
        message = [self.stemmer.stem(x) for x in tokens if len(x) > 2 and x not in self.stops]
        #bigram = bigrams(message)
        #for pair in bigram:
        #    joined = " ".join(pair)
        #    message.append(joined)
        return list(set(message))


    def learn(self):
        for line in self.training:
            new_line, pos_neg = self.proc_line(line)
            self.train_lines.append((new_line, "pos" if pos_neg > 0 else "neg"))

        self.classifier = nltk.classify.NaiveBayesClassifier.train(self.train_lines)


    def guess(self):
        for line in self.testing:
            new_line, pos_neg = self.proc_line(line)
            self.test_lines.append((new_line, "pos" if pos_neg > 0 else "neg"))

        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.test_lines)
        self.classifier.show_most_informative_features()    



sentiment = SentimentAnalysisTest()

sentiment.load_positives(open(r'rt-polarity.pos', 'r').read())
sentiment.load_negatives(open(r'rt-polarity.neg', 'r').read())
sentiment.load_stops(open(r'stopwords.txt', 'r').read())
sentiment.learn()
sentiment.guess()



