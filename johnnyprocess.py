from __future__ import division
import nltk, re, pprint
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import bigrams
import operator
import math


#open positive raw trainging file
#raw_lines_pos = open(r'../lesson3/rt-polarity.pos', 'r').read().splitlines()
#raw_lines_neg = open(r'../lesson3/rt-polarity.neg', 'r').read().splitlines()
#
## convert the training set to nltk text
#train_size = 4800
#
#train_pos = raw_lines_pos[train_size:]
#train_neg = raw_lines_neg[train_size:]
#
#test_pos = raw_lines_pos[:train_size]
#test_neg = raw_lines_neg[:train_size]
#
#training = [(x,1) for x in train_pos] + [(x,-1) for x in train_neg]
#testing = [(x,1) for x in test_pos] + [(x,-1) for x in test_neg]
#
#freq = {}
#training_features = []

class SentimentAnalysisTest:
    def __init__(self):
        self.percent_training = 1000 / 5331
        self.freq = {}
        self.training_features = []
        self.stemmer = nltk.stem.lancaster.LancasterStemmer()
        self.training = []
        self.testing = []
        self.stops = []

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

    def learn_line(self, line):
        raw_line, score = line
        message = self.get_ngram_tokens( raw_line )
        for word in message:
            self.freq[word] = self.freq.get(word, 0) + 1

        self.training_features.append((message, score))

    def get_ngram_tokens(self, line):
        tokens = nltk.wordpunct_tokenize(line)
        message = [self.stemmer.stem(x) for x in tokens if len(x) > 2 and x not in self.stops]
        bigram = bigrams(message)
        for pair in bigram:
            joined = " ".join(pair)
            message.append(joined)
        return list(set(message))


    def learn(self):
        for line in self.training:
            self.learn_line(line)
    
    def guess(self):
        train_size = len(self.training)
        correct = 0

        for line, label in self.testing:
            testtokens = self.get_ngram_tokens(line)
            set_results = []

            for trainwords, trainlabel in self.training_features:
                matches = [x for x in trainwords if x in testtokens]

                # similarity score to just-tested training element
                score = 0.0
                for word in matches:
                    score += math.log(train_size / self.freq[word])

                set_results.append((score, trainlabel))

            # sort by desc by score to move top matches to the top
            set_results.sort(reverse=True)

            # x[1] is the label
            best_matches = [x[1] for x in set_results[:5]]
            prediction = 1 if sum(best_matches) > 1 else -1

            if prediction is label:
                correct += 1

        print "Correct {}%\n".format( 100 * correct / len(self.testing))



sentiment = SentimentAnalysisTest()

sentiment.load_positives(open(r'rt-polarity.pos', 'r').read())
sentiment.load_negatives(open(r'rt-polarity.neg', 'r').read())
sentiment.load_stops(open(r'stopwords.txt', 'r').read())
sentiment.learn()
sentiment.guess()



