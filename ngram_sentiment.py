from __future__ import division
import nltk, re, pprint
from nltk.stem.lancaster import LancasterStemmer
from nltk.util import bigrams
import operator
import math

class SentimentAnalysisTest:
    def __init__(self):
        self.percent_training = 3800 / 5331
        self.train_words = {}
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

    def learn_line(self, line):
        raw_line, pos_neg = line
        message = self.get_ngram_tokens( raw_line )
        for word in message:
            if word not in self.train_words:
                self.train_words[word] = {'freq': 0, 'pos': 0, 'neg': 0 }
            self.train_words[word]['freq'] += 1
            if pos_neg < 0:
                self.train_words[word]['neg'] += 1
            else:
                self.train_words[word]['pos'] += 1
    
                

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
            test_tokens = self.get_ngram_tokens(line)

            matches = [x for x in test_tokens if self.train_words.has_key(x)]

            score = 0.0
            for word in matches:
                # similarity score to just-tested training element
                for word in matches:
                    if self.train_words[word]['pos'] > 0:
                        score += math.log(self.train_words[word]['pos'])
                    if self.train_words[word]['neg'] > 0:
                        score -= math.log(self.train_words[word]['neg'])

            if (score < 0 and label < 0) or (score > 0 and label > 0):
                correct += 1

        print "Correct {}%\n".format( 100 * correct / len(self.testing))

        """
            # sort by desc by score to move top matches to the top
            set_results.sort(reverse=True)

            # x[1] is the label
            best_matches = [x[1] for x in set_results[:5]]
            prediction = 1 if sum(best_matches) > 1 else -1

            if prediction is label:
                correct += 1

        print "Correct {}%\n".format( 100 * correct / len(self.testing))
        """



sentiment = SentimentAnalysisTest()

sentiment.load_positives(open(r'rt-polarity.pos', 'r').read())
sentiment.load_negatives(open(r'rt-polarity.neg', 'r').read())
sentiment.load_stops(open(r'stopwords.txt', 'r').read())
sentiment.learn()
sentiment.guess()



