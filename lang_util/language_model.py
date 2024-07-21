import os
import pickle
import nltk
from functools import reduce

class LanguageModel:
    def __init__(self):
        module_dir = os.path.dirname(__file__)  # get current directory
        self.words = pickle.load(open(os.path.join(module_dir, 'model/_words.p'), 'rb'))
        self.freq_dist = pickle.load(open(os.path.join(module_dir, 'model/_freq_dist.p'), 'rb'))
        self.cond_freq_dist = pickle.load(open(os.path.join(module_dir, 'model/_cond_freq_dist.p'), 'rb'))
        self.cond_prob_dist = pickle.load(open(os.path.join(module_dir, 'model/_cond_prob_dist.p'), 'rb'))

    def unigram_prob(self, word):
        return self.freq_dist[word] / len(self.words)

    def sentence_prob(self, sentence):
        prob_list = [self.cond_prob_dist[a].prob(b) for (a,b) in nltk.bigrams(sentence.split())]
        return reduce(lambda x,y:x*y, prob_list)
