# based on http://norvig.com/spell-correct.html
import os
import pickle
import string

from . import language_model as model

class SpellCorrector:

    def __init__(self):
        module_dir = os.path.dirname(__file__) # get current directory
        self.words = pickle.load(open(os.path.join(module_dir, 'model/_spell_words.p'), "rb"))
        self.counter = pickle.load(open(os.path.join(module_dir, 'model/_spell_counter.p'), "rb"))
        self.model = model.LanguageModel()

        self.candidates_dict = {}

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        if self.candidates_dict.get(word):
            return self.candidates_dict[word]
        else:
            candidates = (self.__known([word]) | self.__known(self.__edits1(word)) | self.__known(self.__edits2(word)) | self.__known(self.__edits3(word)) | {word})

            # cache it
            self.candidates_dict[word] = candidates
            return candidates

    def __known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.counter)

    def __edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'aiueon'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(inserts)

    def __edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.__edits1(word) for e2 in self.__edits1(e1))

    def __edits3(self, word):
        return (e3 for e1 in self.__edits1(word) for e2 in self.__edits1(e1) for e3 in self.__edits1(e2))

    def __trigram_interpolation(self, w1, w2, w3):
        lambda1 = 0.75
        lambda2 = 0.20
        lambda3 = 0.05
        return (lambda1 * self.model.sentence_prob(w1 +' '+ w2 +' '+ w3)) + (lambda2 * self.model.sentence_prob(w2 +' '+ w3)) + (lambda3 * self.model.unigram_prob(w3))

    def __bigram_interpolation(self, w1, w2):
        lambda1 = 0.80
        lambda2 = 0.20
        return (lambda1 * self.model.sentence_prob(w1 +' '+ w2)) + (lambda2 * self.model.unigram_prob(w2))

    def validate(self, sentence):
        # The method translate() returns a copy of the string in which all characters have been translated
        # using table (constructed with the maketrans() function in the str module),
        # optionally deleting all characters found in the string deletechars.
        #translator = str.maketrans({key: ' ' for key in string.punctuation})
        #words = [token.translate(translator).strip() for token in sentence.lower().split()]
        words = sentence #' '.join(words)
        words = [x.strip().lower() for x in words.split() if x.strip()] # Hapus seluruh empty char pada list

        valid = []
        for idx, word in enumerate(words):
            if word in self.words:
                valid.append(word.lower())
            else:
                candidates = self.candidates(word.lower())
                if  idx == 0 :
                    max_word = max([w for w in candidates], key=lambda word: self.model.unigram_prob(word))
                    valid.append(max_word)

                elif idx == 1:
                    max_word = max([w for w in candidates], key=lambda word: self.__bigram_interpolation(valid[0], word))
                    valid.append(max_word)

                else:
                    max_word = max([w for w in candidates], key=lambda word: self.__trigram_interpolation(valid[idx - 2], valid[idx - 1], word))
                    valid.append(max_word)

        return ' '.join(valid)
