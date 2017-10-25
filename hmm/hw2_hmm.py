import os.path
import sys
import numpy as np

# Unknown word token
from collections import defaultdict

UNK = 'UNK'


# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]


# Class definition for a bigram HMM
class HMM:
    @staticmethod
    def read_labeled_data(inputFile):
        """
        Reads a labeled data file
        :param inputFile: labeled data file to be read
        :return: nested list of sentences, where each sentence is a list of TaggedWord objects
        """
        if os.path.isfile(inputFile):
            # open the input file in read-only mode
            file = open(inputFile, "r")
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                # append this list as an element to the list of sentences
                sens.append(sentence)
            return sens
        else:
            # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            print("Error: unlabeled data file %s does not exist" % inputFile)
            sys.exit()  # exit the script

    @staticmethod
    def read_unlabeled_data(inputFile):
        """
        Reads an unlabeled data file
        :param inputFile: unlabeled data file to be read
        :return: a nested list of sentences, where each sentence is a list of strings
        """
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = []
            for line in file:
                sentence = line.split()  # split the line into a list of words
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)
            sys.exit()  # exit the script

    def __init__(self, unknown_word_threshold=5):
        # Unknown word threshold, default value is 5 (words occurring fewer than 5 times should be treated as UNK)
        self.minFreq = unknown_word_threshold

        # The tag transition probability matrix as a 2D array
        self.A = np.ones(1)

        # The word emission probability matrix as a 2D array
        self.B = np.ones(1)
        self.ti = {}  # The map of tag to row index in A and B
        self.wi = {}  # The map of word to col index in B

        # The initial state probability distribution as a dict
        self.pi = np.ones(1)

    def train(self, trainFile):
        """
        Builds the HMM distributions from observed counts in the corpus located in trainFile.
        Using the training corpus as the corpus, A, the transition probability matrix, B, the emission probability
        matrix, and pi, the initial state probability dict, are populated from the distribution in the corpus.
        :param trainFile: the file with the corpus
        """

        # Read the corpus
        data = self.read_labeled_data(trainFile)

        # Count the number of times each word and each tag appears
        wc = defaultdict(lambda: defaultdict(int))
        tc = defaultdict(int)
        pi = defaultdict(float)
        count_key = "__COUNT__"
        for sen in data:
            pi[sen[0].tag] += 1
            for tw in sen:
                wc[tw.word][count_key] += 1
                wc[tw.word][tw.tag] += 1
                tc[tw.tag] += 1

        # Compute the probabilities from the counts for the initial states
        num_sen = len(data)
        for t in pi:
            pi[t] /= num_sen

        # Consolidate infrequent words
        unkc = defaultdict(int)
        rare = []
        for w in wc:
            c = wc[w]
            if c[count_key] < self.minFreq:
                for t in c:
                    unkc[t] += c[t]
                del unkc[count_key]
                rare.append(w)
            del wc[w][count_key]
        wc[UNK] = unkc
        for rw in rare:
            del wc[rw]

        # Count the word emissions for each tag
        self.B = np.ones((len(tc), len(wc)))
        self.ti = {tag: i for i, tag in enumerate(tc)}
        self.wi = {word: j for j, word in enumerate(wc)}
        m, n = self.B.shape
        for w in wc:
            word_tags = wc[w]
            j = self.wi[w]
            for t in word_tags:
                i = self.ti[t]
                self.B[i, j] += word_tags[t]

        # Compute the smoothed emission probabilities as a trellis
        for i in range(m):
            self.B[i, :] /= sum(self.B[i, :])

        # Count the tag bigrams
        self.A = np.ones((m, m))
        i = self.ti["."]
        for sen in data:
            for tw in sen:
                j = self.ti[tw.tag]
                self.A[i, j] += 1
                i = j

        # Compute the smoothed transition probabilities
        for i in range(m):
            self.A[i, :] /= self.A[i, :]

        # Organize initial states probabilities by tag index
        self.pi = np.zeros(m)
        for t in pi:
            self.pi[self.ti[t]] = pi[t]

    def test(self, testFile, outFile):
        """
        Output the Viterbi tag sequences as a labeled corpus from an unlabeled corpus
        :param testFile: The unlabeled corpus to label using Viterbi
        :param outFile: The file to write the labelled corpus in
        """
        data = self.read_unlabeled_data(testFile)
        f = open(outFile, 'w+')
        for sen in data:
            vit_tags = self.viterbi(sen)
            sen_string = ''
            for i in range(len(sen)):
                sen_string += sen[i] + "_" + vit_tags[i] + " "
            print(sen_string)
            f.write(sen_string.rstrip() + "\n")

    def viterbi(self, words):
        """
        Runs the Viterbi algorithm on a list of words, generating a sequence of tags with the highest probability
        according to the provided list of words.
        :param words: list of string words
        :return: tagged_words: a list of tags according the trained HMM
        """
        # Replace unknown words
        for i in range(len(words)):
            if not words[i] in self.wi:
                words[i] = UNK

        # Initialize structures
        m, n = self.B.shape
        n = len(words)
        path = np.zeros((m, n), dtype=int)
        viterbi = np.zeros((m, n))

        # Initialize first column using starting probabilities from pi
        viterbi[:, 0] = self.pi * self.B[:, self.wi[words[0]]]

        # Iterate over remaining columns
        for i in range(1, len(words)):
            for t in range(m):
                for t_ in range(m):
                    tmp = viterbi[t, i - 1] * self.A[t, t_]
                    if tmp > viterbi[t, i]:
                        viterbi[t, i] = tmp
                        path[t, i] = t_
                viterbi[t, i] *= self.B[t, self.wi[words[i]]]

        # Find the best cell in the last column
        t_max = 0
        vit_max = 0
        for t in range(m):
            if viterbi[t, n - 1] > vit_max:
                t_max = t
                vit_max = viterbi[t, n - 1]

        # Unpack the path of tags through the trellis
        i = n - 1
        tags = []
        while i >= 0:
            tag = [t for t in self.ti if self.ti[t] == t_max][0]
            tags.append(tag)
            t_max = path[t_max, i]
            i -= 1

        return list(reversed(tags))


if __name__ == "__main__":
    tagger = HMM()
    tagger.train('../data/train.txt')
    tagger.test('../data/test.txt', '../data/out.txt')
