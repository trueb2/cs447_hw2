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
        self.K = 0  # Number of states
        self.N = 0  # Number of observations
        self.A = np.zeros(0)  # transition matrix, k * k
        self.B = np.zeros(0)  # emission matrix, k * n
        self.PI = np.zeros(0)  # initial state probabilities
        self.state_index = {}  # maps index to state tag
        self.reverse_observation_index = {}  # maps word observation to index

    def train(self, trainFile):
        """
        Builds the HMM distributions from observed counts in the corpus located in trainFile.
        Using the training corpus as the corpus, A, the transition probability matrix, B, the emission probability
        matrix, and pi, the initial state probability dict, are populated from the distribution in the corpus.
        :param trainFile: the file with the corpus
        """
        # Read the corpus
        data = self.read_labeled_data(trainFile)

        # Count tag states and word observations in labeled corpus
        a = defaultdict(lambda: defaultdict(int))
        b = defaultdict(lambda: defaultdict(int))
        p = defaultdict(int)
        cw = defaultdict(int)
        t = "."
        for sen in data:
            p[sen[0].tag] += 1
            for tw in sen:
                a[t][tw.tag] += 1  # Count tag bigrams
                b[tw.word][tw.tag] += 1  # Count tag word emission
                cw[tw.word] += 1  # Count the number  of occurrences of word
                t = tw.tag  # Track previous tag

        # Collect the tags for rare words and consolidate them in b under UNK
        ut = defaultdict(int)
        for w in cw:
            if cw[w] < self.minFreq:
                for t in b[w]:
                    ut[t] += b[w][t]
                del b[w]
        b[UNK] = ut

        # Set the size constants and allocate space for probability matrices
        self.K = len(a)
        self.N = len(b)
        self.A = np.zeros((self.K, self.K))
        self.B = np.zeros((self.K, self.N))
        self.PI = np.zeros(self.K)

        # Generate the indices for states and observations
        reverse_state_index = {}
        for i, t in enumerate(a):
            reverse_state_index[t] = i
            self.state_index[i] = t

        for i, w in enumerate(b):
            self.reverse_observation_index[w] = i

        # Compute the initial state probabilities
        for t0 in p:
            self.PI[reverse_state_index[t0]] += p[t0]

        # Laplacian smoothing
        # self.PI += 1
        self.PI /= sum(self.PI)
        self.PI = np.log(self.PI)

        # Compute the transition probabilities
        for t0 in a:
            i = reverse_state_index[t0]
            for t1 in a[t0]:
                j = reverse_state_index[t1]
                self.A[i, j] += a[t0][t1]

        # Laplacian smoothing
        self.A += 1
        for i in range(self.K):
            self.A[i, :] /= sum(self.A[i, :])
        self.A = np.log(self.A)

        # Compute the emission probabilities
        for w in b:
            j = self.reverse_observation_index[w]
            for t in b[w]:
                i = reverse_state_index[t]
                self.B[i, j] += b[w][t]

        # Laplacian smoothing
        # self.B += 1
        for i in range(self.K):
            self.B[i, :] /= sum(self.B[i, :])
        self.B = np.log(self.B)


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
        # Create the memoization structures
        T1 = np.zeros((self.K, len(words)))
        T2 = np.zeros_like(T1, dtype=int)

        # Replace unknown words and map to indices in HMM structures
        w = list(words)
        for i in range(len(w)):
            if w[i] not in self.reverse_observation_index:
                w[i] = self.reverse_observation_index[UNK]
            else:
                w[i] = self.reverse_observation_index[w[i]]

        # Initialize the first column using the initial state probability
        T1[:, 0] = self.PI + self.B[:, w[0]]
        T2[:, 0] = 0

        # Iterate over the columns of T1 and T2
        for i in range(1, len(w)):
            wi = w[i]  # Column index of word in B
            for j in range(self.K):
                p = T1[:, i - 1] + self.A[:, j]  # State probability * transition probability of previous col
                k = np.argmax(p)  # State with max path probability in previous col
                T1[j, i] = self.B[j, wi] + p[k]  # Emission probability * path probability
                T2[j, i] = k  # Path index

        # Retrace path through memoization structure from highest probability end state
        X = []
        z = np.argmax(T1[:, -1])
        X.append(z)
        for i in range(len(w) - 1, 0, -1):
            z = T2[z, i]
            X.append(z)

        # Return the tags for the states
        return [self.state_index[x] for x in reversed(X)]


if __name__ == "__main__":
    tagger = HMM()
    tagger.train('../data/train.txt')
    tagger.test('../data/test.txt', '../data/out.txt')
