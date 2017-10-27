import time
import os.path
import sys
from collections import defaultdict
import numpy as np
import heapq

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """
    Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r")  # open the input file in read-only mode
        i = 0  # this is just a counter to keep track of the sentence numbers
        corpus = []  # this will become a list of sentences
        print("Reading file %s ..." % f)
        for line in file:
            i += 1
            sentence = line.split()  # split the line into a list of words
            corpus.append(sentence)  # append this list as an element to the list of sentences
        return corpus
    else:
        # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        print("Error: corpus file %s does not exist" % f)
        sys.exit()  # exit the script


class PMI:
    def __init__(self, corpus):
        """
        Given a corpus of sentences, store observations so that PMI can be calculated efficiently
        :param corpus: list of list of strings
        """
        self.corpus = corpus

        # How many tokens does the corpus contain
        self.N = len(corpus)

        # How often does w occur in the corpus
        self.W = defaultdict(int)
        for sen in corpus:
            for w in sen:
                self.W[w] += 1

        # How often does w occur with c in its window (window == sentence here)
        self.WC = defaultdict(lambda: defaultdict(int))
        for s in corpus:
            for w in s:
                for c in s:
                    if w != c:
                        self.WC[w][c] += 1

        print("Created and trained PMI instance")

    def getPMI(self, w1, w2):
        """
        Gets the pointwise mutual information based on sentence co-occurrence frequency for w1 and w2
        :param w1: string word
        :param w2: string word
        :return: PMI
        """
        p_w1 = self.W[w1]
        p_w2 = self.W[w2]
        if p_w1 == 0 or p_w2 == 0:
            raise Exception("Unseen word %s, %s" % ((w1, p_w1), (w2, p_w2)))

        # log_2 ( p(w,c) / (p(w) * p(c)) without calculating p(...) = f(...)/N
        p_w1_w2 = self.WC[w1][w2]
        return np.log2(self.N * p_w1_w2 / p_w1 / p_w2) if p_w1_w2 > 0 else -np.inf

    def getVocabulary(self, k):
        """
        Get a list of observed words that appear in at least k sentences
        :param k: number of sentences that the words must appear in
        :return: a list of k frequent words
        """
        words = [w for w in self.W if self.W[w] >= k]
        print("Vocabulary aggregated with size of %d" % (len(words)))
        return words

    def getPairsWithMaximumPMI(self, words, N):
        """
        Given a list of words, return a list of the pairs of words that have the highest PMI
        without repeated pairs, and without duplicate pairs (wi,wj) and (wj,wi) are considered the same pair.
        Each entry in the list should be a triple (pmiValue, w1, w2), where pmiValue is the PMI of the pair of words
        (w1,w2)
        :param words: list of words to consider
        :param N: number of words to return
        :return: list of pairs of words with the highest PMI as 3 tuple
        """
        # Generate word pairs and calculate their PMI on the fly
        heap = []
        words = sorted(list(set(words)))
        for i in range(len(words)):
            if i % 500 == 0 and i != 0:
                print("Evaluating word pair number ", i)
            w1 = words[i]
            for j in range(i+1, len(words)):
                w2 = words[j]
                if w2 not in self.WC[w1]:
                    continue
                if len(heap) < N:
                    heapq.heappush(heap, (self.getPMI(w1,w2), w1, w2))
                else:
                    heapq.heappushpop(heap, (self.getPMI(w1,w2),w1,w2))

        return list(reversed(sorted(heap)))

    def writePairsToFile(self, numPairs, wordPairs, filename):
        """
        Writes the first numPairs entries in the list of wordPairs to a file, along with each pair's PMI
        :param numPairs: int
        :param wordPairs: list of tuples
        :param filename: string
        :return:
        """
        f = open(filename, 'w+')
        count = 0
        for (pmiValue, wi, wj) in wordPairs:
            if count > numPairs:
                break
            count += 1
            print("%f %s %s" % (pmiValue, wi, wj), end="\n", file=f)

    def pair(self, w1, w2):
        """
        Given two words, returns the words as a sorted tuple where w1 <= w2
        :param w1: string
        :param w2: string
        :return: sorted word pair
        """
        return min(w1, w2), max(w1, w2)


# -------------------------------------------
# The main routine
# -------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('./movies.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("luke", "vader")
    print("  PMI of \"luke\" and \"vader\": %f" % lv_pmi)
    numPairs = 100
    k = 200
    s = time.time()
    for k in [2, 5, 10, 50, 100, 200]:
        print("Getting vocabulary for k=%d" % k)
        commonWords = pmi.getVocabulary(k)  # words must appear in least k sentences
        wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
        print("Got greatest PMI word pairs")
        pmi.writePairsToFile(numPairs, wordPairsWithGreatestPMI, "pairs_minFreq=%d.txt" % k)
    e = time.time()
    print("Generation over all k values took ", e - s)
