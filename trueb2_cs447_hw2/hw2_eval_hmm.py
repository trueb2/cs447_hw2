from collections import defaultdict

import sys

from hmm.hw2_hmm import HMM


class Eval:
    """
    A class for evaluating POS-tagged data
    """

    def __init__(self, goldFile, testFile):
        """
        Reads the labeled corpus at goldFile and the unlabeled corpus at testFile
        :param goldFile: file with the labeled test corpus
        :param testFile: file with the unlabeled test corpus
        """
        # Read the labeled evaluation corpora
        self.pred_corpus = HMM.read_labeled_data(testFile)
        self.gold_corpus = HMM.read_labeled_data(goldFile)

    def getTokenAccuracy(self):
        """
        Compares the labels of the computed corpus against the gold corpus
        :return: float: token accuracy over all observations
        """
        correct = 0
        total = 0
        assert (len(self.pred_corpus) == len(self.gold_corpus))
        for i in range(len(self.pred_corpus)):
            predicted = self.pred_corpus[i]
            actual = self.gold_corpus[i]
            assert (len(predicted) == len(actual))
            for j in range(len(predicted)):
                ptw = predicted[j]
                atw = actual[j]
                assert (ptw.word == atw.word)
                total += 1
                if ptw.tag == atw.tag:
                    correct += 1
        return correct / total

    def getSentenceAccuracy(self):
        """
        Calculates the percentage of sentences that are tagged completely correctly (all words)
        :return: accuracy of predicting entire sentence correctly
        """
        correct = 0
        total = 0
        assert (len(self.pred_corpus) == len(self.gold_corpus))
        for i in range(len(self.pred_corpus)):
            predicted = self.pred_corpus[i]
            actual = self.gold_corpus[i]
            assert (len(predicted) == len(actual))
            sentence_correct = True
            total += 1
            for j in range(len(predicted)):
                ptw = predicted[j]
                atw = actual[j]
                assert (ptw.word == atw.word)
                if ptw.tag != atw.tag:
                    sentence_correct = False
            if sentence_correct:
                correct += 1
        return correct / total

    def writeConfusionMatrix(self, outFile):
        """
        Creates a confusion matrix for the tags then pretty prints it to a file
        :param outFile: location to write confusion matrix
        """
        # Count all of the ways a tag has been tagged
        confusion_dict = defaultdict(lambda: defaultdict(int))
        tags = set()
        assert (len(self.pred_corpus) == len(self.gold_corpus))
        for i in range(len(self.pred_corpus)):
            predicted = self.pred_corpus[i]
            actual = self.gold_corpus[i]
            assert (len(predicted) == len(actual))
            for j in range(len(predicted)):
                ptw = predicted[j]
                atw = actual[j]
                assert (ptw.word == atw.word)
                tags.add(ptw.tag)
                tags.add(atw.tag)
                confusion_dict[atw.tag][ptw.tag] += 1

        # Assign each tag an arbitrary unique id
        ti = {}
        for i, t0 in enumerate(tags):
            ti[i] = t0
        n = len(ti)

        # Print each tag in ordered by its id
        out = "{0:6}".format("")
        for i in range(n):
            out += "{0:>6}".format(ti[i])
        out += "\t\n"

        # Print each row by the tag
        for i in range(n):
            out += "{0:6}".format(ti[i])
            for j in range(n):
                out += "{0:6}".format(confusion_dict[ti[i]][ti[j]])
            out += "\t\n"

        # Write to file
        f = open(outFile, "w+")
        f.write(out)
        f.close()

    def getPrecision(self, tagTi):
        """
        Computes the precision of the HMM model for tagTi
        Precision is true positive divided by all positive predictions
        :param tagTi: tag to check for precision
        :return: precision
        """
        tp = 0
        fp = 0
        assert (len(self.pred_corpus) == len(self.gold_corpus))
        for i in range(len(self.pred_corpus)):
            predicted = self.pred_corpus[i]
            actual = self.gold_corpus[i]
            assert (len(predicted) == len(actual))
            for j in range(len(predicted)):
                ptw = predicted[j]
                atw = actual[j]
                assert (ptw.word == atw.word)
                if ptw.tag == tagTi:
                    if atw.tag == tagTi:
                        tp += 1
                    else:
                        fp += 1

        return tp / (tp + fp)

    def getRecall(self, tagTj):
        """
        Computes the recall of the HMM model for the tagTj
        Recall is the true positive divided by the number of actual positive
        :param tagTj: tag to check for recall
        :return: recall
        """
        tp = 0
        ap = 0
        assert (len(self.pred_corpus) == len(self.gold_corpus))
        for i in range(len(self.pred_corpus)):
            predicted = self.pred_corpus[i]
            actual = self.gold_corpus[i]
            assert (len(predicted) == len(actual))
            for j in range(len(predicted)):
                ptw = predicted[j]
                atw = actual[j]
                assert (ptw.word == atw.word)
                if ptw.tag == tagTj and atw.tag == tagTj:
                    tp += 1
                elif atw.tag == tagTj:
                    ap += 1
        return tp / (tp + ap)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        gold = "../data/gold.txt"
        test = "../data/out.txt"
    else:
        # Pass in the gold and test POS-tagged data as arguments
        gold = sys.argv[1]
        test = sys.argv[2]

    # You need to implement the evaluation class
    eval = Eval(gold, test)
    # Calculate accuracy (sentence and token level)
    print("Token accuracy: ", eval.getTokenAccuracy())
    print("Sentence accuracy: ", eval.getSentenceAccuracy())
    # Calculate recall and precision
    print("Recall on tag NNP: ", eval.getPrecision('NNP'))
    print("Precision for tag NNP: ", eval.getRecall('NNP'))
    # Write a confusion matrix
    eval.writeConfusionMatrix("conf_matrix.txt")
