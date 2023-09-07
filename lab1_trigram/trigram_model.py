import sys
from collections import defaultdict
import math
import random
import os
import os.path
import copy
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer

Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """    
    tmp = copy.deepcopy(sequence)
    for i in range(max(n-1, 1)):
      tmp.insert(0, "START")
    
    tmp.append("STOP")

    res = []
    for i in range(len(tmp)-n+1):
      s = ()
      for j in range(n):
        s = s + (tmp[i+j], )
      res.append(s)
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_uni = 0
        self.count_ngrams(generator)
        

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = defaultdict(lambda: 0) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(lambda: 0)
        self.trigramcounts = defaultdict(lambda: 0)
        ##Your code here

        self.count_uni = 0
        for sentence in corpus:
            unigram = get_ngrams(sentence, 1)
            for i in range(len(unigram)):
                self.unigramcounts[unigram[i]] += 1
                if unigram[i] != 'START':
                    self.count_uni += 1

            bigram = get_ngrams(sentence, 2)
            for i in range(len(bigram)):
                self.bigramcounts[bigram[i]] += 1

            trigram = get_ngrams(sentence, 3)
            for i in range(len(trigram)):
                self.trigramcounts[trigram[i]] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        u = trigram[0]
        v = trigram[1]
        de = self.bigramcounts[(u, v)]
        if de == 0:
            return self.raw_unigram_probability((trigram[2]))
        return self.trigramcounts[trigram]/self.bigramcounts[(u, v)]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        u = bigram[0]
        de = self.unigramcounts[(u, )]
        if de == 0:
            return self.raw_unigram_probability((bigram[1]))
        return self.bigramcounts[bigram]/self.unigramcounts[(u, )]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        return self.unigramcounts[(unigram[0],)]/float(self.count_uni)

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        u, v, w = trigram[0], trigram[1], trigram[2]

        p_tri = lambda1 * self.raw_trigram_probability(trigram) + \
                lambda2 * self.raw_bigram_probability((v, w)) + \
                lambda3 * self.raw_unigram_probability((w,))


        return p_tri
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        p_sen = 0.0
        for trigram in trigrams:
            p_tri = self.smoothed_trigram_probability(trigram)
            p_tri = math.log2(p_tri)
            p_sen += p_tri

        return p_sen

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            M += len(sentence) +1

        l = l/float(M)
        return math.pow(2, -l) 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        for f in os.listdir(testdir1):
            total += 1
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2: #classified as class 1, correct
                correct += 1

    
        for f in os.listdir(testdir2):
            total += 1
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp2 < pp1: #classified as class 2, correct
                correct += 1
        
        return correct/float(total)

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    #print(acc)