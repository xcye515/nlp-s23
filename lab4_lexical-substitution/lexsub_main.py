#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wnpyt
from nltk.corpus import stopwords

import numpy as np
import tensorflow
import string

import gensim
import transformers
from collections import defaultdict

from typing import List

"""
All the functions and classes were implemented by Xingchen (Estella) Ye.
"""

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    synsets = wnpyt.synsets(lemma, pos)
    output = []
    for syn in synsets:
        for syn_lemma in syn.lemmas():
            word = syn_lemma.name().replace('_', ' ')
            if word != lemma:
                output.append(word)
    return output

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos
    freq = defaultdict(lambda: 0)

    synsets = wnpyt.synsets(lemma, pos)
    for syn in synsets:
        for syn_lemma in syn.lemmas():
            word = syn_lemma.name().replace('_', ' ')
            if word != lemma:
                freq[word] += syn_lemma.count()

    max_k = None
    max_val = -np.inf
    for k, val in freq.items():
        if val > max_val:
            max_val = val
            max_k = k
    
    return max_k # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = set(stopwords.words('english'))
    synsets = wnpyt.synsets(context.lemma, context.pos)

    s = " ".join(context.left_context + context.right_context)

    tokens = set(tokenize(s)) - stop_words


    overlaps = [0 for i in range(len(synsets))]
    for i, syn in enumerate(synsets):
        all_def = set()
        all_def.update(tokenize(syn.definition()))
        
        for hyper in syn.hypernyms():
            all_def.update(tokenize(hyper.definition()))
            for exam in hyper.examples():
                all_def.update(tokenize(exam))
        for exam in syn.examples():
            all_def.update(tokenize(exam))
        
        num_overlap = len((all_def - stop_words) & (tokens))
        overlaps[i] += 1000 * num_overlap

        b = 0
        for l in syn.lemmas():
            word = l.name().replace('_', ' ')
            if word == context.lemma:
                b += l.count()
        overlaps[i] += 100 * b

    sort = np.argsort(overlaps)

    most_freq_lemma = ""
    for i in range(len(sort)-1, -1, -1):
        syn = synsets[sort[i]]
        most_freq = -np.inf
        for l in syn.lemmas():
            word = l.name().replace('_', ' ')
            if word != context.lemma:
                if l.count() > most_freq:
                    most_freq_lemma = word
                    most_freq = l.count()
        if most_freq_lemma != "":
            return most_freq_lemma
    
    return most_freq_lemma #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        max_similarity = 0
        res = None
        for candidate in candidates:
            if candidate in self.model and context.lemma in self.model:
                similarity = self.model.similarity(candidate, context.lemma)
                if similarity > max_similarity:
                    max_similarity = similarity
                    res = candidate
                    
        return res # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:

        left_s = " ".join(context.left_context)
        right_s = " ".join(context.right_context)
        sentence = left_s + ' [MASK] ' + right_s
        #print(sentence)

        
        input_toks = self.tokenizer.encode(sentence)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]

        idx = input_toks.index(self.tokenizer.mask_token_id)

        res = None
        highest_score = 0

        candidates = get_candidates(context.lemma, context.pos)     
        for cand in candidates:
            cand_id = self.tokenizer.convert_tokens_to_ids(cand)
            if predictions[0, idx, cand_id] > highest_score:
                highest_score = predictions[0, idx, cand_id]
                res = cand
              
        return res # replace for part 5


# part 6
class BertLesk(BertPredictor):

    def __init__(self): 
        super().__init__()
        self.stop_words = set(stopwords.words('english')) 
    
    def get_synset(self, context : Context): 
        stop_words = set(stopwords.words('english'))
        synsets = wnpyt.synsets(context.lemma, context.pos)

        s = " ".join(context.left_context + context.right_context)

        tokens = set(tokenize(s)) - stop_words


        overlaps = [0 for i in range(len(synsets))]
        for i, syn in enumerate(synsets):
            all_def = set()
            all_def.update(tokenize(syn.definition()))
        
            for hyper in syn.hypernyms():
                all_def.update(tokenize(hyper.definition()))
                for exam in hyper.examples():
                    all_def.update(tokenize(exam))
            for exam in syn.examples():
                all_def.update(tokenize(exam))
        
            num_overlap = len((all_def - stop_words) & (tokens))
            overlaps[i] += 1000 * num_overlap

            b = 0
            for l in syn.lemmas():
                word = l.name().replace('_', ' ')
                if word == context.lemma:
                    b += l.count()
            overlaps[i] += 100 * b

        sort = np.argsort(overlaps)
        for i in range(len(sort)-1, -1, -1):
            syn = synsets[sort[i]]
            for l in syn.lemmas():
                word = l.name().replace('_', ' ')
                if word != context.lemma:
                    return syn
        
        return synsets[np.argmax(overlaps)]

    def predict(self, context : Context) -> str:
        synset = self.get_synset(context)
        candidates = []
        for l in synset.lemmas():
            word = l.name().replace('_', ' ')
            if word != context.lemma:
                candidates.append(word)

        left_s = " ".join(context.left_context)
        right_s = " ".join(context.right_context)
        sentence = left_s + ' [MASK] ' + right_s

        input_toks = self.tokenizer.encode(sentence)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]

        idx = input_toks.index(self.tokenizer.mask_token_id)
        
        res = None
        highest_score = 0

        for cand in candidates:
            cand_id = self.tokenizer.convert_tokens_to_ids(cand)
            if predictions[0, idx, cand_id] > highest_score:
                highest_score = predictions[0, idx, cand_id]
                res = cand
              
        return res 
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec_predictor = Word2VecSubst(W2VMODEL_FILENAME)

    bert_predictor = BertPredictor()
    bertlesk_predictor = BertLesk()
    
    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context) 
        
        # pt 2 precision = 0.098, recall = 0.098 precision = 0.136, recall = 0.136
        # prediction = wn_frequency_predictor(context)

        # pt 3 precision = 0.113, recall = 0.113 precision = 0.160, recall = 0.160
        # prediction = wn_simple_lesk_predictor(context)
        
        # pt 4 precision = 0.115, recall = 0.115 precision = 0.170, recall = 0.170
        # prediction = word2vec_predictor.predict_nearest(context)

        # pt 5 precision = 0.114, recall = 0.114 precision = 0.165, recall = 0.165
        # prediction = bert_predictor.predict(context)

        # pt 6 combines Bert with Simple Lesk Algorithm because the performance of Simple Lesk seems good
        # The model uses simple lesk to generate a selected synset and word candidates, and uses Bert score the candidates
        # precision = 0.103, recall = 0.103 precision = 0.146, recall = 0.146
        #prediction = bertlesk_predictor.predict(context)
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
