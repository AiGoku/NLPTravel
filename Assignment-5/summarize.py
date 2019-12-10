
import re
import jieba
import numpy as np
from sklearn.decomposition import TruncatedSVD
import codecs
import pandas as pd
from collections import Counter
from gensim.models import word2vec
from gensim import models

sqlresultfile = './sqlResult_1558435.csv'
wikipath = '/zh_wiki/'
filepaths = ['zh_wiki_00','zh_wiki_01','zh_wiki_02']
path = "/cos_person/wiki/cut_file/stop_words_cut/result_cut"
weight_file_path = "/cos_person/wiki/cut_file/stop_words_cut/weight_file"
model_path = "/cos_person/wiki/model/stop_wiki_result.model"

class Summarize:

    def __init__(self,model_path):
        self.model = word2vec.Word2Vec.load(model_path)

    def get_weighted_average(self,We, x, w):
        """
        Compute the weighted average vectors
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in sentence i
        :param w: w[i, :] are the weights for the words in sentence i
        :return: emb[i, :] are the weighted average vector for sentence i
        """
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, We.shape[1]))
        for i in range(n_samples):
            emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
        return emb

    def compute_pc(self,X,npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self,X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(self,X, npc)
        if npc==1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX

    def SIF_embedding(self,We, x, w, params):
        """
        Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in the i-th sentence
        :param w: w[i, :] are the weights for the words in the i-th sentence
        :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
        :return: emb, emb[i, :] is the embedding for sentence i
        """
        emb = self.get_weighted_average(We, x, w)
        if  params > 0:
            emb = self.remove_pc(emb, params)
        return emb

    def calc_word_weight(self,textfile,weight_file_path):
        word_weight = []
        with codecs.open(textfile,'r','utf-8') as f:
            lines = f.readlines()
        Tokens =[]
        for line in lines:
            line = line.strip()
            if len(line) ==0:
                continue
            Tokens.extend(line.split())
        word_counts = Counter(Tokens)
        with codecs.open(weight_file_path,'w','utf-8') as wf:
            for word,count in word_counts.items():
                word_weight.append((word,count))

    def getWordmap(self,textfile):
        words={}
        We = []
        with codecs.open(textfile,'r','utf-8') as f:
            lines = f.readlines()
            for (n,i) in enumerate(lines):
                i=i.split()
                j = 1
                v = []
                while j < len(i):
                    v.append(float(i[j]))
                    j += 1
                words[i[0]]=n
                We.append(v)
        return (words, np.array(We))

    def getWordWeight(self,weightfile, a=1e-3):
        if a <=0: # when the parameter makes no sense, use unweighted
            a = 1.0

        word2weight = {}
        with codecs.open(weightfile,'r','utf-8') as f:
            lines = f.readlines()
        N = 0
        for i in lines:
            i=i.strip()
            if(len(i) > 0):
                i=i.split()
                if(len(i) == 2):
                    word2weight[i[0]] = float(i[1])
                    N += float(i[1])
                else:
                    print(i)
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value/N)
        return word2weight

    def getWeight(self,words, word2weight):
        weight4ind = {}
        for word, ind in words.items():
            if word in word2weight:
                weight4ind[ind] = word2weight[word]
            else:
                weight4ind[ind] = 1.0
        return weight4ind

    def prepare_data(list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype('float32')
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype='float32')
        return x, x_mask

    def lookupIDX(self,words,w):
        if w in words:
            return words[w]
        else:
            return len(words) - 1

    def getSeq(self,p1,words):
        p1 = jieba.cut(p1)
        X1 = []
        for i in p1:
            X1.append(self.lookupIDX(words,i))
        return X1

    def sentences2idx(self,sentences, words):
        """
        Given a list of sentences, output array of word indices that can be fed into the algorithms.
        :param sentences: a list of sentences
        :param words: a dictionary, words['str'] is the indices of the word 'str'
        :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
        """
        seq1 = []
        for i in sentences:
            seq1.append(self.selfgetSeq(i, words))

        x1, m1 = self.prepare_data(seq1)
        return x1, m1

    def seq2weight(seq, mask, weight4ind):
        weight = np.zeros(seq.shape).astype('float32')
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if mask[i,j] > 0 and seq[i,j] >= 0:
                    weight[i,j] = weight4ind[seq[i,j]]
        weight = np.asarray(weight, dtype='float32')
        return weight

    def sif(self,sentences,params=1):
        word_weight = self.getWordWeight("")
        weight4ind = self.getWeight(word_vectors.vocab, word_weight)
        x,m,_ = self.sentences2idx(sentences,word_vectors.vocab)
        w = self.seq2weight(x,m,weight4ind)
        return self.SIF_embedding(word_vectors.vectors,x,w,params)

    def sigmoid(self,x):
        return 1./(1 + np.exp(-1 * x))

    def f(self,vec_s,vec_t,vec_c,ws=0.5,wt=0.5):
        s = self.sigmoid(vec_s)
        t = self.sigmoid(vec_t)
        c = self.sigmoid(vec_c)
        result = ws*self.MSE(s,t)+wt*self.MSE(vec_s,vec_c)

    def MSE(self,X,Y):
        return  sum([(y-x)**2 for x,y in zip(X,Y)])/len(X)

    def cut_sentences(self,content):
        content = re.sub('([。！？\?])([^”’])', r"\1\n\2", content)
        content = re.sub('(\.{6})([^”’])', r"\1\n\2", content)
        content = re.sub('(\…{2})([^”’])', r"\1\n\2", content)
        content = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', content)
        content = content.rstrip()
        return content.split("\n")


    def summarize(self,content, title, n=4):
        sentences = self.cut_sentences(content)
        vec_t, vec_c = self.sif(title, word_vectors), self.sif(content, word_vectors)
        # result = sorted(X_y_model(trainX,trainY),key=lambda x1:distance(x1[0],testx))[:count]
        C_list = []
        i = 0
        for sentence in sentences:
            vec_s = self.sif(sentence, word_vectors)
            C = self.f(vec_s, vec_t, vec_c)
            C_list.append((sentence, C, i))

        best_sentence = sorted(C_list, key=lambda x: x[1])[:n]
        best_sentence = sorted(best_sentence, key=lambda x: x[2])
        return best_sentence