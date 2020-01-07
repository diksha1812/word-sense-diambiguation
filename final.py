from __future__ import print_function
from flask import Flask,request,jsonify
import pickle
import math
import operator
import json
import decimal
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings(action = 'ignore')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.wsd import lesk
import os


app = Flask(__name__)

#w= pickle.load(open('model.pkl','rb'))
#@app.route('/hahaa')
@app.route('/predict')


def predict():
    sen=(request.args['sent'])
    word=request.args['word']
    f=sen

    # Cleaing the text
    processed_article = f.lower()  
    processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )  
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # Preparing the dataset
    all_sentences = sent_tokenize(processed_article)

    all_words = [word_tokenize(sent) for sent in all_sentences]



    # Removing Stop Words
    from nltk.corpus import stopwords  
    for i in range(len(all_words)):  
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]


    #lookup=dict()
    corpus = all_words
    class word2vec():
        def __init__ (self):
            self.n = settings['n']
            self.eta = settings['learning_rate']
            self.epochs = settings['epochs']
            self.window = settings['window_size']
            pass
       
       
        # GENERATE TRAINING DATA
        def generate_training_data(self, settings, corpus):

            # GENERATE WORD COUNTS
            word_counts = defaultdict(int)
            for row in corpus:
                for word in row:
                    word_counts[word] += 1

            self.v_count = len(word_counts.keys())
            # GENERATE LOOKUP DICTIONARIES
            self.words_list = sorted(list(word_counts.keys()),reverse=False)
            self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
            self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

            training_data = []
            # CYCLE THROUGH EACH SENTENCE IN CORPUS
            for sentence in corpus:
                sent_len = len(sentence)

                # CYCLE THROUGH EACH WORD IN SENTENCE
                for i, word in enumerate(sentence):
                   
                    #w_target  = sentence[i]
                    w_target = self.word2onehot(sentence[i])

                    # CYCLE THROUGH CONTEXT WINDOW
                    w_context = []
                    for j in range(i-self.window, i+self.window+1):
                        if j!=i and j<=sent_len-1 and j>=0:
                            w_context.append(self.word2onehot(sentence[j]))
                    training_data.append([w_target, w_context])
            return np.array(training_data),word_counts.keys()


        # SOFTMAX ACTIVATION FUNCTION
        def softmax(self, x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)


        # CONVERT WORD TO ONE HOT ENCODING
        def word2onehot(self, word):
            word_vec = [ for i in range(0, self.v_count)]
            word_index = self.word_index[word]
            word_vec[word_index] = 1
            return word_vec


        # FORWARD PASS
        def forward_pass(self, x):
           
            h = np.dot(self.w1.T, x)
           
            u = np.dot(self.w2.T, h)
            y_c = self.softmax(u)
            return y_c, h, u
                   

        # BACKPROPAGATION
        def backprop(self, e, h, x):
            dl_dw2 = np.outer(h, e)  
            dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

            # UPDATE WEIGHTS
            self.w1 = self.w1 - (self.eta * dl_dw1)
            self.w2 = self.w2 - (self.eta * dl_dw2)
            pass


        # TRAIN W2V model
        def train(self, training_data):
            # INITIALIZE WEIGHT MATRICES
            self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))     # embedding matrix
            self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))     # context matrix
           
            # CYCLE THROUGH EACH EPOCH
            for i in range(0, self.epochs):

                self.loss = 0

                # CYCLE THROUGH EACH TRAINING SAMPLE
                for w_t, w_c in training_data:

                    # FORWARD PASS
                    y_pred, h, u = self.forward_pass(w_t)
                   
                    # CALCULATE ERROR
                    EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                    # BACKPROPAGATION
                    self.backprop(EI, h, w_t)

                    # CALCULATE LOSS
                    self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                    #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
                   
                print ('EPOCH:',i, 'LOSS:', self.loss)
            return self.w1
            pass


        # input a word, returns a vector (if available)
        def word_vec(self, word):
            w_index = self.word_index[word]
            v_w = self.w1[w_index]
            return v_w


       
        # input word, returns top [n] most similar words
        def word_sim(self, word):
           
            w1_index = self.word_index[word]
            v_w1 = self.w1[w1_index]
            # CYCLE THROUGH VOCAB
            word_sim = {}
            for i in range(self.v_count):
                v_w2 = self.w1[i]
                theta_num = np.dot(v_w1, v_w2)
                theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
                theta = theta_num / theta_den
                #print(theta_num,theta_den,theta)
                word = self.index_word[i]
                word_sim[word] = theta
     
            words_sorted = sorted(word_sim,key=word_sim.get,reverse = True )
            return words_sorted
             
            pass






    settings = {}
    settings['n'] = 5                   # dimension of word embeddings
    settings['window_size'] = 2         # context window +/- center word
    settings['min_count'] = 0           # minimum word count
    settings['epochs'] = 100            # number of training epochs
    settings['neg_samp'] = 5            # number of negative words to use during training
    settings['learning_rate'] = 0.01    # learning rate
    np.random.seed(0)                   # set the seed for reproducibility



    # INITIALIZE W2V MODEL
    w2v = word2vec()

    # generate training data
    training_data,keys = w2v.generate_training_data(settings, corpus)

    # train word2vec model
    X=w2v.train(training_data)

    sz=word

    syns=wordnet.synsets(sz)
    defi=[]
    for syn in syns:
        defi.append(syn.definition())

    llz=w2v.word_sim(sz)

    count=0
    maxi=0
    finaldf=""
    for de in defi:
        count=0
        for w in llz[:4]:
            if w in de:
                count+=1
        if(count>=maxi):
            maxi=count
            finaldf=de

    #print(finaldf)
    if(finaldf==''):
        finaldf='unidentified :('
    return(finaldf)

if __name__ == '__main__':
    app.run(debug=True)
