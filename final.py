import sys
import nltk
#nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer as wnl
from nltk.wsd import lesk
import os
sentence = sys.argv[1]
word = sys.argv[2]

def predict():
    sen=sentence
    y=word
    sent=sent_tokenize(sen)  
   
    tokens=word_tokenize(sen)
    syns=wordnet.synsets(y)
    defi=[]
    for syn in syns:
        defi.append(syn.definition())
       
    count=0
    maxi=0
    finaldf=""
    for de in defi:
        count=0
        for w in word_tokenize(sent[0]):
            if w in de:
                count+=1
        if(count>maxi):
            maxi=count
            finaldf=de
    return(y)


if __name__ == "__main__":
    x = predict()
    print("hello")
    print(x)
    #print(predict())
