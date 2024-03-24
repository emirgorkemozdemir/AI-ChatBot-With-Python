import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer 
from keras.models import load_model

lm = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbotmodel.h5')

def rearrange_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lm.lemmatize(word) for word in sentence_words]
    return sentence_words

def setWordList(sentence):
    sentece_words = rearrange_sentence(sentence)
    list = [0] * len(words)
    for w in sentece_words:
        for i, word in enumerate(words):
            if word == w:
                list[i]=1
    return np.array(list)


def predictor(sentence):
    wordList = setWordList(sentence)
    prediction= model.predict(np.array([wordList]))[0]
    threshold = 0.20
    results = [[i,r] for i , r in enumerate(prediction) if r > threshold]
    results.sort(key= lambda x:x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return result_list

def giveMeResponse(intentList,intentsJson):
    tag = intentList[0]['intents']
    listIntents = intentsJson['intents']
    for i in listIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    message = input("")
    ints = predictor(message)
    res = giveMeResponse(ints,intents)
    print(res)
