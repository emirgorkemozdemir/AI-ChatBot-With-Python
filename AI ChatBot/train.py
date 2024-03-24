import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer 
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from sklearn.metrics import r2_score

lm = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

my_intents = json.loads(open('intent.json').read())

my_words = []
my_classes = []
my_documents = []
ignore = ['?','!','.',',']

for intent in my_intents['intents']:
    for pattern in intent['patterns']:
        listWords = nltk.word_tokenize(pattern)
        my_words.extend(listWords)
        my_documents.append((listWords,intent['tag']))
        if intent['tag'] not in my_classes:
            my_classes.append(intent['tag'])


my_words = [lm.lemmatize(word) for word in my_words if word not in ignore]
my_words = sorted(set(my_words))

my_classes = sorted(set(my_classes))

pickle.dump(my_words,open('words.pkl','wb'))
pickle.dump(my_classes,open('classes.pkl','wb'))


training_list = []
output_empty = [0]* len(my_classes)

for document in my_documents:
    temp_list = []
    word_patterns = document[0]
    word_patterns = [lm.lemmatize(word.lower()) for word in word_patterns]
    for word in my_words:
        if word in word_patterns:
            temp_list.append(1)
        else:
            temp_list.append(0)
    
    output = list(output_empty)
    output[my_classes.index(document[1])]=1
    training_list.append([temp_list,output])

random.shuffle(training_list)
training_list= np.array(training_list)

train_x = list(training_list[:,0])[:80]
train_y = list(training_list[:,1])[:80]

test_x = list(training_list[:,0])[80:]
test_y = list(training_list[:,1])[80:]


model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
modelfile = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('chatbotmodel.h5',modelfile)
