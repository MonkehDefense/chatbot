import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
import json
import pickle

import matplotlib.pyplot as plt

words = set()
classes = set()
documents = []
ignore = ['?','!','.',',','-',"'"]

lemm = WordNetLemmatizer()	# struktura wyciągająca rdzeń wyrazu
intents = json.loads(open('intents.json').read())	# dane treningowe

for intent in intents['intents']:
	for pattern in intent['patterns']:	# dla każdej kategorii/taga i każdego przykładowego pytania dla tego taga:
		word_list = nltk.word_tokenize(pattern)	# utwórz listę słów w pytaniu
		words.update(word_list)	# dodaj nowe słowa do words
		documents.append((word_list, intent['tag']))	# dodaj tupla (word_list, kategoria) do dokumentów
		classes.update([intent['tag']])	# Dodaj tag do classes, jeśli jeszcze go tam nie ma

#print(documents)
words = [lemm.lemmatize(word.lower()) for word in words if word not in ignore]	# wyciągnij rdzenie słów z words
words = sorted(words)
#print(words)

classes = sorted(classes)
#print(classes)

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
	bag = []	# bag of words
	# word_patterns = lista rdzeni wyrazów dla danego tupla (lista słów, tag)
	word_patterns = [lemm.lemmatize(word.lower()) for word in document[0]]
	for word in words:
		bag.append(1 if word in word_patterns else 0)

	#print(document)
	output_row = output_empty[:] #skopiuj output_empty
	output_row[classes.index(document[1])] = 1	# oznacz klasę zawartą w dokumencie, przez jej indeks w classes
	training.append([bag,output_row])	# zapisz parę bag of words i output row - wektory opisujące występujące słowa i właściwą kategorię

training = np.array(training)
np.random.shuffle(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print(len(train_x))
model = keras.models.Sequential([
	layers.Dense(256,input_shape=(len(train_x[0]),), activation='relu'),
	layers.Dropout(.3),
	layers.Dense(128, activation='relu'),
	layers.Dropout(.3),	#3
	layers.Dense(64, activation='relu'),
	layers.Dropout(.3),
	layers.Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = .0015), metrics=['accuracy'])

fit = model.fit(np.array(train_x), np.array(train_y), epochs=350, batch_size=5, verbose=1, validation_split=.2)

#fig, ax = plt.subplots(1, 2, figsize=(9, 3))
#ax[0].plot(fit.history['loss'])
#ax[0].plot(fit.history['val_loss'])
#ax[1].plot(fit.history['accuracy'])
#ax[1].plot(fit.history['val_accuracy'])
#plt.show()
#model.save('chatbot_model.h5', fit)
