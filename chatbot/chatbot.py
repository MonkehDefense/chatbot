import numpy as np
import json
import pickle
import pandas
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemm = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemm.lemmatize(word) for word in sentence_words]
	return sentence_words

def bag_of_words(sentence):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)
	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)

def predict_class(sentence):
	bag = bag_of_words(sentence)
	res = model.predict(np.array([bag]))[0]
	ERROR_THRESHOLD = .25

	result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
	result.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in result:
		return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

	return return_list

def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if i['tag'] == tag:
			result = np.random.choice(i['responses'])
			break
	return result

#print('dziala')

while True:
	message = input('')
	ints = predict_class(message)
	res = get_response(ints, intents)
	print(res)
