import numpy as np
import json


intents = json.loads(open('dev_light.json').read())
new_intents = []
for intent in intents:
	qsts = [intent['question']]
	answs = []
#	qsts.extend(intent['question'])
	for element in intent['annotations']:
		if element['type'] == 'singleAnswer':
			answs.append(element['answer'])
		else:
			for qa in element['qaPairs']:
				qsts.append(qa['question'])
				answs.append(qa['answer'])
	new_intents.append({
		'tag': intent['id'],
		'patterns': qsts,
		'responses': answs})

for intent in new_intents:
	intent['patterns'] = [pattern[0] if isinstance(pattern, list) else pattern for pattern in intent['patterns']]
	intent['responses'] = [response[0] if isinstance(response, list) else response for response in intent['responses']]

with open('new_data2.json', 'w') as f:
    json.dump({'intents': new_intents}, f)



#print(new_intents)
