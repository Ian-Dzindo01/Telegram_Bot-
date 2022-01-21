import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.optimizers

nltk.download('wordnet')

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)      # don't show dtype deprecation warning

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append(((word_list), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tensorflow.keras.layers.Dropout(0.5))                                 # prevent overfitting
model.add(tensorflow.keras.layers.Dense(64, activation='relu'))                 # 64 neurons
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tensorflow.keras.optimizers.SGD(decay=1e-6, momentum=0.9, nesterov=True)                  # decay parameter reduces the learning rate over time
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

temp = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h5', temp)
print("Finished")
