import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np
import random
import json
import pickle
import warnings
import spacy  
import tkinter as tk
from tkinter import ttk
from tkinter import Scrollbar, Text
import random

nlp = spacy.load("en_core_web_sm")

class StemmerModel(keras.Model):
    def __init__(self):
        super(StemmerModel, self).__init__()
        self.stemming_layer = Dense(units=1, activation='linear')

    def call(self, inputs):
        stemmed_text = self.stemming_layer(inputs)
        return stemmed_text

with open('intents33.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

def stem_word(word):
    doc = nlp(word)
    stemmed_word = ""
    for token in doc:
        stemmed_word += token.lemma_ + " "
    return stemmed_word.strip()


for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        w = pattern.split()
        
        stemmed_words = [stem_word(word.lower()) for word in w if word not in ignore_words]
        words.extend(stemmed_words)
        documents.append((stemmed_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes = sorted(list(set(classes)))

#model data
training = []
output = []
output_empty = [0] * len(classes)
#training set (Bag of words)

for doc in documents:
    
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag)
    output.append(output_row)


training = np.array(training)
output = np.array(output)


indices = np.arange(len(training))
np.random.shuffle(indices)
training = training[indices]
output = output[indices]


#tst lists

train_x = training
train_y = output



#  epochs and batch size
epochs = 80 
batch_size = 8 

# Create a Keras model
model = keras.Sequential()

model.add(Dense(8, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='keras_logs')
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True)
early_stopping_callback = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
          callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_callback])

model.save('model.keras')


pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))


data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
# ... (Previous code)
train_x = data['train_x']
train_y = data['train_y']

# Load model
loaded_model = keras.models.load_model('model.keras')

def clean_up_sentence(sentence):
   
    sentence_words = sentence.split()
    sentence_words = [stem_word(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Define the error threshold
ERROR_THRESHOLD = 0.25


def classify(sentence):
    bow_input = bow(sentence, words)  
    bow_input = np.array(bow_input).reshape(1, -1)
   
    predictions = loaded_model.predict(bow_input)

   
    results = predictions[0]
    
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
   
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))  # Tuple -> Intent and Probability
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    
    if results:
        for i in intents['intents']:
            
            if i['tag'] == results[0][0]:
                return random.choice(i['responses'])

"""while True:
    input_data = input("You- ")
    if input_data in ['exit', 'quit']:
        break
    answer = response(input_data)
    if answer:
        print(answer)
    else:
        print("I'm not sure what you mean.")"""
def add_user_message(message):
    chat_display.configure(state='normal')
    chat_display.insert(tk.END, "You: " + message + "\n", 'user_message')
    chat_display.configure(state='disabled')

def add_bot_message(message):
    chat_display.configure(state='normal')
    chat_display.insert(tk.END, "Bot: " + message + "\n", 'bot_message')
    chat_display.configure(state='disabled')

root = tk.Tk()
root.title("Chatbot UI")

# Create a chat display Text widget with scrollbar
chat_display = tk.Text(root, wrap=tk.WORD, height=15, width=50)
chat_display.pack(pady=10, padx=10, side=tk.LEFT)
chat_display.configure(state='disabled')

# Create a scrollbar for the chat display
scrollbar = ttk.Scrollbar(root, command=chat_display.yview)
scrollbar.pack(side=tk.RIGHT, fill='y')
chat_display.config(yscrollcommand=scrollbar.set)

# Function to handle user input and display responses
def chat():
    input_data = user_input.get()
    user_input.delete(0, tk.END)  # Clear the input field

    if input_data in ['exit', 'quit']:
        root.quit()
    else:
        add_user_message(input_data)
        answer = response(input_data)
        if answer:
            add_bot_message(answer)
        else:
            add_bot_message("I'm not sure what you mean.")

# Create a frame for buttons and text entry on the right side
input_frame = tk.Frame(root, bg='#EDEDED')
input_frame.pack(pady=10, padx=10, side=tk.RIGHT, fill='y')

# Create an input field for the user to type messages
user_input = ttk.Entry(input_frame, width=35)
user_input.grid(row=0, column=0, padx=5, pady=5)

# Create a Send button with a custom style
send_button = ttk.Button(input_frame, text="Send", command=chat)
send_button.grid(row=0, column=1, padx=5, pady=5)
style = ttk.Style()
style.configure('TButton', relief='groove', borderwidth=5, focuscolor='none')
# Create an Exit button with a custom style
exit_button = ttk.Button(input_frame, text="Exit", command=root.quit)
exit_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

# Define styles for user and bot messages
chat_display.tag_configure('user_message', foreground='blue')
chat_display.tag_configure('bot_message', foreground='green')

root.mainloop()
