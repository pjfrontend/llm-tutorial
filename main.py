import tensorflow as tf
import nltk
import numpy as np
import pandas as pd
nltk.download('brown')
from nltk.corpus import brown
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

def preprocess(data):
    stop_words = set(stopwords.words('english'))
    processed_data = []
    for sentence in data:
        sentence = [word.lower() for word in sentence if word.isalpha()]
        sentence = [word for word in sentence if word not in stop_words]
        processed_data.append(sentence)
    return processed_data




if __name__ == "__main__":
    data = brown.sents()
    # print(data[:5])
    processed_data = preprocess(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_data)
    sequences = tokenizer.texts_to_sequences(processed_data)
    input_sequences = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i+1]
            input_sequences.append(n_gram_sequence)
    max_seq_length = max([len(seq) for seq in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
    vocab_size = len(tokenizer.word_index) + 1  # To account for padding token

   

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # model = Sequential()
    # model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length - 1))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(128))
    # model.add(Dense(vocab_size, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # input_sequences = np.array(input_sequences)
    # X = input_sequences[:, :-1]
    # y = input_sequences[:, -1]
    # # Convert labels to one-hot encoding
    # y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
    # history = model.fit(X, y, epochs=50, batch_size=128, verbose=0)
    # plt.plot(history.history['accuracy'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.show()

    def generate_text(seed_text, next_words):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_length - 1, padding='pre')
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=1)[0]
            predicted_word = tokenizer.index_word[predicted_word_index]
            seed_text += " " + predicted_word
        return seed_text

    print(generate_text("The future of AI", 10))
