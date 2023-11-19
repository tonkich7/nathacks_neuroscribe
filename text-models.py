import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import pandas as pd
from tensorflow.keras.models import load_model


def get_sentences(data):
    #  get the sentence text from jsonl file
    all_sentences = []
    for index, row in data.iterrows():
        try:
            text = row['text'].strip()
            all_sentences.append(text)
        except Exception as e:
            print(e)
    return all_sentences

def get_input_seq(sentences, tokenizer):
    input_sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def train_model(data, model_name):
    #  get the sentence text from jsonl file
    all_sentences = get_sentences(data)
    # all_sentences = get_sentences(data)
    # for index, row in data.iterrows():
    #     try:
    #         text = row['text'].strip()
    #         all_sentences.append(text)
    #     except Exception as e:
    #         print(e)

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = get_input_seq(all_sentences, tokenizer)
    # for line in all_sentences:
    #     token_list = tokenizer.texts_to_sequences([line])[0]
    #     for i in range(1, len(token_list)):
    #         n_gram_sequence = token_list[:i+1]
    #         input_sequences.append(n_gram_sequence)

    # Pad sequences and split into predictors and label
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    # Convert target data to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Define the model
    model = Sequential()
    model.add(Embedding(total_words, 10,
                        input_length=max_sequence_len-1))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=10, verbose=1)
    model.save(model_name)





def run_model(model_name, data):
    # Generate next word predictions
    model = load_model(model_name, compile=False)
    model.compile()
    all_sentences = get_sentences(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = get_input_seq(all_sentences, tokenizer)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

    # get the next words
    seed_text = "I"
    next_words = 20


    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        print(token_list)
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word
    
    print("Next predicted words:", seed_text)

def main():
    pos_data = pd.read_json("./sentences/positive.jsonl", lines=True)
    neg_data = pd.read_json("./sentences/negative.jsonl", lines=True)
    neutral_data = pd.read_json("./sentences/neutral.jsonl", lines=True)

    # train_model(pos_data, "./text_models/pos_model.h5")
    # train_model(neg_data, "./text_models/neg_model.h5")
    # train_model(neutral_data, "./text_models/neutral_model.h5")

    run_model("./text_models/pos_model.h5", pos_data)
    run_model("./text_models/neg_model.h5", neg_data)
    run_model("./text_models/neutral_model.h5", neutral_data)

main()