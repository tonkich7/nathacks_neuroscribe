from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import json
import torch
import importlib  
models = importlib.import_module("mood-data-models")
import numpy as np
import math
import mne
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    Scaler,
    Vectorizer
)
import time
import pickle
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1
SERIAL_PORT = 'COM8'
#MOOD_MODEL_FILE = "/Users/alvinwu/Desktop/neuroscribe/nathacks_neuroscribe/mood_model"
MOOD_MODEL_FILE = "../mood_model"
DIR_MODEL_FILE = "../direction_model"
POS_MODEL_FILE = "../text_models/pos_model.h5"
POS_DATA_FILE = "../sentences/positive.jsonl"
NEUT_MODEL_FILE = "../text_models/neutral_model.h5"
#NEUT_MODEL_FILE ="/Users/alvinwu/Desktop/neuroscribe/nathacks_neuroscribe/text_models/neutral_model.h5"
NEUT_DATA_FILE = "../sentences/neutral.jsonl"
NEG_MODEL_FILE = "../text_models/neg_model.h5"
#NEG_MODEL_FILE ="/Users/alvinwu/Desktop/neuroscribe/nathacks_neuroscribe/text_models/neg_model.h5"
NEG_DATA_FILE = "../sentences/negative.jsonl"

#global variables
mood_model, mood_scaler, mood_vectorizer, mood_num_samples = None, None, None, None
dir_model, dir_scaler, dir_vectorizer, dir_num_samples = None, None, None, None
board, board_id, sfreq = None, None, None
pos_lang_model, neut_lang_model, neg_lang_model = None, None, None
pos_tokenizer, neut_tokenizer, neg_tokenizer = None, None, None
pos_max_length, neut_max_length, neg_max_length = None, None, None
sentence, n_requests = "", 0
curr_word_1, curr_word_2 = None, None

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

def start_board():
    print("Starting board...")
    #creates board object (synthetic if board not found)
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        board_id = BOARD_ID
        print("Board correctly prepared.")
    except Exception as e:
        print(e)
        # params = BrainFlowInputParams()
        board = BoardShim(SYNTHETIC_BOARD_ID, params)
        board.prepare_session()
        board_id = SYNTHETIC_BOARD_ID
        print("Board failed to prepare, prepared synthetic board instead.")
    #starts recording data
    board.start_stream(1000)
    return board, board_id, BoardShim.get_sampling_rate(board_id)

def load_lang_model(model_name, data):
    # Generate next word predictions
    model = load_model(model_name, compile=False)
    model.compile()
    all_sentences = get_sentences(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = get_input_seq(all_sentences, tokenizer)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    return model, tokenizer, max_sequence_len

def load_dir_model():
    global dir_model
    global dir_vectorizer
    global dir_scaler
    global dir_num_samples
    file = open(DIR_MODEL_FILE,'rb')
    try:
        dir_model, dir_scaler, dir_vectorizer, dir_num_samples = pickle.load(file)
        return True
    except:
        return False

def load_mood_model():
    global mood_model
    global mood_vectorizer
    global mood_scaler
    global mood_num_samples
    file = open(MOOD_MODEL_FILE,'rb')
    try:
        mood_model, mood_scaler, mood_vectorizer, mood_num_samples = pickle.load(file)
        return True
    except:
        return False

def get_mood(board, model, scaler, vectorizer, mood_num_samples, sfreq):
    #get data
    wait_time = int(math.ceil((mood_num_samples + math.ceil(sfreq*0.1))/sfreq))
    time.sleep(wait_time) #sleep so that there are enough samples to feed into the model
    data = board.get_current_board_data(math.ceil(mood_num_samples + sfreq*0.1)) #gets the last few samples plus 0.1 seconds of baseline
    #process data
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    eeg_data = data[eeg_channels[:4], :] / 1000000 #gets first four eeg channels in volts
    processed_eeg = eeg_data[:,eeg_data.shape[1]-mood_num_samples:] - np.mean(eeg_data[:,:eeg_data.shape[1]-mood_num_samples])
    sample = np.expand_dims(np.array(processed_eeg), 0)
    sample = vectorizer.transform(scaler.transform(sample))
    #get predictions
    test_preds = model(torch.tensor(sample).float())
    max_pred = torch.argmax(test_preds).item()
    mood = max_pred
    return mood

def get_direction(board, model, scaler, vectorizer, dir_num_samples, sfreq):
    #get data
    wait_time = int(math.ceil((dir_num_samples + math.ceil(sfreq*0.1))/sfreq))
    time.sleep(wait_time) #sleep so that there are enough samples to feed into the model
    data = board.get_current_board_data(math.ceil(dir_num_samples + sfreq*0.1)) #gets the last few samples plus 0.1 seconds of baseline
    #process data
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    eeg_data = data[eeg_channels[:4], :] / 1000000 #gets first four eeg channels in volts
    processed_eeg = eeg_data[:,eeg_data.shape[1]-mood_num_samples:] - np.mean(eeg_data[:,:eeg_data.shape[1]-dir_num_samples])
    sample = np.expand_dims(np.array(processed_eeg), 0)
    sample = vectorizer.transform(scaler.transform(sample))
    #get predictions
    test_preds = model(torch.tensor(sample).float())
    max_pred = torch.argmax(test_preds).item()
    direction = max_pred
    return direction

#mood is either 0,1,2 (positive, neutral, negative)
#nth_words refers to getting the nth and (n+1)th top words (e.g. if nth_words=0, 0th and 1st top words are returned)
def get_words(lang_model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = lang_model.predict(token_list)
    top_2_indices = np.argsort(predicted_probs[0])[::-1][n_requests*2:(n_requests*2)+2]
    top_2_words = [tokenizer.index_word[index] for index in top_2_indices]
    return top_2_words

# create the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/get-positive-words')
def get_pos_words():
    global n_requests
    global curr_word_1
    global curr_word_2
    top_2_words = get_words(pos_lang_model, pos_tokenizer, pos_max_length)
    curr_word_1 = top_2_words[0]
    curr_word_2 = top_2_words[1]
    n_requests += 1
    return {"word_1":top_2_words[0], "word_2":top_2_words[1]}

@app.route('/get-neutral-words')
def get_neut_words():
    global n_requests
    global curr_word_1
    global curr_word_2
    top_2_words = get_words(neut_lang_model, neut_tokenizer, neut_max_length)
    curr_word_1 = top_2_words[0]
    curr_word_2 = top_2_words[1]
    n_requests += 1
    return {"word_1":top_2_words[0], "word_2":top_2_words[1]}

@app.route('/get-negative-words')
def get_neg_words():
    global n_requests
    global curr_word_1
    global curr_word_2
    top_2_words = get_words(neg_lang_model, neg_tokenizer, neg_max_length)
    curr_word_1 = top_2_words[0]
    curr_word_2 = top_2_words[1]
    n_requests += 1
    return {"word_1":top_2_words[0], "word_2":top_2_words[1]}

@app.route('/get-direction')
def get_direction_api():
    global n_requests
    global sentence
    direction = get_direction(board, dir_model, dir_scaler, dir_vectorizer, dir_num_samples, sfreq)
    if direction == 0:
        n_requests = 0
        if sentence == "":
            sentence = curr_word_1
        else:
            sentence += " " + curr_word_1
    elif direction == 2:
        n_requests = 0
        if sentence == "":
            sentence = curr_word_2
        else:
            sentence += " " + curr_word_2

    return {"direction":direction}

@app.route('/get-mood')
def get_mood_api():
    mood = get_mood(board, mood_model, mood_scaler, mood_vectorizer, mood_num_samples, sfreq)
    return {"mood":mood}

@app.route('/setup')
def setup():
    global board
    global board_id
    global sfreq
    global pos_lang_model
    global pos_tokenizer
    global pos_max_length
    global neut_lang_model
    global neut_tokenizer
    global neut_max_length
    global neg_lang_model
    global neg_tokenizer
    global neg_max_length

    board, board_id, sfreq = start_board()
    dir_resp = load_dir_model()
    model_resp = load_mood_model()

    pos_data = pd.read_json(POS_DATA_FILE, lines=True)
    pos_lang_model, pos_tokenizer, pos_max_length = load_lang_model(POS_MODEL_FILE, pos_data)

    neut_data = pd.read_json(NEUT_DATA_FILE, lines=True)
    neut_lang_model, neut_tokenizer, neut_max_length = load_lang_model(NEUT_MODEL_FILE, neut_data)

    neg_data = pd.read_json(NEG_DATA_FILE, lines=True)
    neg_lang_model, neg_tokenizer, neg_max_length = load_lang_model(NEG_MODEL_FILE, neg_data)
    
    return {"board_id":board_id, "sfreq": sfreq, "model_load_success":model_resp, "direction_load_success":dir_resp}

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)