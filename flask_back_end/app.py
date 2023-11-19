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

from flask import Flask, request

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1
SERIAL_PORT = 'COM8'
MOOD_MODEL_FILE = "/Users/alvinwu/Desktop/neuroscribe/nathacks_neuroscribe/mood_model"
# MOOD_MODEL_FILE = "../mood_model"

#json variables
MOOD = -1
DIRECTION = -1
WORD_1 = ""
WORD_2 = ""

#global variables
mood_model, mood_scaler, mood_vectorizer, mood_num_samples = None, None, None, None
board, board_id, sfreq = None, None, None
sentence, n_word_requests = "", 0

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
    print(type(board))
    print(type(model))
    print(type(sfreq))
    wait_time = int(math.ceil((mood_num_samples + math.ceil(sfreq*0.1))/sfreq))
    time.sleep(wait_time) #sleep so that there are enough samples to feed into the model
    data = board.get_current_board_data(math.ceil(mood_num_samples + sfreq*0.1)) #gets the last few samples plus 0.1 seconds of baseline
    #process data
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    eeg_data = data[eeg_channels[:4], :] / 1000000 #gets first four eeg channels in volts
    processed_eeg = eeg_data[:,eeg_data.shape[1]-mood_num_samples:] - np.mean(eeg_data[:,:eeg_data.shape[1]-mood_num_samples])
    sample = np.expand_dims(np.array(processed_eeg), 0)
    print(sample.shape)
    sample = vectorizer.transform(scaler.transform(sample))
    #get predictions
    test_preds = model(torch.tensor(sample).float())
    max_pred = torch.argmax(test_preds).item()
    MOOD = max_pred
    return MOOD

def get_direction(board):
    return -1

#mood is either 0,1,2 (positive, neutral, negative)
#nth_words refers to getting the nth and (n+1)th top words (e.g. if nth_words=0, 0th and 1st top words are returned)
def get_words(mood, nth_words):
    word_1 = ""
    word_2 = ""
    return word_1, word_2

# create the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/get-positive-words')
def get_pos_words():
    return {"word_1":"Man", "word_2":"The"}

@app.route('/get-neutral-words')
def get_neut_words():
    return {"word_1":"Man", "word_2":"The"}

@app.route('/get-negative-words')
def get_neg_words():
    return {"word_1":"Man", "word_2":"The"}

@app.route('/get-direction')
def get_direction_api():
    time.sleep(2)
    return {"direction":1}

@app.route('/get-mood')
def get_mood_api():
    mood = get_mood(board, mood_model, mood_scaler, mood_vectorizer, mood_num_samples, sfreq)
    return {"mood":mood}

@app.route('/setup')
def setup():
    global board
    global board_id
    global sfreq
    board, board_id, sfreq = start_board()
    model_resp = load_mood_model()
    return {"board_id":board_id, "sfreq": sfreq, "model_load_success":model_resp}

@app.route('/get-json')
def get_json():
    return str({"mood":MOOD, "direction":DIRECTION, "word_1":WORD_1, "word_2":WORD_2})

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)