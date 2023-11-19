from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import json
import torch
import importlib  
models = importlib.import_module("mood-data-models")
import numpy as np
import math
import mne
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
MOOD_MODEL_FILE = "../mood_model"

#json variables
MOOD = -1
DIRECTION = -1
WORD_1 = ""
WORD_2 = ""

#global variables
mood_model, mood_scaler, mood_vectorizer, mood_num_samples = None, None, None, None
board, board_id, sfreq = None, None, None

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

def get_mood(board, model, scaler, vectorizer, sfreq):
    #get data
    print(type(board))
    print(type(model))
    print(type(sfreq))
    time.sleep((MOOD_NUM_SAMPLES + sfreq*0.1)/sfreq) #sleep so that there are enough samples to feed into the model
    data = board.get_current_data(MOOD_NUM_SAMPLES + sfreq*0.1) #gets the last few samples plus 0.1 seconds of baseline
    #process data
    eeg_data = data[eeg_channels, :] / 1000000 #gets eeg channels in volts
    processed_eeg = eeg_data[:,eeg_data.shape[1]-MOOD_NUM_SAMPLES:] - np.mean(eeg_data[:,:eeg_data.shape[1]-MOOD_NUM_SAMPLES])
    sample = np.expand_dims(np.array(processed_eeg, 0))
    sample = vectorizer.transform(scaler.transform(sample))
    #get predictions
    test_preds = model(torch.tensor(sample).float())
    MOOD = test_preds[0]
    return MOOD

def get_direction(board):
    return -1

def get_words():
    pass

# create the Flask app
app = Flask(__name__)

@app.route('/get-mood')
def get_mood_api():
    mood = get_mood(board, mood_model, mood_scaler, mood_vectorizer, sfreq)
    return {'mood':mood}

@app.route('/setup')
def setup():
    global board
    global board_id
    global sfreq
    board, board_id, sfreq = start_board()
    model_resp = load_mood_model()
    return {'board_id':board_id, 'sfreq': sfreq, 'model_load_success':model_resp}

@app.route('/get-json')
def get_json():
    return str({'mood':MOOD, 'direction':DIRECTION, 'word_1':WORD_1, 'word_2':WORD_2})

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=3000)