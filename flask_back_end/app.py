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

from flask import Flask, request

# create the Flask app
app = Flask(__name__)

@app.route('/get-json')
def get_json():
    return 'JSON Object Example'

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=3000)