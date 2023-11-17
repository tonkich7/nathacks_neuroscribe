from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import random
import time

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1 
SERIAL_PORT = 'COM8'
NUM_SAMPLES_EACH = 1 #how many samples of each label should be gathered in the baseline step
SAMPLE_SECONDS = 2 #how many seconds of data should be gathered for each sample
FILE_PATH = './experiment_data/mood_'

POSITIVE_VAL = 1000
NEUTRAL_VAL = 2000
NEGATIVE_VAL = 3000

stimuli_messages = {
    1000: 'Think positive :)',
    2000: 'Think neutral :|',
    3000: 'Think negative :('
}

def run_experiment(board):
    #put in NUM_SAMPLES_EACH stimuli for each label before shuffling
    stimuli = [i for i in (POSITIVE_VAL, NEUTRAL_VAL, NEGATIVE_VAL) for j in range(NUM_SAMPLES_EACH)]
    #shuffles the labels to randomize order of stimuli shown
    random.shuffle(stimuli)

    print("Starting experiment\n")

    board.prepare_session()
    board.start_stream(450000)
    time.sleep(SAMPLE_SECONDS)

    for stimulus in stimuli:
        print(stimuli_messages[stimulus], '\n') #show stimulus
        board.insert_marker(stimulus) #insert label into stream
        time.sleep(SAMPLE_SECONDS) #wait some time for data collection

    board.stop_stream()
    data = board.get_board_data()
    board.release_session()

    return data

def get_board(debug=False):
    if (debug):
        BoardShim.enable_dev_board_logger()
    else:
        BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        print("Board correctly prepared.")
    except Exception as e:
        print(e)
        board = BoardShim(SYNTHETIC_BOARD_ID, params)
        board.prepare_session()
        print("Board failed to prepare, prepared synthetic board instead.")
    board.release_session()
    return board

def main():
    board = get_board()
    data = run_experiment(board)
    file_name = FILE_PATH + str(int(time.time()))
    np.save(file_name, data)
    print("Saved", file_name)

main()