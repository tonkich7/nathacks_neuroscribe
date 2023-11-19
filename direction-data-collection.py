from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import random
import time
import glob
import cv2
import sys
import tkinter as tk
import turtle

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1
SERIAL_PORT = 'COM8'
NUM_SAMPLES_EACH = 25 #how many samples of each label should be gathered in the baseline step
SAMPLE_SECONDS = 3 #how many seconds of data should be gathered for each sample
FILE_PATH = './experiment_data/direction_'

NUM_DIRECTIONS = 4
directions_map = {
    1: 'up',
    2: 'right',
    3: 'down',
    4: 'left'
}

def run_experiment(board, board_id):
    print("Starting experiment\n")

    board.prepare_session()
    board.start_stream(450000)
    time.sleep(SAMPLE_SECONDS)

    stimuli = [i+1 for i in range(NUM_DIRECTIONS) for j in range(NUM_SAMPLES_EACH)]

    ttl = turtle.Turtle()
    screen=turtle.Screen()
    screen.setup(400,500)
    screen.bgcolor('white')
    ttl.speed(1.2)
    ttl.shape('square')
    ttl.pensize(6)
       
    #setting the size of the pen  
    ttl.pensize(6)
    for stimulus in stimuli:
        ttl.right(90)
        print(stimulus)
        ttl.setpos(0,0)
        ttl.pencolor('grey')   
        ttl.forward(90)
        time.sleep(SAMPLE_SECONDS)
        ttl.setpos(0,0)
        ttl.pencolor('red')
        #show animation of black rectangle moving in stimulus direction
        #return to center and change rectangle color to red
        board.insert_marker(stimulus) #insert label into stream
        #wait for user to move shape themself
        time.sleep(SAMPLE_SECONDS) #wait some time for data collection

    board.stop_stream()
    data = board.get_board_data()
    board.release_session()

    return data

def get_board(debug=False):
    board_id = None
    if (debug):
        BoardShim.enable_dev_board_logger()
    else:
        BoardShim.disable_board_logger()
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
    board.release_session()
    return board, board_id

def build_image_dict():
    image_dict = {}
    for folder_name in ('positive', 'neutral', 'negative'):
        #file_path = "./images/" + folder_name + "/*.jpg"
        images = glob.glob("./images/" + folder_name + "/*.jpg")
        image_dict[stimulus_map[folder_name]] = images
    return image_dict


def main():
    print("Setting up board...")
    board, board_id = get_board()
    data = run_experiment(board, board_id)
    file_name = FILE_PATH + str(int(time.time()))
    np.save(file_name, data)
    print("Saved", file_name + ".npy")

main()