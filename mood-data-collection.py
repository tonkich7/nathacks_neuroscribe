from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import random
import time
import glob
import cv2
import sys
import tkinter as tk
from PIL import ImageTk, Image

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1 
SERIAL_PORT = 'COM8'
NUM_SAMPLES_EACH = 25 #how many samples of each label should be gathered in the baseline step
SAMPLE_SECONDS = 4 #how many seconds of data should be gathered for each sample
FILE_PATH = './experiment_data/mood_'
WORD_ORDER_PATH = './experiment_data/mood_images_'

POSITIVE_VAL = 1000
NEUTRAL_VAL = 2000
NEGATIVE_VAL = 3000

stimulus_map = {
    'positive': 1000,
    'neutral': 2000,
    'negative': 3000
}

stimuli_messages = {
    1000: 'Think positive :)',
    2000: 'Think neutral :|',
    3000: 'Think negative :('
}

def update_image(image_window, panel, image_path):
    img = ImageTk.PhotoImage(Image.open(image_path))
    panel.configure(image=img)
    panel.image = img
    image_window.update_idletasks()
    image_window.update()

def run_experiment(board, image_dict):
    #put in NUM_SAMPLES_EACH stimuli for each label before shuffling
    stimuli = [i for i in (POSITIVE_VAL, NEUTRAL_VAL, NEGATIVE_VAL) for j in range(NUM_SAMPLES_EACH)]
    #shuffles the labels to randomize order of stimuli shown
    random.shuffle(stimuli)

    print("Starting experiment\n")

    board.prepare_session()
    board.start_stream(450000)
    time.sleep(SAMPLE_SECONDS)

    image_order = []

    image_window = tk.Tk()
    image_window.geometry("1920x1080")
    fixation_img = ImageTk.PhotoImage(Image.open("./images/fixation.png"))
    panel = tk.Label(image_window, image=fixation_img)
    panel.pack(side="bottom", fill="both", expand="yes")
    image_window.update_idletasks()
    image_window.update()

    for stimulus in stimuli:
        time.sleep(SAMPLE_SECONDS)
        board.insert_marker(stimulus) #insert label into stream
        print(stimulus) #show stimulus
        random_image = random.choice(image_dict[stimulus])
        image_order.append(random_image)
        update_image(image_window, panel, random_image)
        time.sleep(SAMPLE_SECONDS) #wait some time for data collection
        update_image(image_window, panel, "./images/fixation.png")

    board.stop_stream()
    data = board.get_board_data()
    board.release_session()

    return data, np.array(image_order)

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

def build_image_dict():
    image_dict = {}
    for folder_name in ('positive', 'neutral', 'negative'):
        #file_path = "./images/" + folder_name + "/*.jpg"
        images = glob.glob("./images/" + folder_name + "/*.jpg")
        image_dict[stimulus_map[folder_name]] = images
    return image_dict


def main():
    print("Setting up board...")
    board = get_board()
    print("Building image dictionary...")
    image_dict = build_image_dict()
    print("Finished building image dictionary.")
    data, image_order = run_experiment(board, image_dict)
    file_name = FILE_PATH + str(int(time.time()))
    np.save(file_name, data)
    print("Saved", file_name + ".npy")
    word_file_name = WORD_ORDER_PATH + str(int(time.time()))
    np.save(word_file_name, image_order)
    print("Saved", word_file_name + ".npy")

main()