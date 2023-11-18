from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import random
import time
import glob
# import cv2
import sys
import tkinter as tk
from PIL import ImageTk, Image

BOARD_ID = 1 
SYNTHETIC_BOARD_ID = -1
SERIAL_PORT = 'COM8'
MIN_SAMPLES_EACH = 30 #how many samples of each label should be gathered in the baseline step
MIN_SAMPLES_TOTAL = 100
SAMPLE_SECONDS = 3 #how many seconds of data should be gathered for each sample
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

stimulus_counts = [0, 0, 0] #positive, neutral, negative

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

def run_experiment(board, board_id, images):
    print("Starting experiment\n")

    board.prepare_session()
    board.start_stream(450000)
    time.sleep(SAMPLE_SECONDS)

    image_order = []
    mood_order = []
    num_images = 0

    image_window = tk.Tk()
    image_window.geometry("960x1080")
    fixation_img = ImageTk.PhotoImage(Image.open("./images/fixation.png"))
    panel = tk.Label(image_window, image=fixation_img)
    panel.pack(side="left", fill=None, expand=False)
    image_window.update_idletasks()
    image_window.update()

    while min(stimulus_counts) < MIN_SAMPLES_EACH or num_images < MIN_SAMPLES_TOTAL:
        time.sleep(SAMPLE_SECONDS)
        board.insert_marker(1000) #insert label into stream
        random_image = random.choice(images)
        image_order.append(random_image)
        update_image(image_window, panel, random_image)
        time.sleep(SAMPLE_SECONDS) #wait some time for data collection
        update_image(image_window, panel, "./images/fixation.png")
        print("How did that image make you feel?")
        response = None
        while response not in list(range(1,4)):
            try:
                response = int(input("Positive = 1, Neutral = 2, Negative = 3\nResponse: "))
            except:
                pass
        mood_order.append(response)
        stimulus_counts[response-1] += 1
        num_images += 1

    board.stop_stream()
    data = board.get_board_data()
    board.release_session()
    event_channel = BoardShim.get_marker_channel(board_id)
    stim_indices = np.argwhere(data[event_channel]).flatten()
    data[event_channel, stim_indices] = np.array(mood_order)

    return data, np.array(image_order)

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
    print("Building image dictionary...")
    images = glob.glob("./images/all/*.jpg")
    print("Finished building image dictionary.")
    data, image_order = run_experiment(board, board_id, images)
    file_name = FILE_PATH + str(int(time.time()))
    np.save(file_name, data)
    print("Saved", file_name + ".npy")
    word_file_name = WORD_ORDER_PATH + str(int(time.time()))
    np.save(word_file_name, image_order)
    print("Saved", word_file_name + ".npy")

main()