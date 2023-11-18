from brainflow.board_shim import BoardShim
import mne
import numpy as np

BOARD_ID = 1 #ganglion board id
FILE_PATH = './experiment_data/mood_'
FILE_TIME = '1700266292'

STIM_MAP = {'positive': 1000, 'neutral': 2000, 'negative': 3000}

def get_raw_mne(data, eeg_channels, sfreq, ch_names, event_channel):
    ch_types = ['eeg'] * len(eeg_channels)
    data[eeg_channels, :] /= 1000000 #turns data from microvolts to volts
    eeg_and_event_data = data[eeg_channels + [event_channel], :] #isolate eeg and event channel
    info = mne.create_info(ch_names=ch_names + ['stim'], sfreq=sfreq, ch_types=ch_types + ['stim'])
    raw = mne.io.RawArray(eeg_and_event_data, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.set_eeg_reference(ref_channels='average') #might remove this, depends on if openbci already subtracts the reference values automatically
    return raw

def get_board_info(board_id):
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sfreq = BoardShim.get_sampling_rate(board_id)
    try:
        ch_names = BoardShim.get_eeg_names(board_id)
    except:
        ch_names = ['F8', 'F7', 'T5', 'T6']
    event_channel = BoardShim.get_marker_channel(board_id)
    return eeg_channels, sfreq, ch_names, event_channel

def plot_raw_data(raw):
    raw.plot_psd(average=True)

def preprocess_raw(raw):
    filtered_data = raw.copy().filter(l_freq=0.01, h_freq = 60) # bandpass 
    filtered_data = filtered_data.notch_filter(freqs=60) #removes 60hz spike
    return filtered_data

def build_epochs(raw, filtered_data, stimulus_map, ch_names):
    events = mne.find_events(raw, stim_channel='stim') # find events (mne can look through stim channel, find timings and event codes and return matrix of results)
    epochs = mne.Epochs(filtered_data, events, tmin=-0.1, tmax=2.0, picks=ch_names)
    epochs.event_id = stimulus_map
    return epochs

def main():
    #turns brainflow data into mne and gets epochs
    eeg_channels, sfreq, ch_names, event_channel = get_board_info(BOARD_ID)
    print("eeg_channels:", eeg_channels)
    print("event channel:",event_channel)
    data = np.load(FILE_PATH + FILE_TIME + ".npy")
    print("data.shape:",data.shape)
    #print(data[31])
    #event_channel=31
    raw = get_raw_mne(data, eeg_channels, sfreq, ch_names, event_channel)
    raw.plot_psd(average=True) #figure 1
    filtered_data = preprocess_raw(raw)
    filtered_data.compute_psd().plot() #figure 2
    epochs = build_epochs(raw, filtered_data, STIM_MAP, ch_names)

    #prints out the average signal for each stimulus
    pos_avg = epochs['positive'].average()
    neut_avg = epochs['neutral'].average()
    neg_avg = epochs['negative'].average()
    pos_avg.plot(spatial_colors=True) #figure 3
    neut_avg.plot(spatial_colors=True) #figure 4
    neg_avg.plot(spatial_colors=True) #figure 5
    pos_vs_neg = mne.combine_evoked([pos_avg, neg_avg], weights=[1,-1])
    pos_vs_neg.plot(spatial_colors=True) #figure 6
    pos_vs_neg.plot_joint() #figure 7

    epochs.save("./experiment_data/mood_epochs_" + FILE_TIME)

main()