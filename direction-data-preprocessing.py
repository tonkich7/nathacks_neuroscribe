from brainflow.board_shim import BoardShim
import mne
import numpy as np
import pywt
import pickle

BOARD_ID = 1 #ganglion board id
FILE_PATH = './experiment_data/direction_'
FILE_TIME = '1700377319'

STIM_MAP = {'up': 1, 'right': 2, 'down': 3, 'left': 4}

def get_fft_features(epochs):
    samples = epochs.get_data()
    freqs = np.fft.fftfreq(samples.shape[2])
    fft_features = []
    for sample in samples:
        sample_features = []
        for channel_data in sample:
            channel_fft = np.fft.fft(channel_data)
            dom_power = np.max(np.square(np.abs(channel_fft)))
            sample_features.append(dom_power)
            dom_freq = freqs[np.argmax(np.square(np.abs(channel_fft)))]
            sample_features.append(dom_freq)
        fft_features.append(sample_features)
    return fft_features

def get_wavelet_coefs(epochs):
    samples = epochs.get_data()
    wavelet_coefs = []
    for sample in samples:
        sample_features = []
        for channel_data in sample:
            coefs = pywt.wavedec(channel_data, 'sym20', level=5)
            flat_coefs = []
            for coef_array in coefs[:len(coefs)-2]:
                flat_coefs += list(coef_array)
            sample_features.append(flat_coefs)
        wavelet_coefs.append(sample_features)
    return np.array(wavelet_coefs)

def get_wavelet_features(epochs):
    samples = epochs.get_data()
    wavelet_features = []
    for sample in samples:
        sample_features = []
        for channel_data in sample:
            coefs = pywt.wavedec(channel_data, 'db8', level=8)
            alpha_power = np.mean(np.square(np.abs(coefs[2]))) #gets the squared mean of the D7 coefficient
            sample_features.append(alpha_power)
            delta_power = np.mean(np.square(np.abs(coefs[0]))) #gets the squared mean of the A coefficient
            sample_features.append(delta_power)
            sum_detail_energy = 0
            for dn in coefs[1:]:
                sum_detail_energy += np.mean(np.square(np.abs(dn)))
            total_wav_energy = sum_detail_energy/len(coefs[1:]) + delta_power
            sample_features.append(total_wav_energy)
        wavelet_features.append(sample_features)
    return wavelet_features

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
    filtered_data = raw.copy().filter(l_freq=0.1, h_freq = 60) # bandpass 
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
    up_avg = epochs['up'].average()
    right_avg = epochs['right'].average()
    down_avg = epochs['down'].average()
    left_avg = epochs['left'].average()
    up_avg.plot(spatial_colors=True) #figure 3
    right_avg.plot(spatial_colors=True) #figure 4
    down_avg.plot(spatial_colors=True) #figure 5
    left_avg.plot(spatial_colors=True) #figure 6
    up_vs_down = mne.combine_evoked([up_avg, down_avg], weights=[1,-1])
    up_vs_down.plot(spatial_colors=True) #figure 7
    up_vs_down.plot_joint() #figure 8
    left_vs_right = mne.combine_evoked([left_avg, right_avg], weights=[1,-1])
    left_vs_right.plot(spatial_colors=True) #figure 9
    left_vs_right.plot_joint() #figure 10

    epochs.save("./experiment_data/direction_epochs_" + FILE_TIME, overwrite=True)

    epochs.load_data()
    wavelet_coefs = get_wavelet_coefs(epochs)
    print(wavelet_coefs.shape)
    # fft_features = get_fft_features(resampled_epochs)
    # wavelet_features = get_wavelet_features(resampled_epochs)
    # frequency_features = np.concatenate((np.array(fft_features), np.array(wavelet_features)), axis=1)
    # print(frequency_features.shape)
    y = epochs.events[:,2]
    
    with open("./experiment_data/direction_features_labels_" + FILE_TIME, 'wb') as f:
        pickle.dump((wavelet_coefs, y), f)

main()