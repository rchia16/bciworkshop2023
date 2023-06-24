import warnings
import os
from os.path import join, exists
import time
from pathlib import Path
import mne

from math import ceil
import numpy as np

from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.ndimage import uniform_filter1d

from bsl import StreamPlayer, StreamRecorder, datasets
from bsl import StreamReceiver
from bsl.lsl import resolve_streams
from bsl.utils import Timer
from bsl.triggers import MockTrigger, TriggerDef

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import patches

import pdb

STREAM_NAME = 'EEGLAB'
WEIGHTS_FNAME = None

def eeg_rest_stream():
    fif_file = datasets.eeg_resting_state.data_path()

    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)
    ''' below only necessary if repeat != 1 '''
    time.sleep(10)
    player.stop()
    print(player)

def eeg_rest_trigger():
    tdef = TriggerDef()
    tdef.add("rest", 1)

    player = StreamPlayer(stream_name, fif_file, trigger_def=tdef)
    player.start()
    print(player)


    time.sleep(5)
    player.stop()
    print(player)

def eeg_mock_trigger():
    trigger = MockTrigger()
    trigger.signal(1)

    player = StreamPlayer(stream_name, fif_file, trigger_def=tdef)
    player.start()
    print(player)


    time.sleep(5)
    player.stop()
    print(player)

def recorder():
    fif_file = datasets.eeg_resting_state.data_path()
    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)

    streams = [stream.name for stream in resolve_streams()]
    print(streams)
    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    recorder = StreamRecorder(record_dir, fname='example-rest-state')
    recorder.start()
    print(recorder)

    trigger = MockTrigger()
    trigger.signal(1)

    time.sleep(2)
    recorder.stop()
    print(recorder)

    player.stop()
    print(player)

    fname = join(record_dir, 'fif',
                 'example-resting-state-StreamPlayer-raw.fif')
    time.sleep(3)
    if exists(fname):
        raw = mne.io.read_raw_fif(fname, preload=True)
        print(raw)
        events = mne.find_events(raw, stim_channel='TRIGGER')
        print(events)

def stream_receiver():
    fif_file = datasets.eeg_resting_state.data_path()
    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)

    streams = [stream.name for stream in resolve_streams()]
    print(streams)
    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    recorder = StreamRecorder(record_dir, fname='example-rest-state')
    recorder.start()
    print(recorder)

    trigger = MockTrigger()
    trigger.signal(1)

    time.sleep(2)
    recorder.stop()
    print(recorder)

    player.stop()
    print(player)

    fname = join(record_dir, 'fif',
                 'example-resting-state-StreamPlayer-raw.fif')
    time.sleep(3)
    if exists(fname):
        raw = mne.io.read_raw_fif(fname, preload=True)
        print(raw)
        events = mne.find_events(raw, stim_channel='TRIGGER')
        print(events)

# TODO: not implemented
class Animator():
    def __init__(self, buffer_duration=5):
        fig = plt.figure(figsize=(12,7))
        ax0 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
        ax1 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
        ax1.sharex(ax0)

        ax0.set_title("Raw Signal")
        ax1.set_title("Processed Signal")
        ax1.set_xlabel("Time (s)")

        cls_ax = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        green_rect = patches.Rectangle((0, 0.75), buffer_duration, 0.25,
                                       facecolor='tab:green', alpha=0.5)
        red_rect = patches.Rectangle((0, 0), buffer_duration, 0.25,
                                     facecolor='tab:red', alpha=0.5)
        cls_ax.axhline(0, color='tab:red')
        cls_ax.axhline(1, color='tab:green')
        cls_ax.set_xticks([])
        cls_ax.set_yticks([0,1])

    def data_stream_input(self):
        pass

    def update(self, i):
        pass

def create_bandpass_filter(low, high, fs, n):
    """
    Create a bandpass filter using a butter filter of order n.

    Parameters
    ----------
    low : float
        The lower pass-band edge.
    high : float
        The upper pass-band edge.
    fs : float
        Sampling rate of the data.
    n : int
        Order of the filter.

    Returns
    -------
    sos : array
        Second-order sections representation of the IIR filter.
    zi_coeff : array
        Initial condition for sosfilt for step response steady-state.
    """
    # Divide by the Nyquist frequency
    bp_low = low / (0.5 * fs)
    bp_high = high / (0.5 * fs)
    # Compute SOS output (second order sections)
    sos = butter(n, [bp_low, bp_high], btype="band", output="sos")
    # Construct initial conditions for sosfilt for step response steady-state.
    zi_coeff = sosfilt_zi(sos).reshape((sos.shape[0], 2, 1))

    return sos, zi_coeff

class Buffer():
    """
    A buffer containing filter data and its associated timestamps.

    Parameters
    ----------
    buffer_duration : float
        Length of the buffer in seconds.
    sr : bsl.StreamReceiver
        StreamReceiver connected to the desired data stream.
    """

    def __init__(self, buffer_duration, sr, model, n_classes=1,
                 model_seq_len=200):
        # Store the StreamReceiver in a class attribute
        self.sr = sr

        # Retrieve sampling rate and number of channels
        self.fs = int(self.sr.streams[stream_name].sample_rate)
        self.nb_channels = len(self.sr.streams[stream_name].ch_list) - 1
        self.n_classes = n_classes
        self.model = model

        self.model_seq_len = model_seq_len
        buffer_duration_samples = ceil(buffer_duration * self.fs)

        if self.model_seq_len > buffer_duration_samples:
            self.model_seq_len = int(buffer_duration//1.5)
            warnings.warn("Sequence length too large, amending to"\
                          f"{self.model_seq_len}")

        # Define duration
        self.buffer_duration = buffer_duration
        self.buffer_duration_samples = buffer_duration_samples

        # Create data array
        self.timestamps = np.zeros(self.buffer_duration_samples)
        self.data = np.zeros((self.buffer_duration_samples,
                              self.nb_channels))
        # For demo purposes, let's store also the raw data
        self.raw_data = np.zeros((self.buffer_duration_samples,
                                  self.nb_channels))

        # classifier output
        self.class_data = np.zeros((self.buffer_duration_samples, n_classes))

        # Create filter BP (1, 15) Hz and filter variables
        self.sos, self.zi_coeff = create_bandpass_filter(5.0, 10.0, self.fs, n=2)
        self.zi = None

    def update(self):
        """
        Update the buffer with new samples from the StreamReceiver. This method
        should be called regularly, with a period at least smaller than the
        StreamReceiver buffer length.
        """
        # Acquire new data points
        self.sr.acquire()
        data_acquired, ts_list = self.sr.get_buffer()
        self.sr.reset_buffer()
        print("buff len list: ", len(ts_list))

        if len(ts_list) == 0:
            return  # break early, no new samples

        # Remove trigger channel
        data_acquired = data_acquired[:, 1:]

        # Filter acquired data
        if self.zi is None:
            # Initialize the initial conditions for the cascaded filter delays.
            self.zi = self.zi_coeff * np.mean(data_acquired, axis=0)
        data_filtered, self.zi = sosfilt(self.sos, data_acquired, axis=0, zi=self.zi)

        # Roll buffer, remove samples exiting and add new samples
        self.timestamps = np.roll(self.timestamps, -len(ts_list))
        self.timestamps[-len(ts_list) :] = ts_list
        self.data = np.roll(self.data, -len(ts_list), axis=0)
        self.data[-len(ts_list) :, :] = data_filtered
        self.raw_data = np.roll(self.raw_data, -len(ts_list), axis=0)
        self.raw_data[-len(ts_list) :, :] = data_acquired

        data_to_model = np.expand_dims(self.data[-self.model_seq_len:],
                                       axis=0)
        preds = self.model.predict(data_to_model, verbose=0)
        self.class_data = np.roll(self.class_data, -len(preds), axis=0)
        self.class_data[-len(preds) :, :] = preds

def tutorial_model(weights_fname:str=None):
    n_classes = 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='selu', name='InputLayer'),
        tf.keras.layers.Dense(1000, activation='relu', name='Hidden1'),
        tf.keras.layers.Dense(100, activation='relu', name='Hidden2'),
        tf.keras.layers.Dense(10, activation='relu', name='Hidden3'),
        # tf.keras.layers.GlobalAveragePooling1D(name='GAP1D'),
        tf.keras.layers.Reshape((-1,), name='Reshape'),
        tf.keras.layers.Dense(n_classes, activation='sigmoid',
                              name='OutputLayer'),
    ])
    if weights_fname is not None:
        assert exists(weights_fname), "Cannot find the weights file, is this"\
                " the right path?"
        print("Loading weights... ", end='')
        model.load_weights(weights_fname)
        print("Success!")

    return model

def get_receiver(bufsize=1, winsize=0.5, stream_name="StreamPlayer"):
    receiver = StreamReceiver(bufsize=bufsize, winsize=winsize,
                              stream_name=stream_name)
    time.sleep(winsize)  # wait to fill LSL inlet.
    return receiver

def real_time_stream(stream_name):
    demo_time = 30 # seconds
    buffer_duration = 5
    model_seq_len = 200 # seconds, sequence length for model
    learning_rate = 1e-3
    loss = 'mse'
    pred_buff_display = 15
    buff_refresh = 0.3

    bufsize = 1
    winsize = 0.5

    model = tutorial_model(WEIGHTS_FNAME)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # After the model is created, we then config the model with losses and metrics
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[loss])

    # define the receiver
    # receiver = get_receiver(bufsize=1, winsize=0.5, stream_name=stream_name)
    receiver = get_receiver(bufsize=bufsize, winsize=winsize,
                            stream_name=stream_name)

    record_dir = Path("~/bsl_data/examples").expanduser()
    os.makedirs(record_dir, exist_ok=True)
    print(record_dir)

    # FIXME: doesn't work when included with live lsl
    # set the recorder for recording save
    # recorder = StreamRecorder(record_dir, fname='example-rest-state')
    # recorder.start()
    # print(recorer)

    trigger = MockTrigger()
    trigger.signal(1)

    buffer = Buffer(buffer_duration, receiver, model,
                    model_seq_len=model_seq_len)

    timer = Timer()
    while timer.sec() <= buffer_duration:
        buffer.update()
    timer.reset()

    # fig, ax = plt.subplots(2,2, figsize=(12,7), sharex=True)
    fig = plt.figure(figsize=(12,7))
    ax0 = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=1)
    ax1 = plt.subplot2grid(shape=(2,2), loc=(1,0), colspan=1)
    ax1.sharex(ax0)

    ax0_title = "Raw Signal"
    ax1_title = "Processed Signal"
    ax1_xlabel = "Time (s)"
    cls_title = "Classifier"

    cls_ax = plt.subplot2grid(shape=(1,2), loc=(0,1), colspan=1)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    green_rect = patches.Rectangle((0, 0.75), buffer_duration, 0.25,
                                   facecolor='tab:green', alpha=0.5)
    red_rect = patches.Rectangle((0, 0), buffer_duration, 0.25,
                                 facecolor='tab:red', alpha=0.5)
    cls_ax.axhline(0, color='tab:red')
    cls_ax.axhline(1, color='tab:green')
    cls_ax.set_xticks([])
    cls_ax.set_yticks([0,1])

    idx_last_plot = 1
    dt, t0 = 0, 0
    # set interactive
    while timer.sec() <= demo_time:
        t0 = timer.sec()
        buffer.update()
        print("----\nupdate cmd\n----")
        if dt >= buff_refresh:
            ax0.plot(buffer.timestamps,
                  np.mean(buffer.raw_data[:, 1:], axis=1))
            ax1.plot(buffer.timestamps,
                       np.mean(buffer.data[:, 1:], axis=1))
            cls_ax.scatter(
                buffer.timestamps[-pred_buff_display:],
                buffer.class_data[-pred_buff_display:],
                c=buffer.class_data[-pred_buff_display:], cmap='RdYlGn',
                norm=norm,
            )

            red_rect.set_x(buffer.timestamps[-pred_buff_display])
            red_rect.set_width(
                buffer.timestamps[-1] - buffer.timestamps[-pred_buff_display])
            green_rect.set_x(buffer.timestamps[-pred_buff_display])
            green_rect.set_width(
                buffer.timestamps[-1] - buffer.timestamps[-pred_buff_display])
            cls_ax.add_patch(red_rect)
            cls_ax.add_patch(green_rect)

            cls_ax.set_xticks([])
            cls_ax.set_yticks([0,1])

            idx_last_plot+=1
            plt.pause(0.01)

            ax0.cla()
            ax1.cla()
            cls_ax.cla()
            ax0.set_title(ax0_title)
            ax1.set_title(ax1_title)
            ax1.set_xlabel(ax1_xlabel)
            cls_ax.set_title(cls_title)

            # refresh done, reset timer
            dt = 0

        # increment timer
        dt += timer.sec()-t0

    del receiver
    # recorder.stop()
    plt.show()
    model.summary()

if __name__ == '__main__':
    #######################
    # TESTING
    #######################
    # start the streamer for testing
    stream_name = "StreamPlayer"
    fif_file = datasets.eeg_resting_state.data_path()
    player = StreamPlayer(stream_name, fif_file)
    player.start()
    print(player)

    # stream_name = STREAM_NAME
    streams = [stream.name for stream in resolve_streams()]
    print("List of available streams\n: ", streams)
    assert stream_name in streams, f"Cannot find {stream_name}"
    # run the real-time classifier
    real_time_stream(stream_name)

    player.stop()
    print(player)
