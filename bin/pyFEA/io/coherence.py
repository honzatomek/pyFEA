#!/opt/python385_std/bin/python3
"""author:  Jan Tomek <jan.tomek.ext@stihl.de>
date:    17.3.2023
version: v1.0.0

description:
Coherence Analysis of signals from frequency sweep shaker measurement, where
the driving node input is compared to the accelerations on the measured
nodes."""

import pdb


import os
import sys
import argparse
import warnings
import math


from math import degrees, radians, sin
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import utils
from utils import UNV
from utils import POST

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", module = "matplotlib")

DOFS = {"x": 0,
        "y": 1,
        "z": 2}
DIRS = {v: k for k, v in DOFS.items()}
DOF = lambda x: DOFS[str(x).lower()]
DIR = lambda x: DIRS[int(float(x))]

E = lambda msg, i = 0: " " * i + "[-] " + msg
I = lambda msg, i = 0: " " * i + "[+] " + msg
N = lambda msg, i = 6: " " * i + msg

NEXT_POW2 = lambda x: int(1 if x == 0 else 2 ** (x - 1).bit_length())


def get_sine(times, freqs, signal):
    """
    returns function values of sine cuve with varying frequency

    f(s) = A(t) sin(2π ∫f(t) dt + φ(t))

    where:
        A(t) is amplitude at time t
        f(t) is frequency at time t
        φ(t) is phase at time t

    Iftdt = ∫f(t) dt
    """
    signal_new = []

    for i, t in enumerate(times):
        if i == 0:
            Iftdt = freqs[0] * t
        else:
            Iftdt += (freqs[i] + freqs[i-1]) / 2 * (t - times[i-1])

        s = signal[i].real * sin(2. * np.pi * Iftdt + signal[i].imag)

        if np.isnan(s):
            print(f"[-] Found NaN value for time {t = }, amplitude = {signal[i].real}, " +
                  f"phase = {degrees(signal[i].imag)}")
        signal_new.append(s)
    return np.array(signal_new, dtype=float)



def plot_multiple(x: list, y: list, labels: list, titles: list,
                  xaxis: list, yaxis: list, plot_title: str,
                  show_immediately: bool = False, save: str = None, offset: int = 0):
    """
                i0     i1     i2
    j0  set1 [ plot1, plot2, plot2]
    j1  set2 [ plot1, plot2]
    j2  set3 [ plot1, plot2, plot2]
    """
    print(N(f"Plotting: {plot_title:s}", offset))

    nsets = len(y)
    nplots = max([len(y[i]) for i in range(len(y))])
    fig = plt.figure(figsize=(16, 10), constrained_layout=True, dpi=80)
    gs = gridspec(nrows=1, ncols=nplots, figure = fig)
    gs.update(left = 0.05, right = 0.95, top = 0.90, bottom = 0.05)
    # fig, axs = plt.subplots(nrows=1, ncols=nplots, tight_layout=True)

    axs = []
    for i in range(nplots):
        axs.append(fig.add_subplot(gs[i]))
        for j in range(nsets):
            if i >= len(y[j]):
                continue
            elif y[j][i] is None:
                continue
            elif y[j][i].dtype in (complex, np.complex, np.complex64, np.complex128, np.complex256):
                axs[i].plot(x[j][i], y[j][i].real, label=labels[j][i] + " real")
                axs[i].plot(x[j][i], y[j][i].imag, label=labels[j][i] + " imag")
            else:
                axs[i].plot(x[j][i], y[j][i], label=labels[j][i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(xaxis[i])
        axs[i].set_ylabel(yaxis[i])
        axs[i].legend()
    fig.suptitle(plot_title)
    # fig.tight_layout()

    if save is not None:
        print(N(f"Saving plot: {plot_title:s}", offset + 2))
        print(N(f"{save:s}", offset + 4))
        plt.savefig(save)

    if show_immediately:
        plt.show()



def explode_points(points: np.ndarray, xbounds: np.ndarray, ybounds: np.ndarray,
                   points_offset: float = 25) -> np.ndarray:

    # normalise points to range (0, 1)
    points_new = points.copy()
    points_new[:,0] = (points_new[:,0] - xbounds[0]) / (xbounds[1] - xbounds[0])
    points_new[:,1] = (points_new[:,1] - ybounds[0]) / (ybounds[1] - ybounds[0])

    # center of plot
    center = np.array([0.5 , 0.5], dtype=float)

    # get offset vectors to explode the points
    for i, point in enumerate(points_new):
        vec = point - center
        vec /= np.linalg.norm(vec)
        vec *= points_offset
        points_new[i] = vec.copy()

    points_new = np.array(points_new, dtype=int)

    return points_new


def plot_multiple_rows(x: list, y: list, labels: list, titles: list,
                       xaxis: list, yaxis: list, plot_title: str, annotation: list = None,
                       ylim: list = None, xlim: list = None,
                       show_immediately: bool = False, save: str = None, offset: int = 0):
    """
                i0     i1     i2
    j0  set1 [ plot1, plot2, plot2]
    j1  set2 [ plot1, plot2]
    j2  set3 [ plot1, plot2, plot2]
    """
    print(N(f"Plotting: {plot_title:s}", offset))

    nsets = len(y)
    nplots = max([len(yy) for yy in y])
    fig = plt.figure(figsize=(16, 10), constrained_layout=True, dpi=80)

    nrows = int(math.floor(nplots ** 0.5))
    ncols = int(math.ceil(nplots / nrows))

    gs = gridspec(nrows=nrows, ncols=ncols, figure = fig)
    gs.update(left = 0.05, right = 0.95, top = 0.90, bottom = 0.05)
    # fig, axs = plt.subplots(nrows=1, ncols=nplots, tight_layout=True)

    color = None

    axs = []
    for i in range(nrows):
        for j in range(ncols):
            idx = int(i * nrows + j)
            if idx >= nplots:
                continue

            axs.append(fig.add_subplot(gs[i, j]))

            for k in range(nsets):
                if y[k][idx] is None:
                    continue

                elif y[k][idx].dtype in (complex, np.complex, np.complex64, np.complex128, np.complex256):
                    axs[idx].plot(x[k][idx], y[k][idx].real, label=labels[k][idx] + " real")
                    color = axs[idx].get_lines()[-1].get_color()
                    axs[idx].plot(x[k][idx], y[k][idx].imag, label=labels[k][idx] + " imag")
                else:
                    axs[idx].plot(x[k][idx], y[k][idx], label=labels[k][idx])
                    color = axs[idx].get_lines()[-1].get_color()

                if annotation is not None and annotation[k][idx] is not None and annotation[k][idx].shape[0] != 0:

                    if xlim is not None and xlim[idx] is not None:
                        xbounds = xlim[idx]
                    else:
                        xbounds = (x[k][idx][0], x[k][idx][-1])

                    if ylim is not None and ylim[idx] is not None:
                        ybounds = ylim[idx]
                    else:
                        ybounds = (np.min(y[k][idx]), np.max(y[k][idx]))

                    offset_a = explode_points(annotation[k][idx], xbounds, ybounds, points_offset=25)

                    alignment_h = ["left" if oa[0] >= 0 else "right" for oa in offset_a]
                    alignment_v = ["bottom" if oa[1] >= 0 else "top" for oa in offset_a]

                    for a, (xya, xyo) in enumerate(zip(annotation[k][idx], offset_a)):
                        axs[idx].annotate(f"{xya[0]:.2f}", xya,
                                          textcoords="offset points",
                                          xytext=xyo, ha=alignment_h[a], va=alignment_v[a],
                                          color=color,
                                          arrowprops=dict(arrowstyle="->"))

            axs[idx].set_title(titles[idx])
            axs[idx].set_xlabel(xaxis[idx])
            axs[idx].set_ylabel(yaxis[idx])

            if xlim is not None and xlim[idx] is not None:
                axs[idx].set_xlim(left=xlim[idx][0], right=xlim[idx][1])

            if ylim is not None and ylim[idx] is not None:
                axs[idx].set_ylim(bottom=ylim[idx][0], top=ylim[idx][1])

            axs[idx].legend()
    fig.suptitle(plot_title)
    # fig.tight_layout()

    if save is not None:
        print(N(f"Saving plot: {plot_title:s}", offset + 2))
        print(N(f"{save:s}", offset + 4))
        plt.savefig(save)

    if show_immediately:
        plt.show()



def plot1(x, y, label, title):
    fig, ax = plt.subplots(1, 1)
    if y.dtype in (complex, np.complex, np.complex64, np.complex128, np.complex256):
        ax.plot(x, y.real, label=label + " real")
        ax.plot(x, y.imag, label=label + " imag")
    else:
        ax.plot(x, y.real, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()



def resample(time_resampled: np.ndarray, freq_resampled: np.ndarray, freq: np.ndarray,
             signal: np.ndarray, sampling_frequency: float,
             signal_label: str, offset: int = 0) -> np.ndarray:

    dt = 1. / sampling_frequency
    shifted = False

    print(N(f"Resampling {signal_label:s}", offset))
    print(N(f"Sampling Frequency: {sampling_frequency:13.3E} Hz", offset + 2))
    print(N(f"Time Step:          {dt:13.3E} s", offset + 2))

    if signal.dtype in (complex, np.complex, np.complex64, np.complex128, np.complex256):
        amplitude   = signal.real
        phase       = signal.imag
        amplitude_r = np.interp(freq_resampled, freq, amplitude)
        phase_r     = np.radians(np.interp(freq_resampled, freq, np.degrees(phase), period=360.))

    else:
        amplitude   = signal
        amplitude_r = np.interp(freq_resampled, freq, amplitude)
        phase_r     = np.zeros(amplitude_r.shape, dtype=float)

    signal_r  = np.array([complex(a, p) for a, p in zip(amplitude_r, phase_r)], dtype=complex)
    signal_rt = get_sine(time_resampled, freq_resampled, signal_r)

    return signal_rt



def butterworth_lowpass_filter(signal: np.ndarray, fs: float, cutoff: float, offset: int = 0) -> np.ndarray:
    """
    Butterworth Lowpass filter

    sigal:  (np.ndarray) signal to filter
    fs:     (float)sample rate [Hz]
    cutoff: (float) desired cutoff frequency of the filter, slightly higher than the desired one
    """

    print(N(f"Applying Butterworth Low-Pass Filter", offset))
    print(N(f"Frequency: {cutoff:.2f} Hz", offset + 2))

    n = signal.shape[0]
    T = n / fs
    nyq = 0.5 * fs  # Nyquist frequency
    order = 2       # quadratic sine wave representation

    # normalise the frequency
    normal_cutoff = cutoff / nyq

    # get the filter coefficients
    b, a     = scipy.signal.butter(N = order, Wn = normal_cutoff, btype = 'lowpass', analog = False)

    # filter the signal
    signal_f = scipy.signal.filtfilt(b, a, signal)

    return signal_f



def butterworth_highpass_filter(signal: np.ndarray, fs: float, cutoff: float, offset: int = 0) -> np.ndarray:
    """
    Butterworth Highpass filter

    sigal:  (np.ndarray) signal to filter
    fs:     (float)sample rate [Hz]
    cutoff: (float) desired cutoff frequency of the filter, slightly lower than the desired one
    """
    print(N(f"Applying Butterworth High-Pass Filter", offset))
    print(N(f"Frequency: {cutoff:.2f} Hz", offset + 2))

    n = signal.shape[0]
    T = n / fs
    nyq = 0.5 * fs  # Nyquist frequency
    order = 2       # quadratic sine wave representation

    # normalise the frequency
    normal_cutoff = cutoff / nyq

    # get the filter coefficients
    b, a     = scipy.signal.butter(N = order, Wn = normal_cutoff, btype = 'highpass', analog = False)

    # filter the signal
    signal_f = scipy.signal.filtfilt(b, a, signal)

    return signal_f



def butterworth_bandpass_filter(signal: np.ndarray, fs: float,
                                lowcut: float, highcut: float, offset: int = 0) -> np.ndarray:
    """
    Butterworth Highpass filter

    sigal:   (np.ndarray) signal to filter
    fs:      (float)sample rate [Hz]
    lowcut:  (float) desired cutoff frequency of the filter, slightly higher than the desired one
    highcut: (float) desired cutoff frequency of the filter, slightly lower than the desired one
    """

    print(N(f"Applying Butterworth Band-Pass Filter", offset))
    print(N(f"Low Frequency:  {lowcut:.2f} Hz", offset + 2))
    print(N(f"High Frequency: {highcut:.2f} Hz", offset + 2))

    n = signal.shape[0]
    T = n / fs
    nyq = 0.5 * fs  # Nyquist frequency
    order = 2       # quadratic sine wave representation
    # n = int(T * fs) # total number of samples

    # normalise the frequencies
    normal_lowcut = lowcut / nyq
    normal_highcut = highcut / nyq

    # get the filter coefficients
    b, a     = scipy.signal.butter(N = order, Wn = [normal_lowcut, normal_highcut],
                                   btype = 'bandpass', analog = False)

    # filter the signal
    signal_f = scipy.signal.filtfilt(b, a, signal)

    return signal_f



def read_sweep_table(sweep_table: str, offset: int = 0) -> np.ndarray:
    """
    sweep_table:      (str) sweep table *.txt file with setting of the shaker
                      data can contain columns headers, if headers are present,
                      the needed values are "Frequency", "Freq" or "Freq." and "Time"
                      or "Hz" and "s"
                      headers have to be on the first row and start with "!"
                      on other lines the "!" means comment
                      if there are no heders the assumed order of columns is:
                      Frequency Time

                      data expample:
                      ! Frequency  Time
                        3.0    0.0
                       15.0   96.0
                      305.0 1256.0
    """

    print(I(f"Reading {sweep_table:s}", offset))
    vals = []
    freq = 0
    time = 1

    with open(sweep_table, "r") as swp:

        for i, line in enumerate(swp):
            line = line.strip()

            while "  " in line:
                line = line.replace("  ", " ")

            if i == 0 and line.startswith("!"):
                header = [v.lower() for v in line.split()][1:]
                for j, h in enumerate(header):
                    if h in ("time", "s"):
                        time = j
                    elif h in ("frequency", "freq", "freq.", "hz"):
                        freq == j
                continue

            elif line == "":
                continue

            elif line.startswith("!"):
                continue

            elif "!" in line:
                line = line.split("!")[0].strip()
                if line == "":
                    continue

            line = [float(v) for v in line.split()]
            vals.append([line[freq], line[time]])

        vals = np.array(vals, dtype=float)

    return vals



def add_curve(curves: dict, node: int, dir: int, x: np.ndarray, y: np.ndarray, datalen: int = 3) -> dict:
    """
    adds a signal curve for dof d into a dictionary {frequency: {node: np.ndarray(dof0, dof1, ...)} }
    """
    for i, f in enumerate(x):
        # f = float(f"{f:.3f}")
        if f not in curves.keys():
            curves.setdefault(f, {})

        if node not in curves[f].keys():
            curves[f].setdefault(node, [0.] * datalen)

        curves[f][node][dir] = y[i]

    return curves



def frequency_time_nodes(data: dict, sweep_table: np.ndarray) -> (np.ndarray, np.ndarray, list, int):
    """
    extracts frequencies, node numbers, driving node number from the data dictionary

    also converts the frequencies to the time domain based on the sweep tabel
    """
    # get all frequencies
    freq = np.array(list(sorted(data.keys())), dtype = float)

    # discard all that are out of sweep table bounds
    freq = freq[(freq >= sweep_table[0,0]) & (freq <= sweep_table[-1,0])]

    # convert freq to time using linear interpolation
    time = np.interp(freq, sweep_table[:,0], sweep_table[:,1])

    # get all nodes
    nodes = []
    for f in freq:
        nodes += list(data[f].keys())
    nodes = list(sorted(list(set(nodes))))

    # get driving node - lowest ID
    steuernode = nodes.pop(0)

    return freq, time, nodes, steuernode



def process_signal(tr: np.ndarray, fr: np.ndarray, data: dict, allfreq: np.ndarray,
                   node: int, dir: int, fs: float, butterworth: str, lf: float, hf: float,
                   window: str, nperseg: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    In:
        tr:          vector of resampled time
        fr:          vector fo resampled frequencies
        data:        data dictionary {freq: {node: np.ndarray}} with all signals
        allfreq:     vector with all frequencise in data
        node:        ID of the node to process
        dir:         ID of the dof to process (0 = x, 1 = y, 2 = z)
        fs:          sampling frequency in Hz
        butterworth: apply Butterworth filter (low/high/band)
        lf:          low cutoff frequency for Butterworth filter
        hf:          hight cutoff frequency for Butterworth filter
        window:      name of window function for signal processing
        nperseg:     number of segments for window function application

    Out:
        freq:        frequencies for this node and dof
        signal:      originnal signal in complex domain
        signal_r:    signal resampled to time domain tr
        fxx:         PSD frequencies
        Pxx:         PSD values
    """

    print(I(f"Processing Node: {node:n},{DIRS[dir]:s}"))

    # get signal data
    freq = np.array(list(sorted([f for f in allfreq if node in data[f].keys()])), dtype=float).flatten()
    signal = np.array([data[f][node][dir] for f in freq], dtype=complex)

    print(N(f"Transforming {node:n} to time domain.", 4))

    # resample and convert to time domain
    signal_r = resample(tr, fr, freq, signal, fs,
                        f"Node {node:n},{DIRS[dir]:s}", offset=4)

    # apply butterworth filter, if requested
    if butterworth == "low":
        stsignal_r = butterworth_lowpass_filter(stsignal_r, fs = fs,
                                                cutoff = hf * fr[-1], offset = 4)
    elif butterworth == "high":
        stsignal_r = butterworth_highpass_filter(stsignal_r, fs=fs,
                                                 cutoff = lf * fr[0], offset = 4)
    elif butterworth == "band":
        stsignal_r = butterworth_bandpass_filter(stsignal_r, fs = fs,
                                                 lowcut = lf * fr[0],
                                                 highcut = hf * fr[-1], offset = 4)

    print(N(f"Creating {node:n},{DIRS[dir]:s} PSD", 4))
    print(N(f"Filter Window: {window:s}", 6))
    print(N(f"Segment Length: {nperseg:n}", 6))

    fxx, Pxx = scipy.signal.welch(signal_r, fs, window=window, nperseg=nperseg)
    Pxx = Pxx[(fxx >= fr[0]) & (fxx <= fr[-1])]
    fxx = fxx[(fxx >= fr[0]) & (fxx <= fr[-1])]

    return freq, signal, signal_r, fxx, Pxx



def create_coherence(node: int, dir: int, sx: np.ndarray, sy: np.ndarray,
                     f: np.ndarray, fs: float, window: str, nperseg: int) -> (np.ndarray, np.ndarray):
    """
    In:
        node:        ID of the node to process
        dir:         ID of the dof to process (0 = x, 1 = y, 2 = z)
        sx:          first resampled signal in time domain
        sy:          second resampled signal in time domain
        f:           vector fo resampled frequencies
        fs:          sampling frequency in Hz
        nperseg:     number of segments for window function application

    Out:
        fxx:         Coherence frequencies
        Pxx:         Coherence values
    """
    print(N(f"Creating {node:n},{DIRS[dir]:s} Coherence", 4))

    fxy, Cxy = scipy.signal.coherence(sx, sy, fs=fs, window=window,
                                      nperseg=nperseg)
    Cxy = Cxy[(fxy >= f[0]) & (fxy <= f[-1])]
    fxy = fxy[(fxy >= f[0]) & (fxy <= f[-1])]
    cmax = np.max(Cxy)
    cmin = np.min(Cxy)
    print(N(f"Maximum up to {fxy[-1]:.2f} Hz: {cmax:.2f}", 6))
    print(N(f"Minimum up to {fxy[-1]:.2f} Hz: {cmin:.2f}", 6))

    # find peak values on signal flipped around the x axis
    peaks, _ = scipy.signal.find_peaks(-1. * Cxy, prominence = 0.2)
    print(N(f"Minima at: {', '.join([f'{f:.2f} Hz' for f in fxy[peaks]]):s}", 6))
    peaks = np.array([fxy[peaks], Cxy[peaks]], dtype = float).T

    return fxy, Cxy, peaks



def write_post(filename: str, data: dict, abscissae: str = "FREQUENCY",
               dattype: str = "AMPLITUDE", ncol: str = 2, curve: list = ["X", "Y", "Z"],
               result_name: str = "Coherence", analysis: str = "AuReLa",
               result_type: str = "COHERENCE"):

    settings = {"COMPONENT": "KOMPO_1",
                "RESULTS NAME": result_name,
                "ANALYSIS": analysis,
                "TYPE": result_type,
                "ABSCISSAE": abscissae,
                "NCOL": ncol,
                "DATTYPE": dattype,
                "CURVE": curve}

    POST.write(filename, data, sameID=True, output='X_XYDATA', settings=settings)



def coherence(unv_file: str, sweep_file: str, sampling_rate: float = 4., nperseg: int = 4096,
              window_filter: str = "hann", butterworth: bool = False, print_input: bool = False,
              show_plots: bool = False, show_immediately: bool = False, save_plots: bool = True):
    """
    unv_file:           (str) path to *.unv file with results
                        driving node is the node with lowest ID
                        shaker direction is assumed from driving node results as the only one non-zero

    sweep_table:        (str) sweep table *.txt file with setting of the shaker
                        data can contain columns headers, if headers are present,
                        the needed values are "Frequency", "Freq" or "Freq." and "Time"
                        or "Hz" and "s"
                        headers have to be on the first row and start with "!"
                        on other lines the "!" means comment
                        if there are no heders the assumed order of columns is:
                        Frequency Time

                        data expample:
                        ! Frequency  Time
                          3.0    0.0
                         15.0   96.0
                        305.0 1256.0

    sampling_rate:      highest frequency multiplier for resampling of the results
                        sampling frequency = sampling rate x highest frequency
                        default = 4

    nperseg:            (int) Number of data points per segement for signal processing using the window function

                            nperseg = 2 ** (nperseg - 1).bit_length()  # number of time steps per segment
                                                                       # length should be a poweer of 2
                        default: 4096

    window_filter:      (str) window filter type for signal processing
                        possible filters:
                            boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen,
                            bohman, blackmanharris, nuttall, barthann, cosine, exponential, tukey,
                            taylor, lanczos
                        default: hann

    butterworth:        (bool) apply Butterworth Low-Pass filter to the resampled data at

                        Possible filters:
                            low:  apply Butterworth Low-Pass filter at 1.1 of maximum frequency
                                  of the original signal
                            high: apply Butterworth High-Pass filter at 0.9 of minimum frequecy
                                  of the original signal
                            band: apply Butterworth Band-Pass filter at 0.9 of minimum frequecy
                                  and 1.1 maximum frequency of the original signal

                        default: None

    print_input:        (bool) print curves from the *.unv file to screen
                        default: False

    show_plots:         (bool) if the plots should be shown
                        default: False

    show_immediately:   (bool) show plots as they come (True) or show all at once at the end (False)
                        default: False

                        if show_plots is False then show_immediately is also False
    save_plots:         (bool) save plots as *.png, the name of the *.png is taken from the
                        *.unv file, resulting filename is {unv_path}/{unv_file}_N{node}.png
                        the locaction of the plot is the same as the *.unv file provided
                        if show_plots is False, save_plots is automatically set to True
                        default: True
    """
    # read the UNV file
    _, _, curves, _ = utils.read(unv_file)

    sweep_table = read_sweep_table(sweep_file)

    if not show_plots:
        show_immediately = False
        save_plots = True

    # prepare name for saving figures
    if save_plots:
        png_basename = os.path.join(os.path.dirname(os.path.realpath(unv_file)),
                                    os.path.splitext(os.path.basename(unv_file))[0])
        png_basename += "_" + window_filter

        if butterworth:
            png_basename += "_" + butterworth
    else:
        png_basename = None

    # none window == boxcar window
    if window_filter.lower() == "none":
        window_filter = "boxcar"

    # butterworth filter frequency multipliers
    bl = 0.9
    bh = 1.1

    # containters for *.post output
    signal_data = {}
    signal_resampled = {}
    psd_data = {}
    coherence_data = {}

    print(I("Sweep Table:"))
    print("    +------------+------------+")
    print("    | Freq. [Hz] |  Time [s]  |")
    print("    +------------+------------+")
    for f, t in sweep_table:
        print(f"    | {f:10.3E} | {t:10.3E} |")
    print("    +------------+------------+")

    # print read results
    if print_input:
        for frequency, curve in curves.items():
            print(f"[+] {frequency = }")
            for node, vector in curve.items():
                print(f"       {node = } {vector}")

    # get the frequencies, nodes, driving node and shaker direction
    # TODO:
    # discard data not in sweep table
    allfreq, alltime, nodes, steuernode = frequency_time_nodes(curves, sweep_table)

    # get driving node and shaker direction
    stsignal = np.array([curves[f][steuernode] for f in allfreq if steuernode in curves[f].keys()], dtype=complex)
    shakerdir = int(np.where(np.sum(stsignal.real, axis=0) != 0.)[0][0]) # the only nonzero component

    print(N(f"Preparing data for resampling to time domain", 4))
    # get time bounds
    mint = max(np.min(sweep_table, axis=0)[1], np.min(alltime))
    maxt = min(np.max(sweep_table, axis=0)[1], np.max(alltime))

    # get time and frequency values for resampling
    sampling_frequency = allfreq[-1] * sampling_rate
    time_r = np.linspace(mint, maxt, int((maxt - mint) * sampling_frequency) + 1)
    freq_r = np.interp(time_r, sweep_table[:,1].flatten(), sweep_table[:,0].flatten())

    # number of segments for PSD and Coherence
    # nperseg = NEXT_POW2(time_r.shape[0] // number_of_segments)
    nperseg = NEXT_POW2(nperseg)
    png_basename += f"_{nperseg:06n}"

    # process driving node
    stfreq, stsignal, stsignal_r, fxx, Pxx = process_signal(tr =          time_r,
                                                            fr =          freq_r,
                                                            data =        curves,
                                                            allfreq =     allfreq,
                                                            node =        steuernode,
                                                            dir =         shakerdir,
                                                            fs =          sampling_frequency,
                                                            butterworth = butterworth,
                                                            lf =          bl,
                                                            hf =          bh,
                                                            window =      window_filter,
                                                            nperseg =     nperseg)

    # store for export
    signal_data = add_curve(signal_data, steuernode, shakerdir * 2,     stfreq, stsignal.real, 6)
    signal_data = add_curve(signal_data, steuernode, shakerdir * 2 + 1, stfreq, stsignal.imag, 6)
    signal_resampled = add_curve(signal_resampled, steuernode, shakerdir, time_r, stsignal_r, 4)
    psd_data = add_curve(psd_data, steuernode, shakerdir, fxx, Pxx)

    # plot driving node
    titles = ["Original Signal", "Resampled Signal", "PSD of Resampled Signal"]
    xaxis = ["frequency [Hz]", "time [s]", "frequency [Hz]"]
    yaxis = ["acceleration [mm/s2]", "acceleration [mm/s2]", "PSD mm2/s4"]

    png = f"{png_basename:s}_N{steuernode:n}.png" if png_basename is not None else None
    x = [[stfreq, time_r, fxx]]
    y = [[stsignal.real, stsignal_r, Pxx]]
    labels = [[f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}"] * 3]
    plot_multiple_rows(x, y, labels, titles, xaxis, yaxis,
        plot_title = f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}",
        show_immediately = show_immediately, save = png, offset = 4)

    # prepare plot headers
    titles = ["Original Signal", "Resampled Signal", "PSD of Resampled Signal", "Coherence of Resampled Signal"]
    xaxis = ["frequency [Hz]", "time [s]", "frequency [Hz]", "frequency [Hz]"]
    yaxis = ["acceleration [mm/s2]", "acceleration [mm/s2]", "PSD [mm2/s4]", "Coherence [-]"]
    xlim = [None, None, None, None]
    ylim = [None, None, None, [-0.05, 1.05]]

    for i, node in enumerate(nodes):
        # prepare plot containers - first data the driving signal
        x = [[stfreq, None, fxx, None]]
        y = [[stsignal.real, None, Pxx, None]]
        anno = [[None, None, None, None]]
        labels = [[f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", None,
                   f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", None]]

        raumvector = np.zeros(time_r.shape, dtype=float)
        for dir in range(3):
            # process driving node

            # TODO:
            # ? Raumvector ?
            freq, signal, signal_r, fyy, Pyy = process_signal(tr =          time_r,
                                                              fr =          freq_r,
                                                              data =        curves,
                                                              allfreq =     allfreq,
                                                              node =        node,
                                                              dir =         dir,
                                                              fs =          sampling_frequency,
                                                              butterworth = butterworth,
                                                              lf =          bl,
                                                              hf =          bh,
                                                              window =      window_filter,
                                                              nperseg =     nperseg)

            raumvector += signal_r ** 2

            # store for export
            signal_data = add_curve(signal_data, node, dir * 2,     freq, signal.real, 6)
            signal_data = add_curve(signal_data, node, dir * 2 + 1, freq, signal.imag, 6)
            signal_resampled = add_curve(signal_resampled, node, dir, time_r, signal_r, 4)
            psd_data = add_curve(psd_data, node, dir, fyy, Pyy)

            # Coherence
            fxy, Cxy, peaks = create_coherence(node = node,
                                               dir = dir,
                                               sx = stsignal_r,
                                               sy = signal_r,
                                               f = freq_r,
                                               fs = sampling_frequency,
                                               window = window_filter,
                                               nperseg = nperseg)

            # save Coherence to a dict for *.post output
            coherence_data = add_curve(coherence_data, node, dir, fxy, Cxy, 4)

            x.append([freq, time_r, fyy, fxy])
            y.append([signal.real, signal_r, Pyy, Cxy])
            anno.append([None, None, None, peaks])
            labels.append([f"Node {node:n},{DIRS[dir]:s}"] * 4)



        png = f"{png_basename:s}_N{node:n}.png" if png_basename is not None else None
        plot_multiple_rows(x=x, y=y, labels=labels, titles=titles, xaxis=xaxis, yaxis=yaxis,
            plot_title=f"{os.path.basename(unv_file):s}: Driving Node {steuernode:n},{DIRS[shakerdir]:s} vs. Node {node:n}",
            annotation=anno, xlim=xlim, ylim=ylim, show_immediately=show_immediately, save=png, offset=4)



        # Signal Raumvector
        print(I(f"Processing Resampled Raumvector for node {node:n}.", 0))
        raumvector = raumvector ** (0.5)
        signal_resampled = add_curve(signal_resampled, node, 3, time_r, raumvector, 4)

        # Coherence Raumvector
        fxy, Cxy, peaks = create_coherence(node = node,
                                           dir = dir,
                                           sx = np.abs(stsignal_r),
                                           sy = raumvector,
                                           f = freq_r,
                                           fs = sampling_frequency,
                                           window = window_filter,
                                           nperseg = nperseg)
        # save Coherence to a dict for *.post output
        coherence_data = add_curve(coherence_data, node, 3, fxy, Cxy, 4)



    # write result to *.post file
    write_post(f"{png_basename:s}_signal.post", signal_data, abscissae = "Frequency [Hz]",
               dattype = ["AMPLITUDE", "PHASE", "AMPLITUDE", "PHASE", "AMPLITUDE", "PHASE"],
               result_type = ["Acceleration [mm/s2]", "PHASE [rad]",
                              "Acceleration [mm/s2]", "PHASE [rad]",
                              "Acceleration [mm/s2]", "PHASE [rad]"],
               curve = ["Xreal", "Ximag", "Yreal", "Yimag", "Zreal", "Zimag"])
    write_post(f"{png_basename:s}_resampled.post", signal_resampled, abscissae = "Time [s]",
               result_type = "Acceleration [mm/s2]",
               curve = ["X", "Y", "Z", "R"])
    write_post(f"{png_basename:s}_Pxx.post", psd_data, abscissae = "Frequency [Hz]",
               result_type = "PSD [mm2/s4]")
    write_post(f"{png_basename:s}_Cxy.post", coherence_data, abscissae = "Frequency [Hz]",
               result_type = "Coherence [-]",
               curve = ["X", "Y", "Z", "R"])



    # show plots if selected
    if (not show_immediately) and show_plots:
        plt.show()



if __name__ == "__main__":
    unv_file = "./res/GEH_complex.unv"
    unv_file = "./res/AXH_complex.unv"
    sweep_file = "./res/AXH_sweep_table.txt"

    # Create options parser object.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-r", "--sampling_ratio", dest="sampling_ratio", type=float, default=4.,
                        help="""Highest frequency multiplier for resampling of the results

sampling frequency = sampling rate x highest frequency

default = 4

""")

    parser.add_argument("-w", "--window_filter", dest="window_filter", type=str, default="hann",
                        help="""Window filter type for signal processing.

possible filters:
    boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen,
    bohman, blackmanharris, nuttall, barthann, cosine, exponential, tukey,
    taylor, lanczos

see help for scipy.signal.get_window() for more info. none == boxcar

default: hann

""")

    parser.add_argument("-n", "--number_of_points_per_segment", dest="nperseg", type=int, default=4096,
                        help="""Number of data points per segment for signal processing using the window function

    nperseg = 2 ** (nperseg - 1).bit_length()  # number of time steps per segment length should be a power of 2

default: 4096

""")

    parser.add_argument("-b", "--butterworth_filter", dest="butterworth", type=str, default=None,
                        help="""Apply Butterworth Filter to the signal after resampling

Possible filters:
    low:  apply Butterworth Low-Pass filter at 1.1 of maximum frequency
          of the original signal
    high: apply Butterworth High-Pass filter at 0.9 of minimum frequecy
          of the original signal
    band: apply Butterworth Band-Pass filter at 0.9 of minimum frequecy
          and 1.1 maximum frequency of the original signal

default: do not apply any
""")

    parser.add_argument("-i", "--print_input", dest="print_input", action="store_true", default=False,
                        help="Print curves from the *.unv file to screen, default: do not print")

    parser.add_argument("-p", "--show_plots", dest="show_plots", action="store_true", default=False,
                        help="If the plots should be shown, default: do not show")

    parser.add_argument("-m", "--show_immediately", dest="show_immediately", action="store_true", default=False,
                        help="""Show plots as they come. default: show all at once at the end.
If show_plots is not on then show_immediately is irrelevant.
""")

    parser.add_argument("-s", "--do_not_save_plots", dest="save_plots", action="store_false", default=True,
                        help="""Save plots as *.png, the name of the *.png is taken from the
*.unv file, resulting filename is {unv_path}/{unv_file}_N{node}.png.
The locaction of the plot is the same as the *.unv file provided.
If show_plots is not on, plots are automatically saved.
""")

    parser.add_argument("sweep_file", type=str,
                        help="""Sweep table *.txt file with setting of the shaker.
Data can contain columns headers, if headers are present,
the needed values are "Frequency", "Freq" or "Freq." and "Time"
or "Hz" and "s". Headers have to be on the first row and start with "!", on other
lines the "!" means comment.

If there are no heders the assumed order of columns is: Frequency Time

data expample:

>>> ! Frequency  Time
>>>   3.0    0.0
>>>  15.0   96.0
>>> 305.0 1256.0

""")

    parser.add_argument("unv_file", type=str,
                        help="""Path to *.unv file with results where driving node is the node
with lowest ID and shaker direction is assumed from driving node
result compnents as the one that is non-zero.

Example of input for one frequency 12,5 Hz (Dataset 55, Complex Results (5) of nodes
12000 - 120006, value order = X.real X.imag Y.real Y.imag Z.real Z.imag):

>>>     -1
>>>     55
>>> NONE
>>> NONE
>>> NONE
>>> NONE
>>> NONE
>>>          1         2         2         8         5         3
>>>          2         4         0         1
>>>   1.25000E+01  0.00000E+00  0.00000E+00  0.00000E+00
>>>      12000
>>>   0.00000E+00  0.00000E+00  1.24757E+04  0.00000E+00  0.00000E+00  0.00000E+00
>>>      12001
>>>   1.15250E+02 -7.27847E-02  1.29363E+04  1.73567E-02  8.47000E+02  3.13835E+00
>>>      12002
>>>   4.41889E+02 -1.81012E-02  1.29180E+04  1.68694E-02  1.55559E+02  3.09047E+00
>>>      12003
>>>   5.66397E+01 -1.56279E-01  1.27761E+04  1.68127E-02  1.47413E+02  3.03783E+00
>>>      12004
>>>   1.44444E+02 -2.70232E+00  1.28734E+04  1.68869E-02  1.00614E+03 -3.12329E+00
>>>      12005
>>>   4.65081E+02 -2.49486E+00  1.28965E+04  1.98913E-02  4.40716E+02  3.12769E+00
>>>      12006
>>>   1.97303E+02 -3.05406E+00  1.28391E+04  2.27727E-02  4.80140E+02 -3.08874E+00
>>>     -1

""")

    # Parse command-line arguments.
    args = parser.parse_args()

    unv_file = os.path.realpath(args.unv_file)
    sweep_file = os.path.realpath(args.sweep_file)

    print(I(f"Started script {__file__:s}", 0))
    print(N(f"UNV file:                 {unv_file:s}", 4))
    print(N(f"Sweep file:               {sweep_file:s}", 4))
    print(N(f"Sampling ratio:           {args.sampling_ratio:<.3f}", 4))
    print(N(f"Number of p. per segment: {args.nperseg:<n}", 4))
    print(N(f"Window filter :           {args.window_filter[0].upper() + args.window_filter[1:].lower():s}", 4))
    print(N(f"Butterworth Filter:       {args.butterworth}", 4))
    print(N(f"Print input:              {args.print_input}", 4))
    print(N(f"Show plots:               {args.show_plots}", 4))
    print(N(f"Show immediately:         {args.show_immediately}", 4))
    print(N(f"Save plots:               {args.save_plots}", 4))


    coherence(unv_file, sweep_file, args.sampling_ratio, args.nperseg,
              window_filter = args.window_filter.lower(), butterworth = args.butterworth,
              print_input = args.print_input, show_plots = args.show_plots,
              show_immediately = args.show_immediately, save_plots = args.save_plots)

