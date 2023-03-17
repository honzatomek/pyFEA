#!/usr/bin/python3
"""author:  Jan Tomek
date:    17.3.2023
version: v1.0.0

description:
Coherence Analysis of signals from frequency sweep shaker measurement, where
the driving node input is compared to the accelerations on the measured
nodes."""


import os
import sys
import argparse
import warnings


from math import degrees, radians, sin
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import UNV

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", module = "matplotlib")

DOFS = {"x": 0,
        "y": 1,
        "z": 2}
DIRS = {v: k for k, v in DOFS.items()}

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
    print(N(f"Plotting {plot_title:s}", offset))

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
        print(N(f"Saving plot {plot_title:s} as {save:s}", offset + 2))
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



def resample(time: np.ndarray, freq: np.ndarray,
             time_resampled: np.ndarray, freq_resampled: np.ndarray,
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



def read_sweep_table(sweep_table: str, offset: int = 0) -> dict:
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
    vals = {}
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
            vals.setdefault(line[freq], line[time])

    return vals



def coherence(unv_file: str, sweep_file: str, sampling_rate: float = 4.,
              window_filter: str = "hann", butterworth: bool = False, print_input: bool = False,
              show_plots: bool = False, show_immediately: bool = False, save_plots: bool = True):
    """
    unv_file:         (str) path to *.unv file with results
                      driving node is the node with lowest ID
                      shaker direction is assumed from driving node results as the only one non-zero

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

    sampling_rate:    highest frequency multiplier for resampling of the results
                      sampling frequency = sampling rate x highest frequency
                      default = 4

    window_filter:    (str) window filter type for signal processing
                      possible filters:
                          boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen,
                          bohman, blackmanharris, nuttall, barthann, cosine, exponential, tukey,
                          taylor, lanczos
                      default: hann

    butterworth:      (bool) apply Butterworth Low-Pass filter to the resampled data at

                      Possible filters:
                          low:  apply Butterworth Low-Pass filter at 1.1 of maximum frequency
                                of the original signal
                          high: apply Butterworth High-Pass filter at 0.9 of minimum frequecy
                                of the original signal
                          band: apply Butterworth Band-Pass filter at 0.9 of minimum frequecy
                                and 1.1 maximum frequency of the original signal

                      default: None

    print_input:      (bool) print curves from the *.unv file to screen
                      default: False

    show_plots:       (bool) if the plots should be shown
                      default: False

    show_immediately: (bool) show plots as they come (True) or show all at once at the end (False)
                      default: False

                      if show_plots is False then show_immediately is also False
    save_plots:       (bool) save plots as *.png, the name of the *.png is taken from the
                      *.unv file, resulting filename is {unv_path}/{unv_file}_N{node}.png
                      the locaction of the plot is the same as the *.unv file provided
                      if show_plots is False, save_plots is automatically set to True
                      default: True
    """
    # read the UNV file
    _, _, curves, _ = UNV.read(unv_file)

    sweep_table = read_sweep_table(sweep_file)

    if not show_plots:
        show_immediately = False
        save_plots = True

    # prepare name for saving figures
    if save_plots:
        png_basename = os.path.join(os.path.dirname(os.path.realpath(unv_file)),
                                    os.path.splitext(os.path.basename(unv_file))[0])
    else:
        png_basename = None

    # butterworth filter frequency multipliers
    bl = 0.9
    bh = 1.1

    print(I("Sweep Table:"))
    print("    +------------+------------+")
    print("    | Freq. [Hz] |  Time [s]  |")
    print("    +------------+------------+")
    for f, t in sweep_table.items():
        print(f"    | {f:10.3E} | {t:10.3E} |")
    print("    +------------+------------+")

    # print read results
    if print_input:
        for frequency, curve in curves.items():
            print(f"[+] {frequency = }")
            for node, vector in curve.items():
                print(f"       {node = } {vector}")

    # get the frequencies, nodes, driving node and shaker direction
    freq = np.array(list(sorted(list(curves.keys()))), dtype=float)
    time = np.interp(freq, np.array(list(sweep_table.keys()), dtype=float),
                           np.array(list(sweep_table.values()), dtype=float))
    nodes = list(sorted([node for  node in curves[freq[0]].keys()]))
    steuernode = nodes.pop(0)    # driving node has to have lowes number
    shakerdir = [i for i in range(3) if curves[freq[0]][steuernode][i].real != 0.0][0]

    print(I(f"Processing Driving Node: {steuernode:n},{DIRS[shakerdir]:s}"))

    # transform into time domain
    print(N(f"Transforming {steuernode:n} to time domain.", 4))

    # driving signal in frequency domain
    stsignal = np.array([curves[f][steuernode][shakerdir] for f in freq], dtype=complex)

    # prepare for resampling to uniform spaced time
    print(N(f"Preparing data for resampling to time domain", 4))
    sampling_frequency = freq[-1] * sampling_rate
    dt = 1 / sampling_frequency
    time_r = np.linspace(time[0], time[-1], int((time[-1] - time[0]) * sampling_frequency) + 1)
    freq_r = np.interp(time_r,
                       np.array(list(sweep_table.values()), dtype=float),
                       np.array(list(sweep_table.keys()), dtype=float))

    # resample to uniform spaced time
    stsignal_r = resample(time, freq, time_r, freq_r, stsignal, sampling_frequency,
                          f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", offset=4)

    # apply butterworth filter if requested
    if butterworth == "low":
        stsignal_r = butterworth_lowpass_filter(stsignal_r, fs = sampling_frequency,
                                                cutoff = bh * freq[-1], offset = 4)
    elif butterworth == "high":
        stsignal_r = butterworth_highpass_filter(stsignal_r, fs=sampling_frequency,
                                                 cutoff = bl * freq[0], offset = 4)
    elif butterworth == "band":
        stsignal_r = butterworth_bandpass_filter(stsignal_r, fs = sampling_frequency,
                                                 lowcut = bl * freq[0], highcut = bh * freq[-1], offset = 4)

    # create PSD
    nperseg = NEXT_POW2(stsignal_r.shape[0] // 1000)

    print(N(f"Creating {steuernode:n},{DIRS[shakerdir]:s} PSD", 4))
    print(N(f"Filter Window: {window_filter:s}", 6))
    print(N(f"Segment Length: {nperseg:n}", 6))

    fxx, Pxx = scipy.signal.welch(stsignal_r, sampling_frequency, window=window_filter, nperseg=nperseg)
    Pxx = Pxx[fxx <= freq_r[-1]]
    fxx = fxx[fxx <= freq_r[-1]]

    # plot driving node
    titles = ["Original Signal", "Resampled Signal", "PSD of Resampled Signal"]
    xaxis = ["frequency [Hz]", "time [s]", "frequency [Hz]"]
    yaxis = ["acceleration [mm/s2]", "acceleration [mm/s2]", "PSD mm2/s4"]

    png = f"{png_basename:s}_N{steuernode:n}.png" if png_basename is not None else None
    x = [[freq, time_r, fxx]]
    y = [[stsignal.real, stsignal_r, Pxx]]
    labels = [[f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}"] * 3]
    plot_multiple(x, y, labels, titles, xaxis, yaxis,
                  plot_title = f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}",
                  show_immediately = show_immediately, save = png, offset = 4)


    # prepare plot headers
    titles = ["Original Signal", "Resampled Signal", "PSD of Resampled Signal", "Coherence of Resampled Signal"]
    xaxis = ["frequency [Hz]", "time [s]", "frequency [Hz]", "frequency [Hz]"]
    yaxis = ["acceleration [mm/s2]", "acceleration [mm/s2]", "PSD [mm2/s4]", "Coherence [-]"]

    for i, node in enumerate(nodes):
        # prepare plot containers
        x = [[freq, None, fxx, None]]
        y = [[stsignal.real, None, Pxx, None]]
        labels = [[f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", None,
                   f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", None]]

        for dir in range(3):
            print(I(f"Processing Node {node:n},{DIRS[dir]:s}", 0))

            # original signal
            signal = np.array([curves[f][node][dir] for f in freq], dtype=complex)

            # resampled signal
            signal_r = resample(time, freq, time_r, freq_r, signal, sampling_frequency,
                                f"Node {node:n},{DIRS[dir]:s}", offset=4)

            # apply butterworth filter if requested
            if butterworth == "low":
                signal_r = butterworth_lowpass_filter(signal_r, fs = sampling_frequency,
                                                      cutoff = bh * freq[-1], offset = 4)
            elif butterworth == "high":
                signal_r = butterworth_highpass_filter(signal_r, fs = sampling_frequency,
                                                       cutoff = bl * freq[0], offset = 4)
            elif butterworth == "band":
                signal_r = butterworth_bandpass_filter(signal_r, fs = sampling_frequency,
                                                       lowcut = bl * freq[0], highcut = bh * freq[-1], offset = 4)

            # PSD
            print(N(f"Creating {node:n},{DIRS[dir]:s} PSD", 4))
            print(N(f"Filter Window: {window_filter:s}", 6))
            print(N(f"Segment Length: {nperseg:n}", 6))

            fyy, Pyy = scipy.signal.welch(signal_r, sampling_frequency, window=window_filter,
                                          nperseg=nperseg)
            Pyy = Pyy[fyy < freq_r[-1]]
            fyy = fyy[fyy < freq_r[-1]]

            # Coherence
            print(N(f"Creating {node:n},{DIRS[dir]:s} Coherence", 4))

            fxy, Cxy = scipy.signal.coherence(stsignal_r, signal_r, fs=sampling_frequency, window=window_filter,
                                            nperseg=nperseg)
            Cxy = Cxy[fxy <= freq[-1]]
            fxy = fxy[fxy <= freq[-1]]
            cmax = np.max(Cxy)
            cmin = np.min(Cxy)
            print(N(f"Maximum up to {fxy[-1]:.2f} Hz: {cmax:.2f}", 6))
            print(N(f"Minimum up to {fxy[-1]:.2f} Hz: {cmin:.2f}", 6))

            x.append([freq, time_r, fyy, fxy])
            y.append([signal.real, signal_r, Pyy, Cxy])
            labels.append([f"Node {node:n},{DIRS[dir]:s}"] * 4)


        png = f"{png_basename:s}_N{node:n}.png" if png_basename is not None else None
        plot_multiple(x, y, labels, titles, xaxis, yaxis,
                      plot_title=f"{unv_file:s}: Driving Node {steuernode:n},{DIRS[shakerdir]:s} vs. Node {node:n}",
                      show_immediately = show_immediately, save = png, offset = 4)

    if (not show_immediately) and show_plots:
        plt.show()



def coherence_directly(unv_file: str, sweep_table: dict, sampling_rate: float,
                       window_filter: str = "hann",
                       print_input: bool = False, show_immediately: bool = False):
    """
    unv_file: (str) path to *.unv file with results
    steuernode: (int) number of the driving node
    shakerdir:  (str) shaker DOF direction x / y / z
    """
    # read the UNV file
    _, _, curves, _ = UNV.read(unv_file)

    print(I("Sweep Table:"))
    print("    +------------+------------+")
    print("    | Freq. [Hz] |  Time [s]  |")
    print("    +------------+------------+")
    for f, t in sweep_table.items():
        print(f"    | {f:10.3E} | {t:10.3E} |")
    print("    +------------+------------+")

    # print read results
    if print_input:
        for frequency, curve in curves.items():
            print(f"[+] {frequency = }")
            for node, vector in curve.items():
                print(f"       {node = } {vector}")

    # get the frequencies, nodes, driving node and shaker direction
    freq = np.array(list(sorted(list(curves.keys()))), dtype=float)
    freq_step = freq[1:] - freq[:-1]
    fs = freq_step[0]

    time = np.interp(freq, np.array(list(sweep_table.keys()), dtype=float),
                           np.array(list(sweep_table.values()), dtype=float))
    time_step = time[1:] - time[:-1]

    nodes = list(sorted([node for  node in curves[freq[0]].keys()]))
    steuernode = nodes.pop(0)    # driving node has to have lowes number
    shakerdir = [i for i in range(3) if curves[freq[0]][steuernode][i].real != 0.0][0]
    stsignal = np.array([curves[f][steuernode][shakerdir] for f in freq], dtype=complex)

    nperseg = NEXT_POW2(stsignal.shape[0] // 100)
    fxx, Pxx = scipy.signal.welch(stsignal.real, fs=1/fs, window=window_filter, nperseg=128)

    print(I(f"Processing Driving Node: {steuernode:n},{DIRS[shakerdir]:s}"))
    # prepare plot headers
    titles = ["Original Signal", "PSD", "Coherence"]
    xaxis = ["frequency [Hz]", "1/frequency [1/Hz]", "1/frequency [1/Hz]"]
    yaxis = ["acceleration [mm/s2]", "PSD [mm2/s4]", "Coherence [-]"]
    for i, node in enumerate(nodes):
        x = [[freq, fxx, None]]
        y = [[stsignal.real, Pxx.real, None]]
        labels = [[f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}",
                   f"Driving Node {steuernode:n},{DIRS[shakerdir]:s}", None]]
        for dir in range(3):
            signal = np.array([curves[f][node][dir] for f in freq], dtype=complex)
            fyy, Pyy = scipy.signal.welch(signal.real, fs=1/fs, window=window_filter, nperseg=128)

            fxyp, Pxy = scipy.signal.csd(stsignal.real, signal.real, fs=1/fs, window=window_filter, nperseg=128)

            fxyc, Cxy = scipy.signal.coherence(stsignal.real, signal.real, fs=1/fs, window=window_filter, nperseg=128)

            x.append([freq, fxyp, fxyc])
            y.append([signal.real, Pxx.real, Cxy])
            labels.append([f"Node {node:n},{DIRS[dir]:s}", f"Node {node:n},{DIRS[dir]:s} PSD", f"Node {node:n},{DIRS[dir]:s}"])

            x.append([None, fxyc, None])
            y.append([None,  Pxy.real, None])
            labels.append([None, f"Node {node:n},{DIRS[dir]:s} CSD", None])

        plot_multiple(x, y, labels, titles, xaxis, yaxis,
                      f"{unv_file:s}: Driving Node {steuernode:n},{DIRS[shakerdir]:s} vs. Node {node:n}",
                      show_immediately)
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

see help for scipy.signal.get_window() for more info.

default: hann

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

    print(I(f"Started script {__file__:s}", 0))
    print(N(f"UNV file:           {args.unv_file:s}", 4))
    print(N(f"Sweep file:         {args.sweep_file:s}", 4))
    print(N(f"Sampling ratio:     {args.sampling_ratio:<.3f}", 4))
    print(N(f"Window filter :     {args.window_filter[0].upper() + args.window_filter[1:].lower():s}", 4))
    print(N(f"Butterworth Filter: {args.butterworth}", 4))
    print(N(f"Print input:        {args.print_input}", 4))
    print(N(f"Show plots:         {args.show_plots}", 4))
    print(N(f"Show immediately:   {args.show_immediately}", 4))
    print(N(f"Save plots:         {args.save_plots}", 4))


    coherence(args.unv_file, args.sweep_file, args.sampling_ratio,
              window_filter = args.window_filter.lower(), butterworth = args.butterworth,
              print_input = args.print_input, show_plots = args.show_plots,
              show_immediately = args.show_immediately, save_plots = args.save_plots)

