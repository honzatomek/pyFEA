#!/usr/bin/python3
"""author:  Jan Tomek
date:    18.05.2023
version: v1.0.0

description:
Campbell diagram plot from time data of Velocity over time and RPM over time,
the input is read from *.unv file datasets number 58."""

import os
import sys
import io
import gzip
import argparse
import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# import utils
# from utils import UNV

FIGSIZE = (12, 8)

warnings.filterwarnings("ignore")
NEXT_POW2 = lambda x: int(1 if x == 0 else 2 ** (x - 1).bit_length())

DOFS = { 0: "",
         1: "+X",
        -1: "-X",
         2: '+Y',
        -2: "-Y",
         3: "+Z",
        -3: "-Z",
         4: "+RX",
        -4: "-RX",
         5: "+RY",
        -5: "-RY",
         6: "+RZ",
        -6: "-RZ"}

COLORS = [( 255, 255, 255),
          ( 169, 168, 244),
          (  80,  80, 252),
          (   0,   0, 255),
          (  15, 255, 232),
          ( 150, 255, 103),
          ( 255, 255,   0),
          ( 255,   0,   0),
          ( 124,   0,   0)]
CMAP = LinearSegmentedColormap.from_list("Campbell",
                                         [[c / 255 for c in color] for color in COLORS],
                                         N=256)



class UNV:
    @classmethod
    def _readline_in_dataset(cls, file: io.TextIOWrapper, dataset: str) -> str:
        """read one line from dataset"""
        line = file.readline()
        if not line: # EOF
            raise IOError(f'[-] File {file.name:s} ended before end of dataset {dataset:s}')
        elif line.strip() == '-1':
            raise IOError(f'[-] File {file.name:s} ended before end of dataset {dataset:s}')
        return line.strip('\n')


    @classmethod
    def _process_line(cls, line: str, field_formats: list) -> list:
        """process string based on list of formats - automatically split line into
        fields of correct length and of correct type"""
        start = 0
        result = []
        for fmt in field_formats:
            fmt_type = fmt[:1]
            fmt_len = int(fmt[1:])
            if 'I' == fmt_type.upper():
                ftype = int
            elif 'A' == fmt_type.upper():
                ftype = str
            elif 'E' == fmt_type.upper():
                ftype = float
            elif 'D' == fmt_type.upper():
                ftype = float
            elif 'X' == fmt_type.upper():
                start += fmt_len
                continue
            else:
                raise NotImplementedError(f'[-] Unknown field format {f:s}')

            if fmt_type.upper() in ('E', 'D'):
                result.append(ftype(line[start:start+fmt_len].upper().replace('D', 'E')))
            elif fmt_type.upper() == 'A':
                result.append(ftype(line[start:start+fmt_len].strip(' ')))
            else:
                result.append(ftype(line[start:start+fmt_len]))
            start += fmt_len
            if start >= len(line) - 1:
                break
        return result


    @classmethod
    def _read_dataset_58(cls, file: io.TextIOWrapper,  results: dict = None) -> dict:
        """read dataset 58 - Function at NDOF"""
        if results is None:
            results = {}

        # delim
        line = file.readline()
        # dataset
        dataset = file.readline().strip()

        if dataset == '58':
            print(f'[+] Reading dataset {dataset:s}')
        else:
            raise IOError(f'Wrong dataset number {dataset:s}')

        try:
            # records 1 - 5
            header = []
            for i in range(5):
                line = cls._readline_in_dataset(file, dataset)
                header.append(line.strip())

            # record 6
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I5', 'I10', 'I5', 'I10', 'X1', 'A10', 'I10', 'I4', 'X1', 'A10', 'I10', 'I4'])
            # func_type, func_id, version_number, lcase_id, resp_ent_name, resp_node, resp_dir, ref_ent_name, ref_node, ref_dir = line
            dof_identification = {k: v for k, v in zip(['func_type', 'func_id', 'version_number',
                                                        'lcase_id', 'resp_ent_name', 'resp_node',
                                                        'resp_dir', 'ref_ent_name', 'ref_node',
                                                        'ref_dir'], line)}

            # record 7
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I10', 'I10', 'I10', 'E13', 'E13', 'E13'])
            data_form = {k: v for k, v in zip(['dattype', 'numval', 'spacing', 'abscissa_min',
                                               'abscissa_inc', 'zaxis'], line)}

            # record 8
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I10', 'I5', 'I5', 'I5', 'X1', 'A20', 'X1', 'A20'])
            abscissa_datachar = {k: v for k, v in zip(['spec_dattype', 'len_unit_exp',
                                                       'frc_unit_exp', 'temp_unit_exp',
                                                       'axis_label', 'axis_units_label'], line)}

            # record 9
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I10', 'I5', 'I5', 'I5', 'X1', 'A20', 'X1', 'A20'])
            ordinate_numerator_datachar = {k: v for k, v in zip(['spec_dattype', 'len_unit_exp',
                                                                 'frc_unit_exp', 'temp_unit_exp',
                                                                 'axis_label', 'axis_units_label'],
                                                                line)}

            # record 10
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I10', 'I5', 'I5', 'I5', 'X1', 'A20', 'X1', 'A20'])
            ordinate_denominator_datachar = {k: v for k, v in zip(['spec_dattype', 'len_unit_exp',
                                                                   'frc_unit_exp', 'temp_unit_exp',
                                                                   'axis_label', 'axis_units_label'],
                                                                  line)}

            # record 11
            line = cls._process_line(cls._readline_in_dataset(file, dataset),
                                     ['I10', 'I5', 'I5', 'I5', 'X1', 'A20', 'X1', 'A20'])
            z_axis_datachar = {k: v for k, v in zip(['spec_dattype', 'len_unit_exp', 'frc_unit_exp',
                                                     'temp_unit_exp', 'axis_label',
                                                     'axis_units_label'], line)}

            fmt = []
            if data_form['spacing'] == 0: # uneven
                if data_form['dattype'] == 2: # real single
                    fmt = ['E13'] * 6
                elif data_form['dattype'] == 4: # real double
                    fmt = ['E13', 'E20'] * 2
                elif data_form['dattype'] == 5: # complex single
                    fmt = ['E13'] * 6
                elif data_form['dattype'] == 6: # complex double
                    fmt = ['E13', 'E20', 'E20']
                else:
                    raise IOError(f'[-] Unknown Oridnate Data Type {data_form["dattype"]:n} value.')
            elif data_form['spacing'] == 1: # even
                if data_form['dattype'] == 2: # real single
                    fmt = ['E13'] * 6
                elif data_form['dattype'] == 4: # real double
                    fmt = ['E20'] * 4
                elif data_form['dattype'] == 5: # complex single
                    fmt = ['E13'] * 6
                elif data_form['dattype'] == 6: # complex double
                    fmt = ['E20'] * 4
                else:
                    raise IOError(f'[-] Unknown Oridnate Data Type {data_form["dattype"]:n} value.')
            else:
                raise IOError(f'[-] Unknown Abscissa Spacing {data_form["spacing"]:n} value.')

            values = []
            while True:
                last_pos = file.tell()
                line = file.readline()
                if not line: # EOF
                    raise IOError(f'[-] File {file.name:s} ended before end of dataset {dataset:s}')
                elif line.strip() == '-1':
                    break

                values.extend(cls._process_line(line, fmt))

        except Exception as e:
            file.close()
            raise e

        # process read data
        x = []
        y = []
        if data_form['spacing'] == 0: # uneven
            if data_form['dattype'] in (2, 4): # real
                for i in range(0, len(values), 2):
                    x.append(values[i])
                    y.append(values[i + 1])
            else:                              # complex
                for i in range(0, len(values), 3):
                    x.append(values[i])
                    y.append(complex(values[i + 1], values[i + 2]))
        else:                         # even
            start = data_form['abscissa_min']
            increment = data_form['abscissa_inc']
            if data_form['dattype'] in (2, 4): # real
                for i in range(len(values)):
                    x.append(start + i * increment)
                    y.append(values[i])
            else:                              # complex
                for i in range(0, len(values), 2):
                    x.append(start + i * increment)
                    y.append(complex(values[i], values[i + 1]))

        data_set = dof_identification['func_id']
        abscissa = abscissa_datachar['axis_label']
        abscissa_units = abscissa_datachar['axis_units_label']

        ordinate = ordinate_numerator_datachar['axis_label']
        ordinate_units = ordinate_numerator_datachar['axis_units_label']

        results[data_set] = {'x': {'label': abscissa,
                                   'units': abscissa_units,
                                   'values': np.array(x, dtype=float)
                                  },
                             'y': {'label': ordinate,
                                   'units': ordinate_units,
                                   'values': np.array(y, dtype=type(y[0]))
                                  },
                             'node': dof_identification['resp_node'],
                             'dir': dof_identification['resp_dir'],
                            }
        return results


    @classmethod
    def read(cls, filename: str) -> dict:
        """reads *.unv file"""
        if os.path.isfile(filename):
            if filename.lower().endswith('.gz'):
                file = gzip.open(filename, 'rt')
            else:
                file = open(filename, 'rt')

        results = {}

        try:
            while True:
                last_pos = file.tell()
                line = file.readline()
                if not line: # EOF
                    break

                if line.strip() == '-1':
                    line = file.readline()
                    if not line: # EOF
                        raise IOError(f'[-] File {file.name:s} ended before end of a dataset.')

                    # dataset number
                    dataset = line.strip()
                    if dataset == '58':
                        file.seek(last_pos)
                        results.update(cls._read_dataset_58(file, results))

                    else:
                        print(f'[!] Skipping dataset {dataset:s}')
                        while True:
                            last_pos = file.tell()
                            line = file.readline()
                            if not line: # EOF
                                raise IOError(f'[-] File {file.name:s} ended before end of dataset {dataset:s}')
                            elif line.strip() == '-1':
                                break

                        continue
                else:
                    continue

        except Exception as e:
            file.close()
            raise e

        file.close()

        return results



def butterworth_lowpass_filter(signal_name: str, signal: np.ndarray,
                               fs: float, cutoff: float) -> np.ndarray:
    """Butterworth Lowpass filter

    Args:
        signal_name (str):   name of the signal data
        signal (np.ndarray): signal to filter
        fs (float):          sample rate [Hz]
        cutoff (float):      desired cutoff frequency of the filter,
                             slightly higher than the desired one
    """

    print(f"[+] Applying Butterworth Low-Pass Filter to signal {signal_name:s}")
    print(f"      Frequency: {cutoff:.2f} Hz")

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



def fft_window(signal: np.ndarray, wlen: int, dt: float = None,
               window: str = "hann", remove_mean: bool = True) -> (np.ndarray, np.ndarray):
    """Perform FFT Analysis of a signal with window function

    Args:
        signal (np.ndarray): array with even spaced signal data vector
        wlen (int):          window length (should be power of 2)
        dt (float):          time increment of the signal
        window (str):        name of window function (default: hann)
        remove_mean (bool):  perform mean removal before FFT

    Returns:
        (np.ndarray, np.ndarray): frequencies, fft of the signal
    """

    # prepare window function (periodic)
    if window is not None:
        window_func = scipy.signal.get_window(window, signal.shape[0], True)
    else:
        window_func = np.array([1.] * signal.shape[0], dtype=float)

    # mean removal
    if remove_mean:
        windowed_signal = (signal - np.mean(signal)) * window_func
    else:
        windowed_signal = signal * window_func

    # append zeros to the end of the signal if needed to force all signals to same length
    if windowed_signal.shape[0] < wlen:
        windowed_signal = np.hstack((windowed_signal, [0.] * (wlen - windowed_signal.shape[0])))

    # time step
    if dt is None:
        dt = 1
        t = np.arange(0, windowed_signal.shape[0])
    else:
        t = np.arange(0, windowed_signal.shape[0]) * dt

    # make signal of even length
    if windowed_signal.shape[0] % 2 != 0:
        t = t[0:-1]
        windowed_signal = windowed_signal[:-1]

    # Nyquist frequency for even sampled data
    nyq = 0.5 * windowed_signal.shape[0]

    # divide by Nyquist frequency for coherent magnitude
    signalFFT = np.fft.fft(windowed_signal) / nyq
    freqsFFT = np.fft.fftfreq(windowed_signal.shape[0], d=dt)

    # store only positive frequencies
    firstNegIdx = np.argmax(freqsFFT < 0)
    freqsFFTpositive = freqsFFT[:firstNegIdx]
    signalFFTpositive = signalFFT[:firstNegIdx]

    return freqsFFTpositive, np.abs(signalFFTpositive)



def plot_lines(signals: dict, savefig: str = None):
    """plots signals as curves

    Args:
        signals (list): signal to plot {sid: {x: {values: np.ndarray,
                                                  label: str,
                                                  units: str},
                                              y: {values: np.ndarray,
                                                  label: str,
                                                  units: str}}}
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for sid, signal in signals.items():
        ax.plot(signal["x"]["values"], signal["y"]["values"], label=f"Dataset: {str(sid):s}")
    ax.set_xlabel(f"{signal['x']['label']:s} [{signal['x']['units']:s}]")
    ax.set_ylabel(f"{signal['y']['label']:s} [{signal['y']['units']:s}]")
    ax.legend()
    fig.suptitle(signal["y"]["label"] + " Time series")
    fig.canvas.set_window_title(signal["y"]["label"] + " Time series")

    if savefig is not None:
        print(f"[+] Saving line plot: {savefig:s}")
        fig.savefig(savefig)



def plot_campbell(rpm: np.ndarray, freqs: np.ndarray, fft: np.ndarray, maxfreq: float = 500.,
                  z_label: str = "Velocity [m/s]", title: str = "Campbell Diagram",
                  num_orders: int = 3, flip_xy: bool = False, savefig: str = None):
    """plot Campbell diagram

    Args:
        rpm (np.ndarray):   rpm vector
        freqs (np.ndarray): frequencies vector
        fft (np.ndarray):   FFT results of the time signal (freq as row, rpm as column)
        maxfreq (float):    maximum frequency to plot
        z_label (str):      label of the Campbell diagram Z component
        num_orders (int):   number of motor order lines to plot
        flip_xy (bool):     if True, X Axis is Frequency, Y Axis is RPM
                            if False, X Axis is RPM, Y Axis is Frequency
        savefig (str):      filename to save the plot to
    """
    print(f"[+] plotting " + ", ".join(title.split("\n")))

    rpm_min = np.min(rpm)
    rpm_max = np.max(rpm)

    # get min index of max frequency to plot
    idx = np.argmin(freqs < maxfreq) + 1

    # reduce the FFT data to only frequencies below max frequency
    zgrid = fft[:idx,:]

    # prepare log levels for colorbar
    level_min = int(np.floor(np.log10(np.min(zgrid))))
    level_max = int(np.floor(np.log10(np.max(zgrid)))) + 1
    levels = np.logspace(level_min, level_max, 35)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # select x and y axis
    if flip_xy:
        # x axis is frequency, y axis is rpm
        xgrid, ygrid = np.meshgrid(freqs[:idx], rpm)
        zgrid = zgrid.T
        surf = ax.contourf(xgrid, ygrid, zgrid, levels=levels,
                           norm=mpl.colors.LogNorm(), cmap=CMAP)
        ax.set_ylim(rpm_min, rpm_max)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("RPM [1/min]")
    else:
        # x axis is rpm, y axis is frequency
        xgrid, ygrid = np.meshgrid(rpm, freqs[:idx])
        zgrid = zgrid
        surf = ax.contourf(xgrid, ygrid, zgrid, levels=levels,
                           norm=mpl.colors.LogNorm(), cmap=CMAP)
        ax.set_xlim(rpm_min, rpm_max)
        ax.set_xlabel("RPM [1/min]")
        ax.set_ylabel("Frequency [Hz]")

    # add lines representing motor orders
    line = []
    for i in range(num_orders):
        if flip_xy:
            line.append(ax.plot((0, (i + 1) * rpm_max / 60.), (0, rpm_max),
                        color="white", lw=1.0, linestyle="dashed"))
        else:
            line.append(ax.plot((0, rpm_max), (0, (i + 1) * rpm_max / 60.),
                        color="white", lw=1.0, linestyle="dashed"))

    # add colorbar in log scale
    cbar = fig.colorbar(surf, ticks=np.power(10., np.arange(level_min, level_max + 1)))
    cbar.set_label(z_label, rotation=90)

    fig.suptitle(title)
    fig.canvas.set_window_title(", ".join(title.split("\n")))

    if savefig is not None:
        print(f"[+] Saving Campbell Diagram: {savefig:s}")
        fig.savefig(savefig)


def order_analysis(signal: np.ndarray, rpm: np.ndarray, time: np.ndarray, signal_name: str = None,
                   window: str = "hann", window_len: int = 4096, overlap: float = 0.5,
                   remove_mean: bool = True, savefig: str = None) -> (np.ndarray, np.ndarray):
    """perform order analysis of the signal

    Args:
        signal (np.ndarray): a vector with signal data
        rpm (np.ndarray):    a vector with respective rpm data
        time (np.ndarray):   a vector with respective times
        signal_name (str):   name of signal for plotting
        window (str):        name of a window function to use
        window_len (int):    number of segments for processing (default = 4096)
        overlap (float):     fraction of the window length to use for overlapping
                             windows
        remove_mean (bool):  perform mean removal before FFT
        savefig (str):       filename to save the plot to
    """
    print(f"[+] Performing Order Analysis for {signal_name:s}")

    num_ord = 10

    # time step and sampling frequency
    dt = np.mean(time[1:] - time[:-1])
    fs = 1 / dt

    # filter the rpm a bit to eliminate errors
    butterworth = min(fs / 4, np.max(rpm) / 60. * num_ord)
    rpm = butterworth_lowpass_filter("RPM", rpm, fs, butterworth)

    # filter the signal
    signal = butterworth_lowpass_filter(signal_name, signal, fs, butterworth)

    # convert rpm to rotations per time step
    rpdt = rpm / 60. * dt
    # integrate to get cummulative rotations
    rot = np.cumsum(rpdt)

    # sampling step in revolutions (max sampling rate)
    dr = np.min(rot[1:] - rot[:-1])
    # create equidistant rotations
    rot_e = np.arange(np.min(rot), np.max(rot) + dr, dr)

    # resample to equidistant rotations
    # sig_e = np.interp(rot_e, rot, signal)
    f = scipy.interpolate.interp1d(rot, signal, kind="cubic",
                                   bounds_error = False, fill_value="extrapolate")
    sig_e = f(rot_e)

    # perform FFT
    # orders, amplitudes = fft_window(sig_e, sig_e.shape[0], dr, None, True)
    # ax1.plot(orders, amplitudes, label="FFT Analysis")
    # ax1.set_title((f"{signal_name:s}: " if signal_name is not None else "") + "Order Analysis")
    # ax1.set_xlabel("Order [-]")
    # ax1.set_xlim((0., 10.))
    # ax1.set_xticks(np.arange(1, 11, 1))
    # ax1.set_ylabel("Velocity [m/s]")

    # perform FFT by windows
    amplitudes = []
    for i in range(0, sig_e.shape[0], int((1. - overlap) * window_len)):
        end = i + window_len if i + window_len <= sig_e.shape[0] else sig_e.shape[0]
        orders, amp = fft_window(sig_e[i:end].flatten(), window_len, dr, window, True)

        amplitudes.append(amp)

    # plot it
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE)

    # plot FFT
    for i in range(len(amplitudes)):
        ax1.plot(orders, amplitudes[i], label=f"window: {i+1:n}")
    if signal_name is None:
        sig_name = "Order Analysis"
        sig_name += f" (BLP {butterworth:.0f}Hz)" if butterworth > 0. else ""
    else:
        sig_name = signal_name
        sig_name += f", BLP {butterworth:.0f}Hz" if butterworth > 0. else ""
        sig_name += f", {window:s}, L={window_len:n} ({overlap * 100:.0f}%)"
        sig_name += ", mean remove" if remove_mean else ""
        sig_name += ": Order Analysis"
    ax1.set_title(sig_name)
    ax1.set_xlabel("Order [-]")
    ax1.set_xlim((0., num_ord))
    ax1.set_xticks(np.arange(0, num_ord + 1, 1))
    ax1.set_ylabel("Velocity [m/s]")
    # ax1.legend()

    # plot resampled signal
    ax2.plot(rot, signal, label="Original Signal")
    ax2.plot(rot_e, sig_e, label="Resampled Signal")
    if signal_name is None:
        sig_name = "Signal Resampled to Angle Domain"
        sig_name += f", BLP {butterworth:.0f}Hz" if butterworth > 0. else ""
    else:
        sig_name = signal_name
        sig_name += f", BLP {butterworth:.0f}Hz" if butterworth > 0. else ""
        sig_name += ": Resampled to Angle Domain"
    ax2.set_title(sig_name)
    ax2.set_xlabel("Rotations [-]")
    ax2.set_xlim((0., np.max(rot_e)))
    ax2.set_ylabel("Velocity [m/s]")
    ax2.legend()

    # fig.suptitle("Order Analysis")
    fig.canvas.set_window_title(f"{signal_name:s} Order Analysis")
    fig.tight_layout()

    if savefig is not None:
        print(f"[+] Saving Order Analysis: {savefig:s}")
        fig.savefig(savefig)

    return rot_e, sig_e



def campbell(signal: np.ndarray, rpm_signal: np.ndarray,
             window: str = "hann", window_len: int = 4096, overlap: float = 0.25,
             remove_mean: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """Process signal data for Campbell Rainflow Diagram using windowed FFT
    Analysis with overlapping.

    Args:
        signal (np.ndarray):     signal array [[time, value]]
        rpm_signal (np.ndarray): RPM array [[time, rpm]]
        window (str):            name of a window function to use
        window_len (int):        number of segments for processing (default = 4096)
        overlap (float):         fraction of the window length to use for overlapping
                                 windows
        remove_mean (bool):      perform mean removal before FFT

    Returns:
        (np.ndarray, np.ndarray, np.ndarry): (rpm vector, frequency vector, FFT array)
    """

    # process signal by segments using window function
    RPM = []
    FFT = []
    # loop over signal by number of segments - overlap * number of segments
    for i in range(0, signal.shape[0], int((1. - overlap) * window_len)):
        end = i + window_len if i + window_len <= signal.shape[0] else signal.shape[0]
        RPM.append(np.mean(rpm_signal[i:end,1]))
        time = signal[i:end,0]
        fs = 1 / np.mean(time[1:] - time[:-1])
        freqsFFT, signalFFT = fft_window(signal[i:end,1].flatten(), window_len,
                                         1. / fs, window, remove_mean)
        FFT.append(signalFFT)

    # sort the results in ascending order of rpm
    RPM = np.array(RPM, dtype=float)
    idx = np.argsort(RPM)
    RPM = RPM[idx]
    FFT = np.array(FFT, dtype=float)[idx].T

    return RPM, freqsFFT, FFT



def create_campbell_diagram(filename: str, window: str = "hann", window_length: int = 4096,
                            window_overlap: float = 0.25, max_freq: float = 500.,
                            butterworth: float = -1., remove_mean: bool = True,
                            flip_xy: bool = False, num_orders: int = 3,
                            showfig: bool = False, savefig: bool = False):
    """read *.unv file and create a Campbell diagram for each dataset 58 found inside

    Args:
        filename (str):          *.unv file to read
        window (str):            window function name for signal processing
        window_length (int):     number of segments to process at one time
        window_overlap (float):  fraction of the window length to use for overlapping
                                 windows
        max_freq (float):        maximum frequency to plot
        butterworth (float):     if > 0. apply Butterworth Lowpass Filter to the signals
                                 before processing them
        remove_mean (bool):      perform mean removal before FFT
        flip_xy (bool):          if True, X Axis is Frequency, Y Axis is RPM
                                 if False, X Axis is RPM, Y Axis is Frequency
        num_orders (int):        number of motor order lines to plot
        showfig (bool):          show plots
        savefig (bool):          save plots to file
    """

    # read data
    print(f"[+] creating Campbell Diagram for {filename:s}")
    results = UNV.read(filename)

    # prepare signals
    rpm = None
    dataids = []
    datasets = []
    labels = []
    titles = []
    curve_names = []
    for cid, curve in results.items():
        label = f"Dataset {cid:n}, Node {curve['node']:n} {DOFS[curve['dir']]:s}"
        x = curve["x"]["values"]
        y = curve["y"]["values"]

        # apply filter if selected
        if butterworth > 0.:
            y = butterworth_lowpass_filter(label, y, 1. / np.mean(x[1:] - x[:-1]), butterworth)
            results[cid]["y"]["values"] = y

        # sort signals to rpm and time signals
        if curve["y"]["label"].upper() == "RPM":
            rpm = np.array([x, y], dtype=float).T

        else:
            dataids.append(cid)
            datasets.append(np.array([x, y], dtype=float).T)
            labels.append(f"{curve['y']['label']:s} [{curve['y']['units']:s}]")
            titles.append(f"Dataset {cid:n}, Node {curve['node']:n} {DOFS[curve['dir']]:s}: Campbell Diagram\n" +
                          f"{window:s}, L = {window_length:n} ({window_overlap * 100:.0f}%)" +
                          (", mean remove" if remove_mean else ""))
            curve_names.append(f"Dataset {cid:n}, Node {curve['node']:n} {DOFS[curve['dir']]:s}")

    # check if time signal is present
    if len(dataids) == 0:
        raise IOError(f"[-] Missing time data.")

    # plot time signal
    plot_lines({k: v for k, v in results.items() if k in dataids},
               (os.path.splitext(filename)[0] +
                (f"_blp{butterworth:.0f}Hz" if butterworth > 0. else "") +
                "_curves.png") if (not showfig) or savefig else None)

    # check if rpm data are present
    if rpm is None:
        raise IOError(f"[-] Missing RPM data.")

    # plor rpm
    plot_lines({k: v for k, v in results.items() if k not in dataids},
               (os.path.splitext(filename)[0] +
                (f"_blp{butterworth:.0f}Hz" if butterworth > 0. else "") +
                "_rpm.png") if (not showfig) or savefig else None)

    # process signals
    for i in range(len(dataids)):
        if (not showfig) or savefig:
            # prepare plot name
            plotname = f"{os.path.splitext(filename)[0]:s}_{str(dataids[i]):s}"
            plotname += f"_{window:s}_{window_length:n}_{window_overlap*100:.0f}%"
            plotname += f"_{max_freq:.0f}Hz"
            plotname += f"_blp{butterworth:.0f}Hz" if butterworth > 0. else ""
            plotname += "_campbell.png"
        else:
            plotname = None
        print(f"[+] creating Campbell Diagram for dataset {str(dataids[i]):s}")
        rpms, freqs, FFT = campbell(datasets[i], rpm, window, window_length,
                                    window_overlap, remove_mean)
        print(f"[+] plotting Campbell Diagram for dataset {str(dataids[i]):s}")
        plot_campbell(rpms, freqs, FFT, max_freq, labels[i], titles[i], num_orders, flip_xy, plotname)

    # perform order analysis
    for i in range(len(dataids)):
        if (not showfig) or savefig:
            # prepare plot name
            plotname = f"{os.path.splitext(filename)[0]:s}_{str(dataids[i]):s}"
            plotname += f"_{window:s}_{window_length:n}_{window_overlap*100:.0f}%"
            plotname += f"_blp{butterworth:.0f}Hz" if butterworth > 0. else ""
            plotname += "_order_analysis.png"
        else:
            plotname = None
        order_analysis(datasets[i][:,1], rpm[:,1], datasets[i][:,0], curve_names[i],
                       window, window_length, window_overlap, remove_mean, plotname)

    if showfig:
        plt.show()

    sys.exit()



if __name__ == "__main__":
    # filename = "./res/Derotator_Stihl_2017/unv-Format/25-RunDown-336mm.unv"
    # results = UNV.read_campbell(filename)
    # velocity = np.array([results[1]['x']['values'],
    #                      results[1]['y']['values']], dtype=float).T
    # rpm      = np.array([results[3]['x']['values'],
    #                      results[3]['y']['values']], dtype=float).T
    # rpms, freqs, FFT = campbell(velocity, rpm, "hann", 4096, 0.25)
    # plot_campbell(rpms, freqs, FFT, 500., "Velocity [m/s]", 3, True,
    #               os.path.splitext(filename)[0] + "_campbell.png")
    # plt.show()

    # Create options parser object.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-w", "--window", dest="window", type=str, default="hann",
                        help="""Window filter type for signal processing.

possible filters:
    boxcar, triang, blackman, hann, hamming, bartlett, flattop, parzen, bohman,
    blackmanharris, nuttall, barthann, cosine

see help for scipy.signal.get_window() for more info.

default: hann

""")

    parser.add_argument("-n", "--number_of_points_per_segment", dest="nperseg", type=int, default=4096,
                        help="""Number of data points per segment for signal processing using the window function

    nperseg = 2 ** (nperseg - 1).bit_length()  # number of time steps per segment length should be a power of 2

default: 4096

""")

    parser.add_argument("-r", "--overlap", dest="overlap", type=float, default=0.5,
                        help="""Fraction of the window length to use for overlapping windows <0, 0.5>

default: 0.5

""")

    parser.add_argument("-f", "--max_freq", dest="max_freq", type=float, default=500.,
                        help="""Maximum Frequency in Hz to plot

default: 500.

""")

    parser.add_argument("-b", "--butterworth", dest="butterworth", type=float, default=-1.,
                        help="""Apply Butterworth Lowpass Filter to the signal data before processing
if the value is > 0. [Hz]

default: -1.

""")

    parser.add_argument("-m", "--no_mean_removal", dest="mean_removal", action="store_false", default=True,
                        help="""If switched then do not perform mean removal of the signal before FFT Analysis

default: False

""")

    parser.add_argument("-x", "--flip_xy", dest="flip_xy", action="store_true", default=False,
                        help="""If the diagram plot should be transposed - Frequency on X axis and RPM on Y axis.

default: False

""")

    parser.add_argument("-o", "--number_of_orders", dest="nord", type=int, default=3,
                        help="""Number of order lines to plot

default: 3

""")

    parser.add_argument("-p", "--show_plots", dest="show_plots", action="store_true", default=False,
                        help="""If the plots should be shown, default: do not show

default: False

""")

    parser.add_argument("-s", "--do_not_save_plots", dest="save_plots", action="store_false", default=True,
                        help="""Save plots as *.png, the name of the *.png is taken from the
*.unv file, resulting filename is {unv_path}/{unv_file}_{dataset_number}_campbell.png.
The locaction of the plot is the same as the *.unv file provided.
If show_plots is not on, plots are automatically saved.

""")

    parser.add_argument("data_file", type=str,
                        help="""Path to *.unv file with results. Must contain at least
2 datasets type 58 (one for e.g. Velocity over Time, second for RPM over time).

Example of input dataset 58:

>>>    -1
>>>    58
>>>Response Time Trace
>>>Vib  Geschwindigkeit
>>>13-Dez-17 13:27:47  
>>>25-RunDown-336mm.pvd
>>>NONE
>>>    1         1    1         0       NONE         1   3       NONE         1   3
>>>         2    217696         1  0.00000e+00  1.00000e-04  0.00000e+00
>>>        17    0    0    0                 Time                    s
>>>        11    0    0    0             Velocity                  m/s
>>>         0    0    0    0                 NONE                 NONE
>>>         0    0    0    0                 NONE                 NONE
>>>  1.76129e-01  1.87442e-01  1.98340e-01  2.08079e-01  2.18242e-01  2.30733e-01
>>>  2.37783e-01  2.39642e-01  2.44000e-01  2.50941e-01  2.50596e-01  2.46631e-01
>>>  2.47287e-01  2.47129e-01  2.38452e-01  2.26929e-01  2.22206e-01  2.16598e-01
>>>  2.02887e-01  1.89058e-01  1.77851e-01  1.66961e-01  1.49114e-01  1.37092e-01
                                        .
                                        .
                                        .
>>>  6.59472e-03  6.40927e-03  5.94897e-03  5.86853e-03  4.94813e-03  4.34678e-03
>>>  4.57091e-03  4.97320e-03  5.06980e-03  4.83748e-03
>>>    -1
>>>    -1
>>>    58
>>>Response Time Trace
>>>Ref2  Drehzahl
>>>13-Dez-17 13:27:47  
>>>25-RunDown-336mm.pvd
>>>NONE
>>>    1         3    1         0       NONE         1   3       NONE         1   3
>>>         2    217696         1  0.00000e+00  1.00000e-04  0.00000e+00
>>>        17    0    0    0                 Time                    s
>>>        19    0    0    0                  RPM                  rpm
>>>         0    0    0    0                 NONE                 NONE
>>>         0    0    0    0                 NONE                 NONE
>>>  7.05514e+03  7.06301e+03  7.05156e+03  7.04867e+03  7.04910e+03  7.04479e+03
>>>  7.04726e+03  7.04608e+03  7.04185e+03  7.04422e+03  7.04295e+03  7.03751e+03
>>>  7.04110e+03  7.03654e+03  7.03912e+03  7.04085e+03  7.03590e+03  7.02542e+03
>>>  7.01912e+03  7.02331e+03  7.02117e+03  7.01845e+03  7.01919e+03  7.03180e+03
                                        .
                                        .
                                        .
>>>  1.21571e+03  1.21572e+03  1.21163e+03  1.21508e+03  1.21637e+03  1.21858e+03
>>>  1.23239e+03  1.23126e+03  1.22743e+03  1.23002e+03
>>>    -1

""")

    # Parse command-line arguments.
    args = parser.parse_args()

    data_file = os.path.realpath(args.data_file)

    print(f"[+] Started {os.path.basename(__file__):s}")
    print(f"      Filename:     {data_file:s}")
    print(f"      Window:       {args.window:s}")
    print(f"      NPERSEG:      {args.nperseg:n}")
    print(f"      Overlap:      {args.overlap:.2f}")
    print(f"      Max Freq:     {args.max_freq:.2f}")
    print(f"      Filter:       " + (f"Butterworth Lowpass {args.butterworth:.2f}" if args.butterworth > 0. else "None"))
    print(f"      Mean Removal: {str(args.mean_removal):s}")
    print(f"      Flip X & Y:   {str(args.flip_xy):s}")
    print(f"      N° Orders:    {args.nord:n}")
    print(f"      Show plots:   {str(args.show_plots):s}")
    print(f"      Save plots:   {str(args.show_plots):s}")

    # filename = "./res/Derotator_Stihl_2017/unv-Format/25-RunDown-336mm.unv"
    create_campbell_diagram(filename=data_file,
                            window=args.window,
                            window_length=args.nperseg,
                            window_overlap=args.overlap,
                            max_freq=args.max_freq,
                            butterworth=args.butterworth,
                            remove_mean=args.mean_removal,
                            flip_xy=args.flip_xy,
                            num_orders=args.nord,
                            showfig=args.show_plots,
                            savefig=args.save_plots)
