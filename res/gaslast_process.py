#!/usr/bin/python3
'''
Script to process ADAMS *.asc Gaslast file, plot all rotations of KUW and export
mean values of Gaslast for one specific RPM Speed

Reads data in the following format:

Kurbelwinkel	p_Zyl	FILENAME	
Grad	bar		
-180,0	1,423	TB01T10089-017.001	
-179,0	1,442	TB01T10089-017.001	
'''


# <------------------------------------------------------------------------------------------------ general imports --->
import os
import sys
import re
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from subprocess import check_output


# <------------------------------------------------------------------------------------------------- regex patterns --->
# Kurbelwinkel	p_Zyl	FILENAME	
# Grad	bar		
# -180,0	1,423	TB01T10089-017.001	
# -179,0	1,442	TB01T10089-017.001	
line_check = [re.compile(r'Kurbelwinkel\s+p_Zyl\s+FILENAME\s*', flags=re.I),
              re.compile(r'Grad\s+bar\s*', flags=re.I),
              re.compile(r'^\s*-?\d+[,.]?\d+\s+\d+[,.]?\d+\s+\S+\s*', flags=re.I)]


# <----------------------------------------------------------------------------------------------- global variables --->
FILE_EXT = '_ref.txt'
BILD_EXT = '_ref.png'
BILD_DIR = 'Bildern'


# console colors foreground
CF = {'no_color': '',
      'black': 30,
      'red': 31,
      'green': 32,
      'yellow': 33,
      'blue': 34,
      'purple': 35,
      'cyan': 36,
      'white': 37}

# console colors background
CB = {'no_color': '',
      'black': 40,
      'red': 41,
      'green': 42,
      'yellow': 43,
      'blue': 44,
      'purple': 45,
      'cyan': 46,
      'white': 47}

# console text format
CT = {'normal': 0,
      'bold': 1,
      'underline': 4}


# <-------------------------------------------------------------------------------------------------- console_print --->
def console_color(foreground_color=None, background_color=None, bold=False, underline=False):
    retval = '\033['

    # background color
    if background_color in CB.keys():
        retval += str(CB[background_color]) + ';'

    # text format
    text_type = CT['normal']
    if bold:
        text_type += CT['bold']
    if underline:
        text_type += CT['underline']
    retval += str(text_type)

    # foreground color
    if foreground_color in CF.keys():
        retval += ';' + str(CF[foreground_color])
    retval += 'm'

    return retval


TAG = {'info': '[' + console_color('cyan', bold=True) + 'i' + console_color() + ']',
       'ok': '[' + console_color('green', bold=True) + '+' + console_color() + ']',
       'error': '[' + console_color('red', bold=True, underline=True) + '-' + console_color() + ']',
       'question': '[' + console_color('yellow', bold=True) + '?' + console_color() + ']',
       'verbose': '[' + console_color('purple', bold=True) + 'v' + console_color() + ']'}


def console_print(tag: str = 'info', message: str = None, level: int = 0):
    if tag not in TAG.keys():
        for t in TAG.keys():
            if tag[0] == t[0]:
                tag = t
                break

    if level < 1:
        padding = ''
    else:
        padding = ' ' * int(level) * 2

    if tag not in TAG.keys():
        print('{0}[ ] {1}'.format(padding, message))
    else:
        print('{0}{1} {2}'.format(padding, TAG[tag], message))


# <------------------------------------------------------------------------------------------------- custom errors --->
class InvalidInputError(Exception):
    pass


# <------------------------------------------------------------------------------------------------------ argparse --->
class CheckTXT(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        # if nargs is not None:
        #     raise ValueError("nargs not allowed")
        super(CheckTXT, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        # console_print('info', 'Check {0} file(s):'.format(self.metavar))

        for value in values:
            console_print('info', 'Check {0} file {1}:'.format(self.metavar,  value))
            # check whether exists
            if not os.path.isfile(value):
                console_print('error', 'File exists: False', 1)
                raise InvalidInputError('{0} must be an existing file.'.format(value))
            else:
                console_print('ok', 'exists: True', 1)

            # check whether is *.txt file
            if not value.upper().endswith('.TXT'):
                console_print('error', 'is TXT: False', 1)
                raise InvalidInputError('{0} must be a *.txt file.'.format(value))
            else:
                console_print('ok', 'is TXT: True', 1)

            # check contents
            with open(value, 'r', encoding='utf8') as gd:
                for i in range(3):
                    line = gd.readline().strip('\n')
                    m = re.match(line_check[i], line)
                    if not m:
                        console_print('error', 'correct format: False', 1)
                        raise InvalidInputError('{0} has incorrect format of line {1}. - {2}'.format(value, i + 1, line))
            console_print('ok', 'correct format: True', 1)

        setattr(namespace, self.dest, values)


# <------------------------------------------------------------------------------------------------- process curves --->
class Gasdruck:

    @staticmethod
    def read(file: str, encoding: str = 'utf8'):
        console_print('info', 'Reading curve data...', 1)
        with open(file, 'r', encoding=encoding) as f:
            raw_data = f.read().split('\n')

        for i in range(len(raw_data)):
            m = re.match(line_check[2], raw_data[i])
            if m:
                break
        raw_data = raw_data[i:]
        while raw_data[-1] == '':
            raw_data = raw_data[:-1]

        for i in range(len(raw_data)):
            raw_data[i] = [float(d.replace(',', '.')) for d in raw_data[i].split()[:2]]

        console_print('ok', 'Curve data read.', 1)
        return raw_data

    @staticmethod
    def _get_resolution(dpi=100):
        stdout = check_output("xrandr").decode()
        for line in stdout.split('\n'):
            if '*' in line:
                return [float(x) / dpi for x in line.strip('  ').split(' ')[0].split('x')]
        return None

    def __init__(self, gasdruck_file):
        self.file = str(gasdruck_file)
        self.raw_data = self.__class__.read(self.file)

        self.type = None
        self.data = None
        self.mean = None
        self.average = None
        # self.average = np.mean(self.data, axis=0, dtype=float)

        # main figure
        dpi = 100
        winsize = self.__class__._get_resolution(dpi)
        if not winsize:
            winsize = [18.5, 9.5]
        # print(winsize)
        self.figure = plt.figure(figsize=(winsize[0] - 1.0, winsize[1] - 1.5), dpi=dpi)
        self.figure.canvas.set_window_title('Gasdruck file {0}'.format(self.file))

        # prepare axes variables
        self.ax_main = None

        # prepare plotted curves variables
        self.curves = list()
        self.lines = list()

        # prepare curve label variables
        self.curve_labels = list()
        self.line_labels = list()

        # prepare legend variables
        self.legends = list()

        # positions: [left, bottom, width, height]
        self.p_ax_main = [0.05, 0.10, 0.90, 0.85]

    def process(self, rotations: list = [[0, -1]], min_pressure: float = 28., one_based=False):
        console_print('info', 'Processing curve data filtered...', 1)

        self.data = []

        if one_based:
            # discard data below first 1° angle
            offset = int((self.raw_data[0][0] + 1) % 360)
            self.type = 'one'
        else:
            # discard data below first -180° angle
            offset = 360 - int((self.raw_data[0][0] + 180) % 360)
            self.type = '-180'

        # print('Offset: {0}'.format(offset))

        if offset == 360:
            offset = 0

        data = self.raw_data[offset:]
        # discard data of incomplete last revolution
        offset = int(len(data) / 360) * 360
        data = data[:offset]
        data = np.array(data, dtype=float)

        # print('Type: {0}'.format(self.type))

        # split the data into separate KUW revolutions
        for i in range(int(len(data) / 360.0)):
            x = 360 * i
            y = 360 * i + 360
            self.data.append([d[1] for d in data[x:y]])

        data_filtered = []
        for rotation_range in rotations:
            s = max(0, rotation_range[0] if rotation_range[0] >= 0 else len(self.data) + rotation_range[0])
            e = min(len(self.data), rotation_range[1] + 1 if rotation_range[1] >= 0 else len(self.data) + (rotation_range[1] + 1))
            for i in range(s, e):
                # console_print('info', f'Adding rotation {i:n}')
                if np.max(self.data[i]) > min_pressure:
                    data_filtered.append(self.data[i])

        self.data = np.array(data_filtered, dtype=float)

        self.mean = np.mean(self.data, axis=0, dtype=float)
        self.average = np.average(self.data, axis=0)

        console_print('info', 'Curve data processed (max = {0} bar).'.format(np.max(data)), 1)

    @staticmethod
    def _round_base(x, base=5):
        return int(base * round(float(x) / base))

    def get_ax_size(self, axe):
        bbox = axe.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= self.figure.dpi
        height *= self.figure.dpi

        return width, height

    def get_ax_limits(self, axe):
        x0, x1 = axe.get_xlim()
        y0, y1 = axe.get_ylim()
        return [x0, y0], [x1, y1]

    def plot_line(self, axe, x, y, label, color, dashes, ratio=0.85, lineweight=1.5):
        xtext, ytext, angle = self.text_attr(axe, x, y, label, ratio)
        line, = axe.plot(x, y, color=color, dashes=dashes, lw=lineweight)
        text = axe.text(xtext, ytext, label, ha='left', va='bottom', fontsize='x-large', rotation=angle)
        return line, text

    def plot(self, y_max=-1, save=True):
        console_print('info', 'Plotting...', 1)

        self.ax_main = self.figure.add_axes(self.p_ax_main)
        # self.legends.append(None)
        # self.curve_labels.append(None)

        if self.type == 'one':
            x_axis = np.array([float(x) for x in range(1, 361)], dtype=float)
        else:
            x_axis = np.array([float(x) for x in range(-180, 180)], dtype=float)

        for y_axis in self.data:
            self.curves.append(None)
            self.curves[-1], = self.ax_main.plot(x_axis, y_axis, lw=0.5, color='blue') #, label='{0} - {1:.1f} Hz'.format(l[i][0], y[i][0]))

        self.curves.append(None)
        self.curves[-1], = self.ax_main.plot(x_axis, self.mean, lw=2.0, color='red') #, label='{0} - {1:.1f} Hz'.format(l[i][0], y[i][0]))

        # self.curves.append(None)
        # self.curves[-1], = self.ax_main.plot(x_axis, self.mean, lw=2.0, color='violet') #, label='{0} - {1:.1f} Hz'.format(l[i][0], y[i][0]))

        self.ax_main.set_title('Gasdruck file {0}'.format(self.file))

        self.ax_main.set_xlabel('Angle [°]')
        if self.type == 'one':
            self.ax_main.set_xlim(xmin=0.0, xmax=360.0)
        else:
            self.ax_main.set_xlim(xmin=-180.0, xmax=180.0)
        self.ax_main.xaxis.set_major_locator(ticker.MultipleLocator(30.0))

        self.ax_main.set_ylabel('Gasdruck [bar]')
        if y_max == -1:
            self.ax_main.set_ylim(ymin=0.0, ymax=float(int(np.max(self.data)) + 1))
        else:
            self.ax_main.set_ylim(ymin=0.0, ymax=float(y_max))

        # self.figure.text(0.025, 0.05, 'Eigenfrequency filter: f(max) - f(min) > {0:.2f} Hz'.format(self.filter))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        console_print('ok', 'Curve plotted.', 1)

        if save:
            self.save_figure()

        # plt.show()

    def save_figure(self):
        bildpath = os.path.join(os.getcwd(), BILD_DIR)
        console_print('info', "Saving plot to '{0}'.".format(bildpath), 1)

        if not os.path.isdir(bildpath):
            console_print('info', "Picture directory does not exist, creating '{0}'.".format(bildpath), 2)
            os.mkdir(bildpath)

        if self.type == 'one':
            new_name = os.path.join(bildpath, os.path.splitext(os.path.basename(self.file))[0] + '_1' + BILD_EXT)
        else:
            new_name = os.path.join(bildpath, os.path.splitext(os.path.basename(self.file))[0] + '_180' + BILD_EXT)

        # plt.savefig(new_name)
        plt.draw()
        self.figure.savefig(new_name)
        console_print('ok', "Plot saved as '{0}'.".format(new_name), 1)

    def export(self):
        if self.type == 'one':
            new_name = os.path.splitext(os.path.basename(self.file))[0] + '_1' + FILE_EXT
        else:
            new_name = os.path.splitext(os.path.basename(self.file))[0] + '_180' + FILE_EXT

        console_print('info', "Exporting mean values as '{0}'.".format(new_name), 1)

        # if os.path.isfile(new_name):
        #     console_print('error', "Filename '{0}' already exists, delete or rename it if you want to proceed.".format(new_name))
        #     raise OSError("Filename '{0}' already exists, delete or rename it if you want to proceed.".format(new_name))

        if self.type == 'one':
            x_axis = np.array([float(x) for x in range(1, 361)], dtype=float)
        else:
            x_axis = np.array([float(x) for x in range(-180, 180)], dtype=float)

        with open(new_name, 'w', encoding='utf8') as file:
            for i in range(len(self.mean)):
                file.write('{0:.1f}\t{1:3f}\n'.format(x_axis[i], self.mean[i]))

        console_print('ok', "Mean values exported as '{0}'.".format(new_name), 1)

    def max(self):
        return np.max(self.data)


# <-------------------------------------------------------------------------------------------------- main function --->
def main(gasdruck_files, rpm_range = [[0, -1]], filter = 0., one_based=False, no_plot=False, save=True):
    if not save:
        no_plot = False

    gasdruck = []
    y_max = 0.0
    for file in gasdruck_files:
        console_print('info', "'{0}':".format(file))
        gasdruck.append(Gasdruck(file))
        gasdruck[-1].process(rpm_range, filter, one_based)
        max = gasdruck[-1].max()
        if y_max < max:
            y_max = max

    y_max = float(int(y_max) + 1.0)
    # print('y_max: {0}'.format(y_max))

    for g in gasdruck:
        console_print('info', "'{0}':".format(g.file))
        g.plot(y_max, save)
        if save:
            g.export()

    if not no_plot:
        plt.show()


# <------------------------------------------------------------------------------------------------ main entrypoint --->
if __name__ == '__main__':
    print('{0} Started script: {1}'.format(TAG['info'], os.path.basename(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, epilog='[CWD: {0}]\n'.format(os.getcwd()),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--rpm_range', action='append', nargs=2, type=int, metavar=('start', 'end'), default=[[0, -1]],
                        help='limit rpm range to [start, end] rotations. Can be used multiple times to specify more ranges, default = [0, -1]')

    parser.add_argument('-f', '--filter', nargs=1, type=float, metavar=('min_pressure'), default=[0.],
                        help='specify the minimal pressure for the rotation to be added to evaluation (default = 0.).')


    parser.add_argument('-o', '--one_based', action='store_true',
                        help='Plot Gasdruck data as one based (1° - 360°) instead of (-180° - 179°).')

    parser.add_argument('-n', '--no_plot', action='store_true',
                        help='Do not plot Gasdruck in app, only create picture.')

    parser.add_argument('-e', '--no_export', action='store_true',
                        help='Do not save any data on disk (if -n is used, it is ignored).')

    parser.add_argument('files', metavar='Gasdruck', nargs='+', type=str, action=CheckTXT,
                        help='*.txt file(s) with Gasdruck data for one RPM speed.')

    args = parser.parse_args()

    main(args.files, args.rpm_range, args.filter[0], args.one_based, args.no_plot, not args.no_export)

