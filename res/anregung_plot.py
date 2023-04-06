#!/usr/bin/python3
'''
Script to process ADAMS *.asc Gaslast file, plot all rotations of KUW and export
mean values of Gaslast for one specific RPM Speed

Reads data in following format:

Kurbelwinkel	p_Zyl_avg	p_Zyl_min	p_Zyl_max	N_AVG	FILENAME	pmi_avg	
Grad	bar	bar	bar	1/min		bar	
-180,0	1,191	1,157	1,189	5491,5	TB01T10089-017.002	4,7420	
-179,0	1,192	1,112	1,138	5491,5	TB01T10089-017.002	4,7420	
                                  .
                                  .
                                  .
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
# Kurbelwinkel	p_Zyl_avg	p_Zyl_min	p_Zyl_max	N_AVG	FILENAME	pmi_avg	
# Grad	bar	bar	bar	1/min		bar	
# -180,0	1,191	1,157	1,189	5491,5	TB01T10089-017.002	4,7420	
# -179,0	1,192	1,112	1,138	5491,5	TB01T10089-017.002	4,7420	
line_check = [re.compile(r'Kurbelwinkel\s+p_Zyl_avg\s+p_Zyl_min\s+p_Zyl_max\s+N_AVG\s+FILENAME\s+pmi_avg\s*', flags=re.I),
              re.compile(r'Grad\s+bar\s+bar\s+bar\s+1/min\s+bar\s*', flags=re.I),
              re.compile(r'^\s*-?\d+,?\d+(\s+\d+,?\d+)+\s+\S+\s*', flags=re.I)]


# <----------------------------------------------------------------------------------------------- global variables --->
FILE_EXT = '_ref.txt'
BILD_EXT = '_ref.png'
BILD_DIR = 'Bildern'


# <--------------------------------------------------------------------------------------------------- help classes --->
class Console:
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

    def __init__(self):
        pass

    @classmethod
    def color(cls, foreground_color=None, background_color=None, bold=False, underline=False):
        retval = '\033['

        # background color
        if background_color in cls.CB.keys():
            retval += str(cls.CB[background_color]) + ';'

        # text format
        text_type = cls.CT['normal']
        if bold:
            text_type += cls.CT['bold']
        if underline:
            text_type += cls.CT['underline']
        retval += str(text_type)

        # foreground color
        if foreground_color in cls.CF.keys():
            retval += ';' + str(cls.CF[foreground_color])
        retval += 'm'

        return retval

    @classmethod
    def print(cls, tag, message, level=0):
        if tag.lower() in ['info', 'i']:
            TAG = '[' + cls.color('cyan', bold=True) + 'i' + cls.color() + ']'
        elif tag.lower() in ['ok', 'o']:
            TAG = '[' + cls.color('green', bold=True) + '+' + cls.color() + ']'
        elif tag.lower() in ['error', 'e']:
            TAG = '[' + cls.color('red', bold=True, underline=True) + '-' + cls.color() + ']'
        elif tag.lower() in ['question', 'q']:
            TAG = '[' + cls.color('yellow', bold=True) + '?' + cls.color() + ']'
        elif tag.lower() in ['verbose', 'v']:
            TAG = '[' + cls.color('purple', bold=True) + 'v' + cls.color() + ']'
        else:
            TAG = '[' + cls.color('cyan', bold=True) + '.' + cls.color() + ']'

        if '\n' in message:
            message = message.replace('\n', '\n' + '  ' * int(level) + '    ')

        print('{0}{1} {2}'.format('  ' * int(level), TAG, message))


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
        # Console.print('info', 'Check {0} file(s):'.format(self.metavar))

        for value in values:
            Console.print('info', 'Check {0} file {1}:'.format(self.metavar,  value))
            # check whether exists
            if not os.path.isfile(value):
                Console.print('error', 'File exists: False', 1)
                raise InvalidInputError('{0} must be an existing file.'.format(value))
            else:
                Console.print('ok', 'exists: True', 1)

            # check whether is *.txt file
            if not value.upper().endswith('.TXT'):
                Console.print('error', 'is TXT: False', 1)
                raise InvalidInputError('{0} must be a *.txt file.'.format(value))
            else:
                Console.print('ok', 'is TXT: True', 1)

            # check contents
            with open(value, 'r', encoding='utf8') as gd:
                for i in range(3):
                    line = gd.readline().strip('\n')
                    m = re.match(line_check[i], line)
                    if not m:
                        Console.print('error', 'correct format: False', 1)
                        raise InvalidInputError('{0} has incorrect format of line {1}. - {2}'.format(value, i + 1, line))
            Console.print('ok', 'correct format: True', 1)

        setattr(namespace, self.dest, values)


# <------------------------------------------------------------------------------------------------- process curves --->
class Gasdruck:

    @staticmethod
    def read(file: str, encoding: str = 'utf8'):
        Console.print('info', 'Reading curve data...', 1)
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
            raw_data[i] = [float(d.replace(',', '.')) for d in raw_data[i].split()[:-3]]

        # for i in range(10):
        #     print(raw_data[i])

        Console.print('ok', 'Curve data read.', 1)
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

        self.kurbelwinkel = None
        self.p_max = None
        self.p_min = None
        self.p_avg = None

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
        self.p_ax_main = [0.05, 0.10, 0.80, 0.85]

    def process(self):
        Console.print('info', 'Processing curve data...', 1)

        self.kurbelwinkel = []
        self.p_max = []
        self.p_min = []
        self.p_avg = []

        self.kurbelwinkel = np.array([d[0] for d in self.raw_data], dtype=float)

        data = np.array([d[1:4] for d in self.raw_data], dtype=float)
        # print(data)
        max = np.max(data, axis=0)
        # print(max)

        for i in range(3):
            if max[i] == np.max(max):
                self.p_max = data[:, i]
                break
        # print(self.p_max)

        for j in range(3):
            if j == i:
                continue
            if max[j] == np.min(max):
                self.p_min = data[:, j]
                break
        # print(self.p_min)

        for k in range(3):
            if k != i and k != j:
                self.p_avg = data[:, k]
                break
        # print(self.p_avg)

        Console.print('info', 'Curve data processed (max = {0} bar).'.format(np.max(max)), 1)

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

    def plot(self, y_max=-1, save=False):
        Console.print('info', 'Plotting...', 1)

        self.ax_main = self.figure.add_axes(self.p_ax_main)
        self.legends.append(None)
        self.curve_labels.append(None)

        # print(self.kurbelwinkel)
        # print(self.p_max)
        self.curves.append(None)
        self.curves[-1], = self.ax_main.plot(self.kurbelwinkel, self.p_max, lw=2.0, color='red', label='{0} - {1:.1f} bar'.format('p_Zyl_max', np.max(self.p_max)))

        # print(self.kurbelwinkel)
        # print(self.p_avg)
        self.curves.append(None)
        self.curves[-1], = self.ax_main.plot(self.kurbelwinkel, self.p_avg, lw=2.0, color='blue', label='{0} - {1:.1f} bar'.format('p_Zyl_avg', np.max(self.p_avg)))

        # print(self.kurbelwinkel)
        # print(self.p_min)
        self.curves.append(None)
        self.curves[-1], = self.ax_main.plot(self.kurbelwinkel, self.p_min, lw=2.0, color='green', label='{0} - {1:.1f} bar'.format('p_Zyl_min', np.max(self.p_min)))

        self.ax_main.set_title('Gasdruck file {0}'.format(self.file))

        self.ax_main.set_xlabel('Angle [°]')
        self.ax_main.set_xlim(xmin=self.kurbelwinkel[0], xmax=self.kurbelwinkel[-1])
        self.ax_main.xaxis.set_major_locator(ticker.MultipleLocator(30.0))

        self.ax_main.set_ylabel('Gasdruck [bar]')
        if y_max == -1:
            self.ax_main.set_ylim(ymin=0.0, ymax=float(int(np.max(self.p_max)) + 1))
        else:
            self.ax_main.set_ylim(ymin=0.0, ymax=y_max)

        bbox = [self.p_ax_main[0] + self.p_ax_main[2] + 0.005, self.p_ax_main[1] + self.p_ax_main[3]]
        self.ax_main.legend(handles=self.curves, title='Gasdruck:',
                            loc='upper left', bbox_transform=self.figure.transFigure,
                            bbox_to_anchor=bbox,
                            ncol=int(len(self.curves)/65 + 1), fontsize='medium') #, mode = 'expand')

        # self.figure.text(0.025, 0.05, 'Eigenfrequency filter: f(max) - f(min) > {0:.2f} Hz'.format(self.filter))

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        Console.print('ok', 'Curve plotted.', 1)

        if save:
            self.save_figure()

        # plt.show()

    def save_figure(self):
        bildpath = os.path.join(os.getcwd(), BILD_DIR)
        Console.print('info', "Saving plot to '{0}'.".format(bildpath), 1)

        if not os.path.isdir(bildpath):
            Console.print('info', "Picture directory does not exist, creating '{0}'.".format(bildpath), 2)
            os.mkdir(bildpath)

        new_name = os.path.join(bildpath, os.path.splitext(os.path.basename(self.file))[0] + '_180' + BILD_EXT)

        # plt.savefig(new_name)
        plt.draw()
        self.figure.savefig(new_name)
        Console.print('ok', "Plot saved as '{0}'.".format(new_name), 1)

    def export(self):
        new_name = os.path.splitext(os.path.basename(self.file))[0] + '_1' + FILE_EXT
        Console.print('info', "Exporting mean values as '{0}'.".format(new_name), 1)

        # if os.path.isfile(new_name):
        #     Console.print('error', "Filename '{0}' already exists, delete or rename it if you want to proceed.".format(new_name))
        #     raise OSError("Filename '{0}' already exists, delete or rename it if you want to proceed.".format(new_name))

        with open(new_name, 'w', encoding='utf8') as file:
            for i in range(181, len(self.p_avg)):
                file.write('{0:.1f}\t{1:3f}\n'.format(self.kurbelwinkel[i], self.p_avg[i]))
            for i in range(0, 181):
                file.write('{0:.1f}\t{1:3f}\n'.format(self.kurbelwinkel[i] + 360.0, self.p_avg[i]))

        Console.print('ok', "Mean values exported as '{0}'.".format(new_name), 1)

    def max(self):
        return np.max(self.p_max)


# <-------------------------------------------------------------------------------------------------- main function --->
def main(gasdruck_files, save_plot=False, export=False, y_limit=-1.0):
    gasdruck = []

    y_max = 0.0
    for file in gasdruck_files:
        Console.print('info', "'{0}':".format(file))
        gasdruck.append(Gasdruck(file))
        gasdruck[-1].process()
        max = gasdruck[-1].max()
        if y_max < max:
            y_max = max

    for g in gasdruck:
        Console.print('info', "'{0}':".format(g.file))
        if y_limit == -1.0:
            g.plot(float(int(y_max) + 1), save_plot)
        else:
            g.plot(y_limit, save_plot)
        if export:
            g.export()

    plt.show()

# <------------------------------------------------------------------------------------------------ main entrypoint --->
if __name__ == '__main__':
    Console.print('info', 'Started script: {0}'.format(os.path.basename(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, epilog='[CWD: {0}]\n'.format(os.getcwd()),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--save_plot', action='store_true',
                        help='Save plot as *.png')
    parser.add_argument('-e', '--export', action='store_true',
                        help='Export mean value curve as *.txt from (1° - 360°).')
    parser.add_argument('-y', '--y_limit', metavar='Y Axis Limit', type=float, nargs=1, default=[-1.0],
                        help='Set Y Axis limit for plot, -1.0 = set limit automatically - the same for all input files.')
    parser.add_argument('files', metavar='Gasdruck', nargs='+', type=str, action=CheckTXT,
                        help='*.txt file(s) with Gasdruck data for one RPM speed.')

    args = parser.parse_args()

    main(args.files, args.save_plot, args.export, args.y_limit[0])
    Console.print('ok', 'Done')

    # main(args.files, args.one_based, args.no_plot, not args.no_export)

