import cmath
import math
import os.path
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np

import CompareSignal
import ConvTest
import Shift_Fold_Signal
import comparesignal2
import comparesignals
import DerivativeSignal

from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from comparesignals import SignalSamplesAreEqual
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2
from signalcompare import SignalComapreAmplitude, SignalComaprePhaseShift
from CompareSignal import Compare_Signals
from task2_test import AddSignalSamplesAreEqual,SubSignalSamplesAreEqual,MultiplySignalByConst,ShiftSignalByConst,NormalizeSignal

amplitude = []
phases = []
num_samples = 0
amp = []
phas = []
result = []


def load_file(type_operation):
    global num_samples

    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, num_samples, signal_domain = read_signal(x_values, y_values, file_path)

    if type_operation == "test":
        global amp
        global phas
        amp = x_values
        phas = y_values


def open_new_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    button1 = tk.Button(new_window, text="Task 1", fg="black", command=first_task)
    button1.place(x=220, y=20)
    button2 = tk.Button(new_window, text="Task 2", fg="black", command=second_task)
    button2.place(x=220, y=70)
    button3 = tk.Button(new_window, text="Task 3", fg="black", command=quantize_page)
    button3.place(x=220, y=120)
    button4 = tk.Button(new_window, text="Task 4", fg="black", command=task_4_page)
    button4.place(x=220, y=170)
    button5 = tk.Button(new_window, text="Task 5", fg="black", command=task_5_page)
    button5.place(x=220, y=220)
    button6 = tk.Button(new_window, text="Task 6", fg="black", command=task_6_page)
    button6.place(x=220, y=270)
    button7 = tk.Button(new_window, text="Task 7", fg="black", command=task_7_page)
    button7.place(x=220, y=320)
    button8 = tk.Button(new_window, text="Task 8", fg="black", command=task_8_page)
    button8.place(x=220, y=370)
    button9 = tk.Button(new_window, text="Task 9", fg="black", command=task_9_page)
    button9.place(x=220, y=420)


# -------------------------------------------------------------

# Task 1
# done Review
def first_task():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Task 1")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    sin_button = tk.Button(new_window2, text="Show Sin Wave",
                           command=lambda: open_screen("This Is Sin Screen", "Generate Sin Wave", "Sin"))
    sin_button.place(x=200, y=200)

    cos_button = tk.Button(new_window2, text="Show Cos Wave",
                           command=lambda: open_screen("This Is Cos Screen", "Generate Cos Wave", "Cos"))
    cos_button.place(x=200, y=300)

    button1 = tk.Button(new_window2, text="Load File & Show Signals", command=load_signal)

    button1.place(x=200, y=100)


def open_screen(text1, text2, text3):
    new_window3 = tk.Toplevel(root)
    new_window3.geometry("500x500")
    new_window3.title("Task1")
    new_window3.configure(background="lightblue")

    label = tk.Label(new_window3, text=text1)

    label1 = tk.Label(new_window3, text="Amplutide")
    label2 = tk.Label(new_window3, text="Theta")
    label3 = tk.Label(new_window3, text="sampling frequency")
    label4 = tk.Label(new_window3, text="Analog frequency")

    label.pack()
    label1.place(x=50, y=50)
    label2.place(x=50, y=150)
    label3.place(x=50, y=250)
    label4.place(x=50, y=350)

    entry1 = tk.Entry(new_window3)
    entry1.place(x=250, y=50)
    entry2 = tk.Entry(new_window3)
    entry2.place(x=250, y=150)
    entry3 = tk.Entry(new_window3)
    entry3.place(x=250, y=250)
    entry4 = tk.Entry(new_window3)
    entry4.place(x=250, y=350)

    generate_button = tk.Button(new_window3, text=text2, command=lambda: wave(entry1, entry2, entry3, entry4, text3))

    generate_button.place(x=200, y=450)


def wave(entry1, entry2, entry3, entry4, type):
    if len(entry1.get()) == 0:
        amplitude = 0.0
    else:
        amplitude = float(entry1.get())
    if len(entry2.get()) == 0:
        theta = 0.0
    else:
        theta = float(entry2.get())
    if len(entry3.get()) == 0:
        sampling_frequency = 0.0
    else:
        sampling_frequency = float(entry3.get())
    if len(entry4.get()) == 0:
        analog_frequency = 0.0
    else:
        analog_frequency = float(entry4.get())

    new_window = tk.Toplevel(root)
    new_window.geometry("5000x5000")
    new_window.title("Task1")
    new_window.configure(background="lightblue")

    if sampling_frequency == 0:
        x = np.arange(0, 1, 0.01)
        # this line will return empty array because sampling_frequency
        # is 0 it can't create an array.
        x_discrete = np.arange(0, sampling_frequency, 1)

    elif sampling_frequency >= 2 * analog_frequency:
        x = np.arange(0, 1, 1 / sampling_frequency)
        x_discrete = np.arange(0, sampling_frequency, 1)

    else:
        label = tk.Label(new_window,
                         text="Error,sampling frequency Should be greater than or equal double of analog frequency",
                         font=("Helvetica", 30), anchor="center", fg="red")

        label.pack(pady=350)

    if type == "Sin":
        y = amplitude * np.sin(2 * np.pi * analog_frequency * x + theta)
        y_discrete = amplitude * np.sin((2 * np.pi * analog_frequency / sampling_frequency * x_discrete) + theta)

    else:
        y = amplitude * np.cos(2 * np.pi * analog_frequency * x + theta)
        y_discrete = amplitude * np.cos((2 * np.pi * analog_frequency / sampling_frequency * x_discrete) + theta)

    discrete_fun(new_window, "Time", x_discrete, y_discrete, "LEFT", "Amplitude")
    continuous_fun(new_window, "Time", x, y, "RIGHT", "Amplitude")

    if type == "Sin":
        SignalSamplesAreEqual(r"Task_1\SinOutput.txt", x, y)
    else:
        SignalSamplesAreEqual(r"Task_1\CosOutput.txt", x, y)


def load_signal():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x = []
    y = []
    x, y, samples, domain = read_signal(x, y, file_path)
    show_signals(x, y, "Time")


def read_signal(x_val1, y_val1, file_names):
    with open(file_names, 'r') as file:
        lines = file.read().splitlines()
        num_samples = int(lines[2])
        domain = int(lines[1])

        for line in lines[3:]:
            if line.__contains__(','):
                line = line.replace(',', " ")
            values = line.strip().split()
            if (values[0].__contains__('f') or values[1].__contains__('f')):
                values[0] = values[0].removesuffix('f')
                values[1] = values[1].removesuffix('f')
            x = float(values[0])
            y = float(values[1])
            x_val1.append(x)
            y_val1.append(y)

    return x_val1, y_val1, num_samples, domain


# ----------------------------------------------------------------
# Task 2
# done Review


def second_task():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Task 2")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    add_button = tk.Button(new_window2, text="Addtion", command=lambda: add_sub('+'))
    add_button.place(x=250, y=50)

    sub_button = tk.Button(new_window2, text="Subtraction", command=lambda: add_sub('-'))
    sub_button.place(x=250, y=100)

    mul_button = tk.Button(new_window2, text="Multiplication", command=multiply_page)
    mul_button.place(x=250, y=150)

    square_button = tk.Button(new_window2, text="Squaring", command=squaring)
    square_button.place(x=250, y=200)

    shift_button = tk.Button(new_window2, text="Shifting", command=shifting_page)
    shift_button.place(x=250, y=250)

    normalize_button = tk.Button(new_window2, text="Normalization", command=normalization_page)
    normalize_button.place(x=250, y=300)

    accumulate_button = tk.Button(new_window2, text="Accumulation", command=accumulation)
    accumulate_button.place(x=250, y=350)


def add_sub(op):
    # told him to enter the file from desktop
    file_paths = filedialog.askopenfilenames(filetypes=[("Text file", '*.txt')])
    max_length = 0
    file_names = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()
            num_samples = int(lines[2])
            max_length = max(max_length, num_samples)
            file_names.append(file_path)

    x_values = None
    if op == '+':
        y_values = np.zeros(max_length)
    elif op == '-':
        y_values = None

    for file_path in file_paths:
        if file_path:
            with open(file_path, 'r') as file:
                lines = file.read().splitlines()

            #periodicity = int(lines[0])
            signal_domain = int(lines[1])
            num_samples = int(lines[2])

            x_values_file = []
            y_values_file = []

            for line in lines[3:]:
                values = line.strip().split()
                x = float(values[0])
                y = float(values[1])
                x_values_file.append(x)
                y_values_file.append(y)

            if signal_domain == 0:
                domain = "Time"
            else:
                domain = "Frequency"

            if num_samples < max_length:
                pad_length = max_length - num_samples
                y_values_file += [0] * pad_length

            if op == '+':
                y_values += np.array(y_values_file)
            elif op == '-':
                if y_values is None:
                    y_values = np.array(y_values_file)
                else:
                    y_values -= np.array(y_values_file)

            if x_values is None:
                x_values = x_values_file
    if op == '-':
        y_values *= -1

    # Extract the file name
    f1 = os.path.splitext(os.path.basename(file_names[0]))[0]
    f2 = os.path.basename(file_names[1])
    file_name = f1 + op + f2
    path = os.path.join(
        r"Task_2\output signals",
        file_name)
    SignalSamplesAreEqual(path, x_values, y_values)
    if file_name == "Signal1+Signal2.txt":
        AddSignalSamplesAreEqual("Signal1.txt","Signal2.txt",x_values,y_values)
    elif file_name == "Signal1+Signal3.txt":
        AddSignalSamplesAreEqual("Signal1.txt", "Signal3.txt", x_values, y_values)
    elif file_name == "Signal1-Signal2.txt":
        SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", x_values, y_values)
    elif file_name == "Signal1-Signal3.txt":
        SubSignalSamplesAreEqual("Signal1.txt", "Signal3.txt", x_values, y_values)

    show_signals(x_values, y_values, domain)
    #print(path)


def multiply_page():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Multiplication")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    label1 = tk.Label(new_window2, text="Factor of Multiplication")
    label1.place(x=75, y=50)
    entry1 = tk.Entry(new_window2)
    entry1.place(x=275, y=50)

    generate_button = tk.Button(new_window2, text="Generate New Signal", command=lambda: multiplication(entry1))
    generate_button.place(x=200, y=200)


def multiplication(const):
    const = int(const.get())
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    y_values = [const * y for y in y_values]
    if const == 10:
        SignalSamplesAreEqual(
            r"Task_2\output signals\MultiplySignalByConstant-signal2 - by 10.txt", x_values,
            y_values)
        MultiplySignalByConst(10,x_values,y_values)
    elif const == 5:
        SignalSamplesAreEqual(
            r"Task_2\output signals\MultiplySignalByConstant-Signal1 - by 5.txt", x_values,
            y_values)
        MultiplySignalByConst(5,x_values,y_values)
    show_signals(x_values, y_values, 'Time')


def squaring():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    square_values = [(y_values[i] * y_values[i]) for i in range(len(y_values))]
    SignalSamplesAreEqual(
        r"Task_2\output signals\Output squaring signal 1.txt"
        , x_values,
        square_values)

    show_signals(x_values, square_values, 'Time')


def shifting_page():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Shifting Signal")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    label = tk.Label(new_window2, text="Please enter the amplitude of shifting")
    label.place(x=150, y=100)
    entry1 = tk.Entry(new_window2)
    entry1.place(x=185, y=170)

    shf_button = tk.Button(new_window2, text="Show Shifted Signal", command=lambda: shifting(entry1))
    shf_button.place(x=180, y=250)


def shifting(const):
    const = int(const.get())
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    x_values = [(y - const) for y in x_values]

    if const == -500:
        SignalSamplesAreEqual(
            r"Task_2/output signals/output shifting by minus 500.txt",
            x_values, y_values)
        ShiftSignalByConst(-500,x_values,y_values)
    elif const == 500:
        SignalSamplesAreEqual(
            r"Task_2/output signals/output shifting by add 500.txt",
            x_values, y_values)
        ShiftSignalByConst(500, x_values, y_values)
    show_signals(x_values, y_values, 'Time')


def normalization_page():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Normalization Signal")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    label = tk.Label(new_window2, text="Please enter the range that you want")
    label.place(x=150, y=100)

    label1 = tk.Label(new_window2, text="Start")
    label1.place(x=50, y=175)

    entry1 = tk.Entry(new_window2)
    entry1.place(x=100, y=175)

    label2 = tk.Label(new_window2, text="End")
    label2.place(x=250, y=175)

    entry2 = tk.Entry(new_window2)
    entry2.place(x=300, y=175)

    shf_button = tk.Button(new_window2, text="Show Normalizated Signal",
                           command=lambda: normalization(entry1))
    shf_button.place(x=180, y=250)


def normalization(const):
    const = int(const.get())
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    min_val = np.min(y_values)
    max_val = np.max(y_values)
    div = max_val - min_val
    # in y we apply this function : ((x-min) /(max-min))*(end-start)+start
    if const == 0:
        y_values = [(x - min_val) / div for x in y_values]
        SignalSamplesAreEqual(
            r"Task_2\output signals\normlize signal 2 -- output.txt",
            x_values, y_values)
        NormalizeSignal(0,1,x_values,y_values)
    elif const == -1:
        # we have a law (value - min )/ (max-min)*(new max - new min)+ new min
        y_values = [(2 * (x - min_val) / div) - 1 for x in y_values]

        SignalSamplesAreEqual(
            r"Task_2\output signals\normalize of signal 1 -- output.txt",
            x_values, y_values)
        NormalizeSignal(-1, 1, x_values, y_values)
    show_signals(x_values, y_values, 'Time')


def accumulation():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)
    # here we sum all values and take last value of i+1
    accumulated_values = [sum(y_values[:i + 1]) for i in range(len(y_values))]
    SignalSamplesAreEqual(r"Task_2\output signals\output accumulation for signal1.txt", x_values,
                          accumulated_values)
    show_signals(x_values, accumulated_values, 'Time')


# ------------------------------------------------------------------------
# Task 3
# done Review

# used for converting from infinite set to a finite set.
def quantize_page():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Quantization")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    selected = IntVar()
    label = tk.Label(new_window2, text="Please Select if you need to enter :")
    label.place(x=50, y=50)
    R1 = Radiobutton(new_window2, text="Number Of Levels", value=1, variable=selected)
    R1.place(x=300, y=50)
    R2 = Radiobutton(new_window2, text="Number Of Bits ", value=2, variable=selected)
    R2.place(x=300, y=80)
    label2 = tk.Label(new_window2, text="Please enter Value :")
    label2.place(x=50, y=150)

    entry1 = tk.Entry(new_window2)
    entry1.place(x=300, y=150)

    generate_button = tk.Button(new_window2, text="Generate Quantized Signal",
                                command=lambda: quantization(entry1, selected))
    generate_button.place(x=150, y=250)


def quantization(levels, operation):
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    levels = int(levels.get())
    selected = operation.get()
    if selected == 2:
        levels = 2 ** levels

    def min(y_values):
        min_val = 50000
        for i in y_values:
            if i < min_val:
                min_val = i
        return min_val

    def max(y_values):
        max_val = -50000
        for i in y_values:
            if i > max_val:
                max_val = i
        return max_val

    def Range(rounded_number):
        ranges = []
        ranges.append(min_val)
        l = min_val
        for i in range(levels - 1):
            l = l + delta
            ranges.append(round(l, rounded_number))
            ranges.append(round(l, rounded_number))
        ranges.append(max_val)
        return ranges

    def midPoints(roundNumber):
        mid_point = []
        c = 0
        for i in range(len(ranges)):
            i = c
            if c == len(ranges):
                break
            res = (ranges[i] + ranges[i + 1]) / 2
            mid_point.append(round(res, roundNumber))
            c = i + 2
        return mid_point

    def levels_fun():
        point_level = []
        for y in y_values:
            ran = None
            for i in range(len(ranges) // 2):
                if ranges[2 * i] <= y <= ranges[2 * i + 1]:
                    ran = i
                    break
            point_level.append(ran)
        return point_level

    def quantization():
        quantized = []
        for x in point_level:
            quantized.append(mid_point[x])
        return quantized

    def encoded_fun(number_bits):
        encoded = []
        for i in point_level:
            x = str(bin(i))[2:].zfill(number_bits)
            encoded.append(x)
        return encoded

    def interval_indices_fun():
        interval_indices = []
        for x in point_level:
            interval_indices.append(x + 1)
        return interval_indices

    def cal_eq():
        eq = []
        for i in range(samples):
            f = quantized[i] - y_values[i]
            eq.append(f)
        return eq

    min_val = min(y_values)
    max_val = max(y_values)
    delta = (max_val - min_val) / levels

    if (os.path.basename(file_path) == "Quan1_input.txt"):

        ranges = Range(2)
        mid_point = midPoints(2)
        point_level = levels_fun()
        quantized = quantization()
        encoded = encoded_fun(3)

        QuantizationTest1(
            r"Task_3\test 1\Quan1_Out.txt", encoded, quantized)
    elif (os.path.basename(file_path) == "Quan2_input.txt"):
        ranges = Range(3)
        mid_point = midPoints(3)
        point_level = levels_fun()
        interval_indices = interval_indices_fun()
        quantized = quantization()
        encoded = encoded_fun(2)
        eq = cal_eq()

        QuantizationTest2(
            r"Task_3\test 2\Quan2_Out.txt", interval_indices, encoded, quantized, eq)

    show_signals(x_values, y_values, 'Time')
# we input 3 bits for the first test , 2 bits for the second test

# ------------------------------------------------------------
# Task 4
# done Review

# help in converting to frequency domain , decomposition for complex signals
def task_4_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    label = tk.Label(new_window, text="Task 4", font=("Helvetica", 20), fg="black", bg="lightblue")
    label.pack(pady=50)
    button1 = tk.Button(new_window, text="Apply Fourier Transform", fg="black", command=FT_page)
    button1.place(x=200, y=150)
    button1 = tk.Button(new_window, text="Apply IDFT", fg="black", command=lambda: fourier_transform(None))
    button1.place(x=200, y=230)


def FT_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    label = tk.Label(new_window, text="Please enter sampling frequency : ", bg="lightblue")
    label.place(x=50, y=100)
    entry1 = tk.Entry(new_window)
    entry1.place(x=250, y=100)
    button1 = tk.Button(new_window, text="Apply FT and drawing relations ", fg="black",
                        command=lambda: fourier_transform(entry1))
    button1.place(x=170, y=200)
    button1 = tk.Button(new_window, text="Modify amplitude and phase", fg="black", command=Modifying_page)
    button1.place(x=170, y=300)
    button1 = tk.Button(new_window, text="Save new values", fg="black", command=Saving_in_polar_form)
    button1.place(x=170, y=350)
    button1 = tk.Button(new_window, text="Reconstruction using IDFT", fg="black",
                        command=lambda: fourier_transform(None))
    button1.place(x=170, y=400)


def fourier_transform(q):
    global amplitude
    global phases

    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, signal_domain = read_signal(x_values, y_values, file_path)

    if signal_domain == 0:
        sampling_frequency = float(q.get())
        x_k = dft(y_values, samples)
        real_parts = np.real(x_k)
        np.set_printoptions(formatter={'float': lambda x: "{:0.1f}".format(x)})
        imaginary_parts = np.imag(x_k)
        amplitude = np.sqrt(real_parts ** 2 + imaginary_parts ** 2)
        phases = np.arctan2(imaginary_parts, real_parts)
        np.set_printoptions(precision=3)
        time_sampling = (1 / sampling_frequency)
        omega = (2 * np.pi) / (samples * time_sampling)
        first_omega = round(omega, 2)
        frequencies = []
        for i in range(samples):
            x = (i + 1) * first_omega
            frequencies.append(x)
        print(amplitude)
        print("#####################################")
        print(phases)
        show_two_discrete(frequencies, amplitude, phases)

        load_file("test")
        X = SignalComaprePhaseShift(phas, phases)
        Y = SignalComapreAmplitude(amp, amplitude)
        print(str(X) + " First Test")
        print(str(Y) + " Second Test")

    else:
        output = IDFT(x_values, y_values, samples)
        show_signals(np.arange(samples), output, 'Frequency')
        print(output)


def dft(y_values, number_samples):
    output = np.zeros(number_samples, dtype=np.complex128)
    for k in range(number_samples):
        for n in range(number_samples):
            output[k] += y_values[n] * np.exp(-2j * np.pi * k * n / number_samples)
    return output


def IDFT(amplitude_polar, phases_polar, number_samples):
    results = np.zeros(number_samples, dtype=np.complex128)
    for k in range(number_samples):
        results[k] = cmath.rect(amplitude_polar[k], phases_polar[k])
    output = []
    res = 0
    np.set_printoptions(formatter={'float': lambda x: "{:0.1f}".format(x)})
    np.set_printoptions(precision=3)
    for n in range(number_samples):
        for k in range(number_samples):
            res += results[k] * np.exp(2j * np.pi * k * n / number_samples)
        real_parts = np.real(res)
        real_parts = round(real_parts, 3)
        output.append((real_parts / number_samples))
        res = 0
    return output


def Modifying_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    label0 = tk.Label(new_window, text="Please enter Row Number : ", bg="lightblue")
    label0.place(x=50, y=50)
    entry0 = tk.Entry(new_window)
    entry0.place(x=250, y=50)
    label = tk.Label(new_window, text="Please enter new amplitude : ", bg="lightblue")
    label.place(x=50, y=150)
    entry1 = tk.Entry(new_window)
    entry1.place(x=250, y=150)
    label1 = tk.Label(new_window, text="Please enter new phase shift: ", bg="lightblue")
    label1.place(x=50, y=250)
    entry2 = tk.Entry(new_window)
    entry2.place(x=250, y=250)
    button1 = tk.Button(new_window, text="Calulate", fg="black", command=lambda: edit(entry0, entry1, entry2))
    button1.place(x=230, y=350)


def edit(e1, e2, e3):
    global amplitude
    global phases

    row_number = int(e1.get())
    new_amplitude = float(e2.get())
    new_phase = float(e3.get())
    amplitude[row_number - 1] = new_amplitude
    phases[row_number - 1] = new_phase
    print(amplitude)
    print("#########################")
    print(phases)


def Saving_in_polar_form():
    global num_samples
    with open('Saved file .txt', 'w') as f:
        f.write('0')
        f.write('\n')
        f.write('1')
        f.write('\n')
        f.write(str(num_samples))
        f.write('\n')
        for i in range(num_samples):
            f.write(str(amplitude[i]) + 'f' + ',' + str(phases[i]) + 'f')
            f.write('\n')
        f.close()
    print("File Saved Successfully")


# ----------------------------------------------------
# Task 5
# done Review

# used in compression for signal
def task_5_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    button1 = tk.Button(new_window, text="Computing DCT ", fg="black", command=DCT)
    button1.place(x=200, y=150)
    label0 = tk.Label(new_window, text="Please enter Number of Coefficients : ", bg="lightblue")
    label0.place(x=50, y=100)
    entry = tk.Entry(new_window)
    entry.place(x=300, y=100)
    button2 = tk.Button(new_window, text="Save M coefficients", fg="black", command=lambda: save_coeff(entry))
    button2.place(x=200, y=250)
    button3 = tk.Button(new_window, text="Remove DC component", fg="black",
                        command=DC_component)
    button3.place(x=200, y=360)


def DCT():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    global result
    result = []
    first = math.sqrt(2 / samples)
    for j in range(samples):
        r = 0
        for i in range(samples):
            com = math.pi / (4 * samples)
            l1 = (2 * i) - 1
            l2 = (2 * j) - 1
            r += y_values[i] * math.cos(com * l1 * l2)
        result.append(first * r)

    # if doctor want to write it like folder we will
    # remove for loop only
    samples = np.zeros(6, dtype=int)
    for i in range(6):
        samples[i] = i + 1

    show_signals(samples, result, "Frequency")
    comparesignal2.SignalSamplesAreEqual(
        r'Task_5\DCT\DCT_output.txt',
        result)


def save_coeff(entry):
    m = int(entry.get())
    with open('DCT_coeff .txt', 'w') as f:
        f.write('0')
        f.write('\n')
        f.write('1')
        f.write('\n')
        f.write(str(num_samples))
        f.write('\n')
        for i in range(m):
            f.write(str(0) + ' ' + str(result[i]))
            f.write('\n')
        f.close()
    print("File Saved Successfully")


def DC_component():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    summation = 0
    for j in range(samples):
        summation += y_values[j]
    avarage = summation / samples
    for j in range(samples):
        y_values[j] = y_values[j] - avarage
    show_signals(x_values, y_values, "Frequency")
    comparesignal2.SignalSamplesAreEqual(
        r'Task_5\Remove DC component\DC_component_output.txt'
        , y_values)


# --------------------------------------------------

# Task 6
# done Review
def task_6_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    button1 = tk.Button(new_window, text="Smoothing", fg="black", command=smoothing_page)
    button1.place(x=200, y=100)
    button2 = tk.Button(new_window, text="Sharping", fg="black", command=sharp)
    button2.place(x=200, y=150)
    label0 = tk.Label(new_window, text="Please enter Number of shifting : ", bg="lightblue")
    label0.place(x=50, y=200)
    entry = tk.Entry(new_window)
    entry.place(x=300, y=200)
    button3 = tk.Button(new_window, text="Shifting", fg="black", command=lambda: shifting(entry))
    button3.place(x=200, y=250)
    button4 = tk.Button(new_window, text="Folding", fg="black", command=fold_signal)
    button4.place(x=200, y=300)
    button5 = tk.Button(new_window, text="Shifting&folding", fg="black",
                        command=lambda: folding_shifting(entry))
    button5.place(x=200, y=350)
    button6 = tk.Button(new_window, text="Remove DC ", fg="black", command=remove_dc)
    button6.place(x=200, y=400)


def smoothing_page():
    new_window2 = tk.Toplevel(root)
    new_window2.title("Shifting Signal")
    new_window2.geometry("500x500")
    new_window2.configure(background="lightblue")

    label = tk.Label(new_window2, text="Please enter the Window Size")
    label.place(x=150, y=100)
    entry1 = tk.Entry(new_window2)
    entry1.place(x=170, y=170)

    shf_button = tk.Button(new_window2, text="Apply Smoothing", command=lambda: smoothing(entry1))
    shf_button.place(x=180, y=250)


def smoothing(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    # use average to reduce the noise.
    window_size = int(entry.get())
    # print(window_size)
    samples = samples - window_size + 1
    x_axis = np.arange(samples)
    y = []
    if window_size == 3:
        for i in range(samples):
            y.append((y_values[i] + y_values[i + 1] + y_values[i + 2]) / 3)
        comparesignals.SignalSamplesAreEqual(
            r"Task_6\Moving Average\MovAvgTest1.txt",
            x_axis, y)
    elif window_size == 5:
        for i in range(samples):
            y.append((y_values[i] + y_values[i + 1] + y_values[i + 2] + y_values[i + 3] + y_values[i + 4]) / 5)
        comparesignals.SignalSamplesAreEqual(
            r"Task_6\Moving Average\MovAvgTest2.txt",
            x_axis, y)
    show_two_continous(x_values, x_axis, y_values, y)


def sharp():
    expectedOutput_first, expectedOutput_second = DerivativeSignal.DerivativeSignal()
    show_two_continous(np.arange(0, len(expectedOutput_first)), np.arange(0, len(expectedOutput_second)),
                       expectedOutput_first, expectedOutput_second)


def fold_signal():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    output = folding(y_values, samples)
    Shift_Fold_Signal.Shift_Fold_Signal(
        r'Task_6\Shifting and Folding\Output_fold.txt',
        x_values, output)
    show_two_continous(x_values, x_values, y_values, output)


def folding(y, num_samples):
    ind = 0
    new_y = num_samples * [None]
    for i in reversed(range(len(y))):
        new_y[ind] = y[i]
        ind += 1
    # print(new_y)

    return new_y


def folding_shifting(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    const = int(entry.get())
    x_values_new = [(y + const) for y in x_values]
    output = folding(y_values, samples)

    print("x", x_values_new)
    print("o", output)

    if const == 500:
        Shift_Fold_Signal.Shift_Fold_Signal(
            r'Task_6\Shifting and Folding\Output_ShifFoldedby500.txt',
            x_values_new, output)
    elif const == -500:
        Shift_Fold_Signal.Shift_Fold_Signal(
            r'Task_6\Shifting and Folding\Output_ShiftFoldedby-500.txt',
            x_values_new, output)

    show_two_continous(x_values, x_values_new, y_values, output)


def remove_dc():
    file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    x_values = []
    y_values = []

    x_values, y_values, samples, domain = read_signal(x_values, y_values, file_path)

    output = dft(y_values, samples)
    real_parts = np.real(output)
    real_parts[0] = 0
    imaginary_parts = np.imag(output)
    imaginary_parts[0] = 0
    amplitude_DC = np.sqrt(real_parts ** 2 + imaginary_parts ** 2)
    phases_DC = np.arctan2(imaginary_parts, real_parts)
    # print(amplitude_DC)
    output2 = IDFT(amplitude_DC, phases_DC, samples)
    comparesignal2.SignalSamplesAreEqual(
        r'Task_6\Remove DC component\DC_component_output.txt'
        , output2)
    show_two_continous(x_values, x_values, y_values, output2)


# ---------------------------------------------------------

# task 7
# done Review

def task_7_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")
    button1 = tk.Button(new_window, text="Convolution", fg="black", command=seven_task)
    button1.place(x=210, y=230)


def seven_task():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text file", '*.txt')])

    x_val1 = []
    x_val2 = []
    y_val1 = []
    y_val2 = []

    file_names = []

    y = []
    x = []

    for file_path in file_paths:
        file_names.append(file_path)

    x_val1, y_val1, num_samples1, domain = read_signal(x_val1, y_val1, file_names[0])
    x_val2, y_val2, num_samples2, domain = read_signal(x_val2, y_val2, file_names[1])

    min_1 = int(min(x_val1))
    min_2 = int(min(x_val2))
    max_1 = int(max(x_val1))
    max_2 = int(max(x_val2))

    start = min_1 + min_2
    end = max_1 + max_2

    my_dictionary_1 = {x_val1[i]: y_val1[i] for i in range(num_samples1)}
    my_dictionary_2 = {x_val2[i]: y_val2[i] for i in range(num_samples2)}

    for n in range(start, end + 1, 1):
        summation = 0
        for k in range(min_1, end + 1, 1):
            if my_dictionary_1.__contains__(k) and my_dictionary_2.__contains__(n - k):
                summation += my_dictionary_1[k] * my_dictionary_2[n - k]
        y.append(summation)
        x.append(n)
    ConvTest.ConvTest(x, y)
    show_signals(x, y, "Frequency")


# ----------------------------------------------------------

# Task 8
# done Review
def task_8_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")

    button = tk.Button(new_window, text="Correlation", fg="black", command=task_8)
    button.place(x=210, y=230)


def task_8():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text file", '*.txt')])

    x_val1 = []
    x_val2 = []
    y_val1 = []
    y_val2 = []

    y = []
    x = []

    file_names = []

    sum_of_x_1_square = 0
    sum_of_x_2_square = 0

    for file_path in file_paths:
        file_names.append(file_path)

    x_val1, y_val1, num_samples1, domain = read_signal(x_val1, y_val1, file_names[0])
    x_val2, y_val2, num_samples2, domain = read_signal(x_val2, y_val2, file_names[1])

    for i in y_val1:
        sum_of_x_1_square += i * i

    for i in y_val2:
        sum_of_x_2_square += i * i

    mult = sum_of_x_1_square * sum_of_x_2_square
    domainator = (1 / num_samples1) * (math.pow(mult, 0.5))

    def shifting(list):
        shift = []
        for i in range(num_samples1 - 1):
            shift.append(list[i + 1])
        shift.append(list[0])
        return shift

    shifted_list = []
    shifted_list.append(y_val2)
    input = y_val2
    for i in range(num_samples1):
        shift = shifting(input)
        shifted_list.append(shift)
        input = shift

    for shift in range(num_samples1):
        corr_value = 0
        for i in range(num_samples1):
            corr_value += y_val1[i] * shifted_list[shift][i]

        value = (1 / num_samples1) * corr_value
        y.append(value / domainator)

    Compare_Signals(
        r"Task_8\Point1 Correlation\CorrOutput.txt", x,
        y)
    show_signals(range(0, num_samples1), y, "Frequency")


# -----------------------

# Task 9
# done Review
def task_9_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")

    button = tk.Button(new_window, text="Fast Correlation", fg="black", command=fast_correlation_page)
    button.place(x=220, y=120)

    # put here the command of fast convolution
    button2 = tk.Button(new_window, text="Fast Convolution", fg="black", command=fast_convolution_page)
    button2.place(x=220, y=240)


def fast_correlation_page():
    new_window = tk.Toplevel(root)
    new_window.title("Tasks")
    new_window.geometry("500x500")
    new_window.configure(background="lightblue")

    button = tk.Button(new_window, text="Auto Correlation", fg="black", command=auto_correlation)
    button.place(x=220, y=120)

    # put here the command of fast convolution
    button2 = tk.Button(new_window, text="Cross Correlation", fg="black", command=cross_correlation)
    button2.place(x=220, y=240)


def auto_correlation():
    # file_path = filedialog.askopenfilename(filetypes=[("Text file", '*.txt')])
    #
    # x_val1 = []
    # y_val1 = []
    #
    # x,x_corr,samples,domain=read_signal(x_val1,y_val1,file_path)
    #
    # print(x)
    # print(x_corr)
    # print(samples)

    x_corr = [1, 0, 0, 1]
    samples = len(x_corr)

    final_output = correlation(x_corr, [], samples, "auto")


def correlation(y_value1, y_value2, samples, type):
    xk = dft(y_value1, samples)
    # print(xk)

    approximate_xk = take_complex(xk)
    # print(approximate_xk)

    conjugate_xk = eliminate_zeroj(approximate_xk)
    # print(conjugate_xk)

    if type == "auto":
        input_inverse = np.array(approximate_xk) * np.array(conjugate_xk)
        # print(input_inverse)
    else:
        xk2 = dft(y_value2, samples)
        # print(xk2)

        approximate_xk2 = take_complex(xk2)
        # print(approximate_xk2)

        input_inverse = np.array(conjugate_xk) * np.array(approximate_xk2)
        # print(input_inverse)

    fd_input = take_complex(input_inverse)
    # print(fd_input)

    real_parts = np.real(fd_input)
    np.set_printoptions(formatter={'float': lambda x: "{:0.1f}".format(x)})
    imaginary_parts = np.imag(fd_input)
    amplitude = np.sqrt(real_parts ** 2 + imaginary_parts ** 2)
    phases = np.arctan2(imaginary_parts, real_parts)
    np.set_printoptions(precision=3)

    output = IDFT(amplitude, phases, samples)
    # print(output)

    round_output = [int(round(value)) for value in output]
    # print(round_output)

    final_output = (1 / samples) * np.array(round_output)

    print(final_output)
    return final_output


def cross_correlation():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text file", '*.txt')])

    x_val1 = []
    x_val2 = []

    y_val1 = []
    y_val2 = []

    file_names = []

    for file_path in file_paths:
        file_names.append(file_path)

    x_val1, y_val1, num_samples1, domain = read_signal(x_val1, y_val1, file_names[0])
    x_val2, y_val2, num_samples2, domain = read_signal(x_val2, y_val2, file_names[1])

    output = correlation(y_val1, y_val2, num_samples2, "")
    CompareSignal.Compare_Signals(
        r"Task_9\Fast Correlation\Corr_Output.txt",
        x_val1, output)
    show_signals(x_val1, output, "Frequency")


def eliminate_zeroj(approximate_xk):
    conjugate_xk = []
    for element in approximate_xk:
        if element.imag == 0:
            conjugate_xk.append((element))

        else:

            x = np.conjugate(element)
            conjugate_xk.append(x)

    return conjugate_xk


def take_complex(input_inverse):
    fd_input = []
    for value in input_inverse:
        im = round(value.imag, 2)
        if im == 0:
            x = round(value.real, 2)
        else:
            x = round(value.real, 2) + round(value.imag, 2) * 1j
        fd_input.append(x)

    return fd_input


def fast_convolution_page():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text file", '*.txt')])

    x_val1 = []
    x_val2 = []
    y_val1 = []
    y_val2 = []

    file_names = []

    for file_path in file_paths:
        file_names.append(file_path)

    x_val1, y_val1, num_samples1, domain = read_signal(x_val1, y_val1, file_names[0])
    x_val2, y_val2, num_samples2, domain = read_signal(x_val2, y_val2, file_names[1])

    min_1 = int(min(x_val1))
    min_2 = int(min(x_val2))
    max_1 = int(max(x_val1))
    max_2 = int(max(x_val2))

    start = min_1 + min_2
    end = max_1 + max_2

    n = num_samples1 + num_samples2 - 1
    s1 = np.pad(y_val1, (0, n - len(y_val1)))
    s2 = np.pad(y_val2, (0, n - len(y_val2)))

    signal1 = dft(s1, len(s1))
    signal2 = dft(s2, len(s2))

    x = np.arange(start, end + 1, 1)

    out_of_mult = signal1 * signal2

    # print("out_of_mult", out_of_mult)
    real_parts = np.real(out_of_mult)
    np.set_printoptions(formatter={'float': lambda x: "{:0.1f}".format(x)})
    imaginary_parts = np.imag(out_of_mult)
    # print("real", real_parts)
    amplitude = np.sqrt(real_parts ** 2 + imaginary_parts ** 2)
    # print("amplitude", amplitude)
    phases = np.arctan2(imaginary_parts, real_parts)
    np.set_printoptions(precision=3)
    # print("phases", phases)

    y = IDFT(amplitude, phases, len(out_of_mult))
    # print("y", y)
    ConvTest.ConvTest(x, y)
    show_signals(x, y, "Frequency")


# --------------------------------
# Show Signals

def show_signals(x, y, domain):
    signal_page = tk.Toplevel(root)
    signal_page.title("Signals")
    signal_page.geometry("5000x5000")
    signal_page.configure(background="lightblue")

    dis_con(signal_page, domain, x, y)


def show_two_discrete(frequencies, amplitude, phase_shift):
    signal_page = tk.Toplevel(root)
    signal_page.title("Signals")
    signal_page.geometry("5000x5000")
    signal_page.configure(background="lightblue")

    discrete_fun(signal_page, "Frequency", frequencies, amplitude, "LEFT", "Amplitude")
    discrete_fun(signal_page, "Frequency", frequencies, phase_shift, "RIGHT", "Phase_Shift")


def show_two_continous(x1, x_axis, y1, y2):
    signal_page = tk.Toplevel(root)
    signal_page.title("Signals")
    signal_page.geometry("5000x5000")
    signal_page.configure(background="lightblue")

    continuous_fun(signal_page, "Frequency", x1, y1, "LEFT", "SIGNAL BEFORE")
    continuous_fun(signal_page, "Frequency", x_axis, y2, "RIGHT", "SIGNAL AFTER")


def dis_con(page_name, domain, x, y):
    discrete_fun(page_name, domain, x, y, "LEFT", "Amplitude")
    continuous_fun(page_name, domain, x, y, "RIGHT", "Amplitude")


def discrete_fun(signal_page, domain, x, y, place, label):
    # figsize is the width and the height of the plot
    # dpi it's determine the resolution of figure size of dots in page
    # we use plt.figure to create a figure
    # we add subplot in the figure, and we told him that it's the only sub_
    # plot through 111
    fig_discrete = plt.Figure(figsize=(5, 4), dpi=100)
    discrete_signal = fig_discrete.add_subplot(111)

    # we make a canvas to show our figure on it from its parent widget
    # take canvas and add it in the parent widget on the left side
    # then draw its x-axis and y-axis and expand for fill the parent widget

    canvas_discrete = FigureCanvasTkAgg(fig_discrete, master=signal_page)
    if place == "LEFT":
        canvas_discrete.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    else:
        canvas_discrete.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    discrete_signal.clear()

    # linefmt --> blue solid line in plot
    # markerfmt --> blue circles in the plot
    # basefmt --> solid red line that show in discrete figure

    discrete_signal.stem(x, y, linefmt='b-', markerfmt='ro', basefmt='r-')
    discrete_signal.set_title('Discrete Signal')
    discrete_signal.set_xlabel(domain)
    discrete_signal.set_ylabel(label)
    canvas_discrete.draw()


def continuous_fun(signal_page, domain, x, y, label, name):

    fig_continuous = plt.Figure(figsize=(5, 4), dpi=100)
    continous_signal = fig_continuous.add_subplot(111)
    canvas_continuous = FigureCanvasTkAgg(fig_continuous, master=signal_page)
    if label == "RIGHT":
        canvas_continuous.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
    else:
        canvas_continuous.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    continous_signal.clear()
    # control the smoothness of signal
    # window_size =3
    # smoothed_signal = np.convolve(y, np.ones(window_size) / window_size, mode='same')
    continous_signal.plot(x, y, 'b-')
    continous_signal.set_title('Continuous Signal')
    continous_signal.set_xlabel(domain)
    continous_signal.set_ylabel(name)
    # used to show points in the continuous signal
    # continous_signal.scatter(x, y, color='red', label='Sampled Points')
    canvas_continuous.draw()




root = tk.Tk()
root.title("DSP tasks")
root.geometry("500x500")
root.configure(background="lightblue")

label = tk.Label(root, text="Welcome to DSP Tasks", font=("Helvetica", 30), fg="black", bg="lightblue")
label.pack(pady=150)

button = tk.Button(root, text="Let's Start", command=open_new_page, fg="black")
button.pack()

root.mainloop()
