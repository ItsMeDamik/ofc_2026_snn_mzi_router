import json
import os
import __main__
from time import sleep

from pydaq.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray

import numpy as np
import matplotlib.pyplot as plt

class LaserArray:
    def __init__(self, size, nets, netlist):
        self.size = size
        self.nets = nets
        self.netlist = netlist
    def set_voltages(self, voltages):
        voltages = np.clip(voltages, 0, 10)  # Ensure voltages are within the range [0, 10]
        for net, voltage in zip(self.nets, voltages):
            self.netlist[net].vout(voltage)

with netlist:
    inputDetectors = DetectorArray(
            size=4,
            nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0"],
            netlist=netlist,
            transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
        )
    outputDetectors = DetectorArray(
            size=4,
            nets=["PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
            netlist=netlist,
            transimpedance=220e3,  # 220k ohms
    )

    lasers = LaserArray(
        size=6,
        nets=["laser_1", "laser_2", "laser_3", "laser_4"],
        netlist=netlist
    )

    lasers.set_voltages([10, 7, 7, 7])  # Initialize all lasers to 0V
    sleep(2)

    input_readings = np.full((100, 4), np.nan)  # Initialize an array to store the input detector readings
    output_readings = np.full((100, 4), np.nan)  # Initialize an array to store the output detector readings

    for volt_index, voltage in enumerate(np.linspace(0, 4.2, 100)):  # Sweep from 0V to 4.2V
        netlist["1_3_i"].vout(voltage)  # Set the voltage for the first laser
        sleep(0.1)  # Wait for the system to stabilize
        output_readings[volt_index] = outputDetectors.read()  # Read the output detectors
        input_readings[volt_index] = inputDetectors.read()  # Read the input detectors

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(np.linspace(0, 4.2, 100), input_readings, label=["PD_2_0", "PD_3_0", "PD_4_0", "PD_5_0"])
axs[0].set_title("Input Detector Readings")
axs[0].set_xlabel("Voltage (V)")
axs[0].set_ylabel("Reading")
axs[1].plot(np.linspace(0, 4.2, 100), output_readings, label=["PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5"])
axs[1].set_title("Output Detector Readings")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Reading")
axs[0].legend()
axs[1].legend()
plt.show()