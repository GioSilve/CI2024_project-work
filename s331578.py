# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(1.0001002004008015 * x[0])

def f2(x: np.ndarray) -> np.ndarray:
    return np.cbrt(11352873.606913827 * x[0]) + (1073947.8888777555 * x[0]) - ((-358239.5135 * x[2]) + (-264470.6121242485 * x[1]))

def f3(x: np.ndarray) -> np.ndarray:
    return (np.abs(8.721543086172344 * x[0]) - (3.547995991983968 * x[2] - (-5.942384769539078 * x[1]))) - (np.arctan(-77.23937875751504 * x[1]) - (-5.848196392785571 * x[1]))

def f4(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5142284569138278 * x[1]) / np.arctan(0.13949363961025193 * x[1])

def f5(x: np.ndarray) -> np.ndarray:
    return (2.6638330738300757e-10 * x[1]) * ((2.6638330738300757e-10 * x[1]) * (3.243909184359598e-09 * x[0]))

def f6(x: np.ndarray) -> np.ndarray:
    return (1.6943887775551103 * x[1]) + ((-6.0562124248497 * x[0]) * np.abs(0.11473376637639754))

def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(-1.6458917835671343 * x[1]) * np.abs(-6.632865731462926 * x[0])

def f8(x: np.ndarray) -> np.ndarray:
    return (((np.tan(1.672745490981964) - np.cbrt(2129.0328 * x[5])) - 1.6233)+ (((np.tan(1.6693386773547094) - np.cbrt(2141.8583 * x[5])) * (np.arctan(1.0613) * 1.6662)) * ((np.arctan(1.6167) * np.cbrt(3888.8189 * x[5])) - (np.sin(2159.0791583166333 * x[5]) - np.arctan(2133.2823 * x[5]))))) + (((4007.2269539078156 * x[5]) - np.abs(1862.8793 * x[4])) * np.arctan(np.arctan(0.30671472687944096)))
