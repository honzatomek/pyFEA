#!/usr/bin/env python3
"""Simple Unit converter utility written in python

author:  Jan Tomek
date:    14.06.2023
version: v1.0.0

Takes three arguments: FROM, TO and VALUE, where FROM and TO should be strings
representing the units to convert between and TO a float value.


+---------------------------------------------------------------------------------------+
!                                                                                       !
!                                 !!! IMPORTANT !!!                                     !
!                                                                                       !
!    It is upon the user to check the units, the script will happily convert between    !
!    energy and time or volume and angle                                                !
!                                                                                       !
!    It is important to separate the ditinct units by spaces (see Examples) !!!         !
!                                                                                       !
+---------------------------------------------------------------------------------------+


The script first converts the value to SI units and from there to the desired TO units.


The conversion of temperatures works only for relative values, because what matters
is the value per degree change, not the temperature itself:

                      1 J / K = 1 J / °C = 9 / 5 J / °F


If the sole units are temperature, e.g. 'C', 'K', or 'F' it does exact conversion.

If the units desired are not listed, it is necessary to express them in basic or
derived ones - e.g. N as Newton is 'kg * m / s ** 2', Pa is 'kg / ( m * s ** 2 )' etc...

Examples:
>>> $ ./unit_converter.py -f 'kg / m3' -t 't / mm3' 7850.0
>>> 7.850000E+03 kg / m3  -> 7.850000E-09 t / mm3
>>>
>>> $ ./unit_converter.py -t 't / mm3' 7850.0
>>> 7.850000E+03 kg / m3 -> 7.850000E-09 t / mm3
>>>
>>> $ ./unit_converter.py -f 'kg * m * s ** 2' -t 't * mm * s ** 2' 1000.
>>> 1.000000E+03 kg * m * s ** 2 -> 1.000000E+03 t * mm * s ** 2
>>>
>>> $ ./unit_converter.py -f 'G ** 2 / Hz' -t '( mm ** 2 / s ** 4 ) / Hz' 10.
>>> 1.000000E+01 G ** 2 / Hz -> 9.617136E+08 ( mm ** 2 / s ** 4 ) / Hz
>>>
>>> $ ./unit_converter.py -f 'G ** 2 / Hz' -t '( mm / s ** 2 ) ** 2 / Hz' 10.
>>> 1.000000E+01 G ** 2 / Hz -> 9.617136E+08 ( mm / s ** 2 ) ** 2 / Hz
>>>
>>> $ ./unit_converter.py -f 'm/s' -t 'km/h' 20.
>>> 2.000000E+01 m/s -> 7.200000E+01 km/h
>>>

"""

import os
import sys
import math
import argparse


class UnitConverter:
    mass = {"kg": 1., "mg": 1.E-6, "g": 1.E-3, "t": 1.E+3, "oz": 0.028349523125, "lb": 0.45359237}
    length = {"m": 1., "cm": 1.E-2, "dm": 1.E-1, "mm": 1.E-3, "km": 1.E+3, "in": 0.0254, "ft": 0.3048, "yd": 0.9144}
    angle = {"rad": 1., "mrad": 1.E-3, "\"": 4.8481E-6, "'": 0.000290888, "deg": 180. / math.pi}
    temperature = {"K": 1., "°C": 1., "C": 1., "°F": 9. / 5., "F": 9. / 5.}
    time = {"s": 1., "ms": 1.E-3, "min": 60., "h": 3600., "days": 86400., "weeks": 604800., "months": 60.*60.*24.*31., "years": 60.*60.*24.*31.*365.}

    area = {"m2": 1., "mm2": 1.E-6, "cm2": 1.E-4, "dm2": 1.E-2, "km2": 1.E+6, "sqin": 0.00064516, "in2": 0.00064516, "sqft": 0.092903, "ft2": 0.092903, "sqyd": 0.836127, "yd2": 0.836127}
    volume = {"m3": 1., "mm3": 1.E-9, "cm3": 1.E-6, "dm3": 1.E-3, "km3": 1.E+9, "l": 1.E-3, "ml": 1.E-6, "in3": 1.6387E-5, "ft3": 0.0283168, "yd3": 0.764554858, "floz": 2.8413E-5, "brfloz": 2.8413E-5, "usfloz": 2.9574E-5, "# brfloz: British fluid ounce, usfloz: US fluid ounce": None}
    velocity = {"m/s": 1., "mm/s": 1.E-3, "km/h": 1. / 3.6}
    acceleration = {"m/s2": 1., "mm/s2": 1.E-3, "G": 9.8067}

    force = {"N": 1., "mN": 1.E-3, "kN": 1.E+3, "MN": 1.E+6, "GN": 1.E+9, "lbf": 4.44822}
    pressure = {"Pa": 1., "mPa": 1.E-3, "kPa": 1.E+3, "MPa": 1.E+6, "GPa": 1.E+9, "atm": 101325., "psi": 6894.76, "bar": 1.0E+5, "tor": 133.322}

    energy = {"J": 1., "mJ": 1.E-3, "kJ": 1.E+3, "MJ": 1.E+6, "GJ": 1.E+9, "cal": 4.184, "kcal": 4184., "Ws": 1., "Wh": 3600., "kWh": 3.6E+6, "BTU": 1055.06, "ft.lbf": 1.3558179483314004, "# BTU: British Thermal Unit": None}
    power = {"W": 1., "mW": 1.E-3, "kW": 1.E+3, "MW": 1.E+6, "hp": 735.49875, "hpuk": 745.69987158, "BTU/h": 0.2930710702, "BTU/min": 17.58426421, "BTU/s": 1055.0558526, "ft.lbf/h": 0.0003766161, "ft.lbf/min": 0.0225969658, "ft.lbf/s": 1.3558179483, "# hp: horse power (metric), hpuk: horse power (UK), BTU: British Thermal Unit, ft.lbf: foot-pound-force": None}

    frequency = {"Hz": 1., "mHz": 1.E-3, "kHz": 1.E+3, "MHz": 1.E+6, "GHz": 1.E+9}


    unit_types = {"mass": mass,
                  "length": length,
                  "angle": angle,
                  "temperature": temperature,
                  "time": time,
                  "area": area,
                  "volume": volume,
                  "velocity": velocity,
                  "acceleration": acceleration,
                  "force": force,
                  "pressure": pressure,
                  "energy": energy,
                  "power": power,
                  "frequency": frequency,
                  }


    units = {}
    units.update(mass)
    units.update(length)
    units.update(angle)
    units.update(temperature)
    units.update(time)

    units.update(area)
    units.update(volume)
    units.update(velocity)
    units.update(acceleration)

    units.update(force)
    units.update(pressure)
    units.update(energy)
    units.update(power)

    units.update(frequency)


    @classmethod
    def convert(cls, value: float, from_units: str = None, to_units: str = None) -> float:
        if from_units in ["C", "°C", "K", "F", "°F"] and to_units in  ["C", "°C", "K", "F", "°F"]:
            if from_units in ["C", "°C"]:
                value = cls.Celsius2Kelvin(value)
            elif from_units in ["F", "°F"]:
                value = cls.Fahrenheit2Kelvin(value)

            if to_units in ["C", "°C"]:
                value = cls.Kelvin2Celsius(value)
            elif to_units in ["F", "°F"]:
                value = cls.Kelvin2Fahrenheit(value)

            return value


        if from_units is None:
            coeff_from = 1.
        else:
            coeff_from = [str(cls.units[u]) if u in cls.units.keys() else u for u in from_units.split(" ")]
            coeff_from = float(eval(" ".join(coeff_from)))

        if to_units is None:
            coeff_to = 1.
        else:
            coeff_to   = [str(cls.units[u]) if u in cls.units.keys() else u for u in   to_units.split(" ")]
            coeff_to   = float(eval(" ".join(coeff_to  )))

        return value * coeff_from / coeff_to

    @classmethod
    def get_SI_units(cls, units: str) -> str:
        units_from = units.split(" ")
        units_SI = {}
        for unit in [u for u in units_from if u in cls.units.keys()]:
            for ut, uti in cls.unit_types.items():
                if unit in uti.keys():
                    unit_SI = list(uti.keys())[0]
                    units_SI.setdefault(unit, unit_SI)
                    break
        units_to = [units_SI[u] if u in units_SI.keys() else u for u in units_from]
        return " ".join(units_to)

    @classmethod
    def Celsius2Kelvin(cls, value: float) -> float:
        return value + 273.15

    @classmethod
    def Kelvin2Celsius(cls, value: float) -> float:
        return value - 273.15

    @classmethod
    def Fahrenheit2Kelvin(cls, value: float) -> float:
        return (value - 32.) * 5. / 9. + 273.14

    @classmethod
    def Kelvin2Fahrenheit(cls, value: float) -> float:
        return (value - 273.15) * 9. / 5. + 32.

    @classmethod
    def Celsius2Fahrenheit(cls, value: float) -> float:
        return cls.Kelvin2Fahrenheit(cls.Celsius2Kelvin(value))

    @classmethod
    def Fahrenheit2Celsius(cls, value: float) -> float:
        return cls.Kelvin2Celsius(cls.Fahrenheit2Kelvin(value))



def test_unit_converter():
    uc = UnitConverter()

    val = 1000.
    units_from = "mm"
    units_to   = "m"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 1000.
    units_from = "N * mm"
    units_to   = "N * m"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 1000.
    units_from = "kg * m * s ** 2"
    units_to   = "t * mm * s ** 2"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 7850.
    units_from = "kg / m3"
    units_to   = "t / mm3"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 0.0254
    units_from = "m"
    units_to   = "in"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 9.81
    units_from = "m / s ** 2"
    units_to   = "mm / s ** 2"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 9.81 ** 2
    units_from = "( m2 / s ** 2 ) ** 2"
    units_to   = "( mm2 / s ** 2 ) ** 2"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 9.81 ** 2
    units_from = "( m2 / s ** 2 ) ** 2"
    units_to   = "( mm2 / s ** 2 ) ** 2"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")

    val = 1.
    units_from = "m/s"
    units_to   = "km/h"
    cval = UnitConverter.convert(val, units_from, units_to)
    print(f"{val:E} {units_from:<30s} -> {cval:E} {units_to:s}")




if __name__ == "__main__":
    all_formats = "\n".join(f"{unit_type:<12s}: {', '.join(list(units.keys())):s}" for unit_type, units in UnitConverter.unit_types.items())
    parser = argparse.ArgumentParser(description=__doc__ + "\nAll Available Units:\n" + all_formats,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-f", "--from", dest="from_units", type=str, default=None,
                        help="Units to convert from, e.g. 'kg * m / s ** 2'. If not specified, presumes SI units.")

    parser.add_argument("-t", "--to", dest="to_units", type=str, default=None,
                        help="Units to convert to, e.g. 't * mm / s ** 2'. Returns SI units if not specified.")

    parser.add_argument("value", type=float, help="Value to convert.")

    args = parser.parse_args()

    # print(args)

    converted_value = UnitConverter.convert(args.value, args.from_units, args.to_units)
    res = f"{args.value:E}"
    if args.from_units is not None:
        res += f" {args.from_units:s}"
    else:
        res += f" {UnitConverter.get_SI_units(args.to_units):s}"
    res += f" -> {converted_value:E}"
    if args.to_units is not None:
        res += f" {args.to_units:s}"
    else:
        res += f" {UnitConverter.get_SI_units(args.from_units):s}"

    print(res)

