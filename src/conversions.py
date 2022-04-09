"""module contains all functions for unit conversions"""


def svedrup_to_kg_year(svedrup):
    """svedrup (1e6 m3/s) to kg/yr"""
    return svedrup * 1e6 * 1026 * 3.154e7


def moles_to_micromoles_kg(tracer_in_moles, mass_of_box):
    tracer_converted = tracer_in_moles * 10 ** -6 / mass_of_box
    return tracer_converted


def ratio_to_frac(ratio):
    """# convert isotope ratio to fractional abundance of isotope"""
    return ratio / (1 + ratio)


def frac_to_ratio(fraction):
    """convert fractional abundance of isotope to isotope ratio"""
    return fraction / (1 - fraction)
