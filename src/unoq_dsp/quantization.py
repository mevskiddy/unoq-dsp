"""TFLite double-rounding fixed-point preparation, outside the inference loop."""
import math
import numpy as np
from .errors import UnsupportedModel


def multiplier(value):
    if not math.isfinite(value) or value < 0:
        raise UnsupportedModel("Invalid quantization multiplier")
    significand, exponent = math.frexp(value)
    integer = int(math.floor(significand*(1 << 31)+.5))
    if integer == 1 << 31:
        integer //= 2
        exponent += 1
    if exponent < -31:
        integer = exponent = 0
    return integer, exponent


def multiply(values, m, shift):
    v = np.asarray(values, dtype=np.int64)
    if not -31 <= shift <= 0 or not 0 <= m <= 2147483647:
        raise UnsupportedModel("Expected a multiplier no greater than one")
    high = (v*m+(1 << 30)) >> 31
    if shift == 0:
        return high
    mask = (1 << -shift)-1
    return (high >> -shift)+((high & mask) > ((mask >> 1)+(high < 0)))


def add_parameters(xs, xz, bs, bz, ys, yz, low, high):
    twice_max = 2*max(xs, bs)
    xm, xshift = multiplier(xs/twice_max)
    bm, bshift = multiplier(bs/twice_max)
    ym, yshift = multiplier(twice_max/((1 << 20)*ys))
    if yshift > 0:
        raise UnsupportedModel("ADD output scale is too small for the bounded double-rounding kernel")
    domain = np.arange(-128, 128, dtype=np.int64)
    xtable = multiply((domain-xz)*(1 << 20), xm, xshift)
    btable = multiply((domain-bz)*(1 << 20), bm, bshift)
    return np.concatenate(([ym, yshift, yz, low, high, 0, 0, 0], xtable, btable)).astype("<i4")
