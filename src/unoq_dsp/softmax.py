"""Q5.26 exponent tables for the TFLite/gemmlowp INT8 softmax contract.

Fixed-point polynomial and constants adapted from gemmlowp fixedpoint.h
(Copyright 2015 The Gemmlowp Authors, Apache-2.0). See THIRD_PARTY_NOTICES.md.
"""
import math
from .quantization import multiplier
from .errors import UnsupportedModel

def high_mul(a,b):return max(-2147483648,min(2147483647,(a*b+(1<<30))>>31))
def divide_pot(value,shift):
    mask=(1<<shift)-1
    return (value>>shift)+((value&mask)>((mask>>1)+(value<0)))

def exponent_q31(raw):
    if raw==0:return 2147483647
    quarter=1<<24;reduced=(raw&(quarter-1))-quarter
    x=(reduced<<5)+(1<<28);x2=high_mul(x,x);x3=high_mul(x2,x);x4=high_mul(x2,x2)
    polynomial=divide_pot(high_mul(divide_pot(x4,2)+x3,715827883)+x2,1)
    result=1895147668+high_mul(1895147668,x+polynomial)
    remainder=reduced-raw
    for exponent,constant in enumerate((1672461947,1302514674,790015084,290630308,39332535,720401,242),start=-2):
        if remainder&(1<<(26+exponent)):result=high_mul(result,constant)
    return result

def exp_table(scale,beta):
    if not math.isfinite(scale*beta) or scale<=0 or beta<=0:raise UnsupportedModel('Invalid softmax scale')
    qm,shift=multiplier(min(scale*beta*(1<<26),(1<<31)-1))
    if not 0<=shift<=31:raise UnsupportedModel('Softmax input scale below supported fixed-point range')
    radius=((1<<5)-1)*(1<<26)//(1<<shift)
    return [exponent_q31(high_mul(-i*(1<<shift),qm)) if i<=radius else 0 for i in range(256)]
