"""
Contains LD statistics object
"""
import logging
logging.basicConfig()
logger = loggin.getLogger('LDstats_mod')

import os,sys
import numpy, numpy as np

class LDstats(numpy.ma.masked_array):
    """
    Represents linkage disequilibrium statistics (LDstats).
    
    LDstats are represented as an array of 
    
    The constructor has the format:
        y = moments.LD.LDstats(data, mask, order, num_pops)
        
        data: The list of statistics
        mask: An optional array of the same size. 'True' entries in this array
              are masked in LDstats. These represent missing data, invalid entries,
              or entries we wish to ignore when performing inference using LD statistics.
        order: The order of the system for D (D^n). For more than one population, we only
               have order 2 statistics.
        num_pops: Number of populations. For one population, higher order statistics may be 
                  computed.
    
    
    """
    def __new__(subtype, data, mask=numpy.ma.nomask, order=None, num_pops=None,
                dtype=float, copy=True, fill_value=numpy.nan, keep_mask=True):
        if num_pops == None:
            raise ValueError('Specify number of populations as num_pops=.')
        
        if num_pops == 1 and order == None:
            raise ValueError('If number of populations is one, order must be specified.')
        elif num_pops > 1 and order == None:
            order = 2
        
        data = numpy.asanyarray(data)
        
        if mask is numpy.ma.nomask:
            mask = numpy.ma.make_mask_none(data.shape)
        
        subarr = numpy.ma.masked_array(data, mask=mask, dtype=dtype, copy=copy,
                                       fill_value=fill_value, keep_mask=keep_mask)
        
        subarr = subarr.view(subtype)
        if hasattr(data, 'order'):
            subarr.order = order
        if hasattr(data, 'num_pops'):
            subarr.num_pops = num_pops
        
        return subarr
    
    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, return_comments=False):
        pass
    
    def to_file():
        pass


    # Ensures that when arithmetic is done with LDstats objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec(% {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec(% {'method':method})
    
    def integrate(self, nu, tf, dt=0.01, )


# Allow LDstats objects to be pickled.
import copy_reg
def LDstats_pickler(y):
    return LDstats_unpickler, (y.data, y.mask, y.order, y.num_pops)
def LDstats_unpickler(data, mask, order, num_pops):
    return LDstats(data, mask, order=order, num_pops=num_pops)
copy_reg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
