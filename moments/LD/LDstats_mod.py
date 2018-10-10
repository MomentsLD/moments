"""
Contains LD statistics object
"""
import logging
logging.basicConfig()
logger = logging.getLogger('LDstats_mod')

import os,sys
import numpy, numpy as np

from moments.LD import Numerics

class LDstats(numpy.ma.masked_array):
    """
    Represents linkage disequilibrium statistics (LDstats).
    
    LDstats are represented as an array of statistics over two locus pairs for 
    a given recombination distance.
    
    The constructor has the format:
        y = moments.LD.LDstats(data, mask, order, num_pops)
        
        data: The list of statistics
        mask: An optional array of the same size. 'True' entries in this array
              are masked in LDstats. These represent missing data, invalid 
              entries, or entries we wish to ignore when performing inference 
              using LD statistics.
        order: The order of the system for D (D^n). For more than one 
               population, we only have order 2 statistics.
        num_pops: Number of populations. For one population, higher order 
                  statistics may be computed.
    """
    def __new__(subtype, data, mask=numpy.ma.nomask, order=2, num_pops=1,
                dtype=float, copy=True, fill_value=numpy.nan, keep_mask=True,
                shrink=True, pop_ids=None, basis=None):
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
                                    fill_value=fill_value, keep_mask=keep_mask,
                                    shrink=True)
        
        subarr = subarr.view(subtype)
        if hasattr(data, 'order'):
            subarr.order = data.order
        else:
            subarr.order = order
        
        if hasattr(data, 'num_pops'):
            subarr.num_pops = data.num_pops
        else:
            subarr.num_pops = num_pops
        
        if hasattr(data, 'pop_ids'):
            subarr.pop_ids = data.pop_ids
        else:
            subarr.pop_ids = pop_ids
        
        if hasattr(data, 'basis'):
            subarr.basis = data.basis
        else:
            subarr.basis = basis
        return subarr

    # See http://www.scipy.org/Subclasses for information on the
    # __array_finalize__ and __array_wrap__ methods. I had to do some debugging
    # myself to discover that I also needed _update_from.
    # Also, see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    # Also, see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    #
    # We need these methods to ensure extra attributes get copied along when
    # we do arithmetic on the LD stats.
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.order = getattr(obj, 'order', 'unspecified')
        self.num_pops = getattr(obj, 'num_pops', 'unspecified')
        self.pop_ids = getattr(obj, 'pop_ids', 'unspecified')
        self.basis = getattr(obj, 'basis', 'unspecified')
    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.order = self.order
        result.num_pops = self.num_pops
        result.pop_ids = self.pop_ids
        result.basis = self.basis
        return result
    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'order'):
            self.order = obj.order
        if hasattr(obj, 'num_pops'):
            self.num_pops = obj.num_pops
        if hasattr(obj, 'pop_ids'):
            self.pop_ids = obj.pop_ids
        if hasattr(obj, 'basis'):
            self.basis = obj.basis
    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        return 'LDstats(%s, num_pops=%s, order=%s, basis=%s)'\
                % (str(self), str(self.num_pops), str(self.order), self.basis)

    def names(self):
        if self.order == None:
            raise ValueError('Order must be specified (as stats.order)')
        if self.num_pops == None:
            raise ValueError('Number of populations must be specified (as stats.num_pops)')
        
        if self.num_pops == 1:
            if self.basis == 'pi':
                return Numerics.moment_names_onepop(self.order)
            elif self.basis == 'z':
                return Numerics.moment_names_multipop(self.num_pops)
        else:
            return Numerics.moment_names_multipop(self.num_pops)
    
    
    def split(self, pop_to_split):
        """
        y: LDstats object for 
        pop_to_split: index of population to split
        So split(y, 2, 1) would split 2 to 3 pops, splitting pop1
        If the populations are labeled [pop1, pop2], with the new pop pop_new, 
        the output statistics would have population order [pop1, pop2, pop_new]
        New population always appended on end. 
        """
        if self.basis != 'z':
            raise ValueError("We need to be in the z-basis to split.")
        mom_list_from = Numerics.moment_names_multipop(self.num_pops)
        mom_list_to = Numerics.moment_names_multipop(self.num_pops+1)
        y_from = self.data
        y_new = np.ones(len(mom_list_to))
        # dictionary point where stats in mom_list_to come from in mom_list_from
        points = {}
        for mom_to in mom_list_to:
            if mom_to in mom_list_from:
                points[mom_to] = mom_to
            else:
                mom_from_split = mom_to.split('_')
                for ii in range(1,len(mom_from_split)):
                    if int(mom_from_split[ii]) == self.num_pops+1:
                        mom_from_split[ii] = str(pop_to_split)
                    if mom_from_split[0] == 'DD':
                        mom_from_split[1:] = [str(s) for s in sorted([int(x) for x in mom_from_split[1:]])]
                    if mom_from_split[0] == 'zz':
                        mom_from_split[1:3] = [str(s) for s in sorted([int(x) for x in mom_from_split[1:3]])]
                        mom_from_split[3:] = [str(s) for s in sorted([int(x) for x in mom_from_split[3:]])]
                    if mom_from_split[0] == 'zp':
                        mom_from_split[1:] = [str(s) for s in sorted([int(x) for x in mom_from_split[1:]])]
                    if mom_from_split[0] == 'zq':
                        mom_from_split[1:] = [str(s) for s in sorted([int(x) for x in mom_from_split[1:]])]
                mom_from = '_'.join(mom_from_split)
                points[mom_to] = mom_from
        for ii,mom_to in zip(range(len(mom_list_to)),mom_list_to):
            y_new[ii] = self[mom_list_from.index(points[mom_to])]
        return LDstats(y_new, num_pops=self.num_pops+1, order=self.order, basis=self.basis)
    
    def swap_pops(self,pop1,pop2):
        """
        like swapaxes for switching population ordering
        """
        if pop1 > self.num_pops or pop2 > self.num_pops or pop1 < 1 or pop2 < 1:
            raise ValueError("Invalid population number specified.")
        if pop1 == pop2:
            return self
        else:
            mom_list = Numerics.moment_names_multipop(self.num_pops)
            y_new = np.zeros(len(mom_list))
            pops_old = range(1,self.num_pops+1)
            pops_new = range(1,self.num_pops+1)
            pops_new[pop1-1] = pop2
            pops_new[pop2-1] = pop1
            d = dict(zip(pops_old, pops_new))
            for ii,mom in enumerate(mom_list):
                if mom == '1':
                    y_new[ii] = 1.
                    continue
                pops_mom = [int(p) for p in mom.split('_')[1:]]
                pops_mom_new = [d.get(p) for p in pops_mom]
                mom_new = mom.split('_')[0] + '_' + '_'.join([str(p) for p in pops_mom_new])
                y_new[ii] = self.data[mom_list.index(Numerics.map_moment(mom_new))]
            return LDstats(y_new, num_pops=self.num_pops, order=self.order, basis=self.basis)

    def marginalize(self,pops):
        """
        Marginalize over the LDstats, removing moments for given pops
        pops could be a single population or a list of pops
        assume that we have 
        """
        if self.num_pops == 1:
            print("You're trying to marginalize a single population model.")
            return self
        if hasattr(pops,"__len__") == False:
            pops = [pops]
        
        # multiple pop indices to marginalize over
        names_from = Numerics.moment_names_multipop(self.num_pops)
        names_to = Numerics.moment_names_multipop(self.num_pops - len(pops))
        y = np.zeros(len(names_to))
        count = 0
        for mom in names_from:
            mom_pops = [int(p) for p in mom.split('_')[1:]]
            if len(np.intersect1d(pops, mom_pops)) == 0:
                y[count] = self[names_from.index(mom)]
                count += 1
        return LDstats(y, num_pops=self.num_pops-len(pops), order=self.order, basis=self.basis)

#    def merge(self, f):
#        if self.num_pops == 2:
#            y_new = Numerics.merge_2pop(self.data,f)
#            return LDstats(y_new, num_pops=1, order=self.order, basis=self.basis)
#        else:
#            raise ValueError("merge function is 2->1 pop.")
    
    def merge(self, pop1, pop2, f):
        y_new = Numerics.admix_npops(self.data, self.num_pops, pop1, pop2, f)
        y_new = LDstats(y_new, num_pops=self.num_pops+1, order=self.order, basis=self.basis)
        y_new = y_new.swap_pops(pop1, y_new.num_pops)
        y_new = y_new.marginalize(pop2)
        y_new = y_new.marginalize(y_new.num_pops)
        return y_new

    def admix(self, pop1, pop2, f):
        """
        admixture between pop1 and pop2, with fraction f of migrants from pop 1 (1-f from pop 2)
        returns a new LDstats object with the admixed population in last index
        """
        if self.num_pops < 2 or pop1 > self.num_pops or pop2 > self.num_pops:
            raise ValueError("Something wrong with calling admix.")
        elif pop1 >= pop2:
            raise ValueError("pop1 must be less than pop2, and f is fraction from pop1.")
        else:
            y_new = Numerics.admix_npops(self.data, self.num_pops, pop1, pop2, f)
            return LDstats(y_new, num_pops=self.num_pops+1, order=self.order, basis=self.basis)
    
    def pulse_migrate(self, pop_from, pop_to, f):
        """
        Pulse migration event, from pop_from to pop_to.
        After pulse migration event, pop_to is composed of (1-f) from pop_to and
        f from pop_from.
        
        This is equivalent to admix into new, so we'll use the admix function 
            and then rearrange and marginalize populations
        """
        if pop_from < pop_to:
            y_new = self.admix(pop_from, pop_to, f)
        elif pop_to < pop_from:
            y_new = self.admix(pop_to, pop_from, 1-f)
        
        y_new = y_new.swap_pops(pop_to, y_new.num_pops)
        y_new = y_new.marginalize([y_new.num_pops])
        return y_new
    
    def switch_basis(self):
        if self.basis == 'pi':
            pi2 = self.data[2]
            sigp = self.data[3]
            sigq = self.data[4]
            zz = 1-4*sigp-4*sigq+16*pi2
            zp = 1-4*sigp
            zq = 1-4*sigq
            y_new = [self.data[0], self.data[1], zz, zp, zq, 1.]
            return LDstats(y_new, num_pops=1, order=2, basis='z')
        elif self.basis == 'z':
            zz = self.data[2]
            zp = self.data[3]
            zq = self.data[4]
            pi2 = 1./16*(zz-zp-zq+1)
            sigp = 1./4*(1-zp)
            sigq = 1./4*(1-zq)
            y_new = [self.data[0], self.data[1], pi2, sigp, sigq, 1.]
            return LDstats(y_new, num_pops=1, order=2, basis='pi')
        else:
            raise ValueError("basis is not defined on this LD object")

    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, return_comments=False):
        """
        Read LD statistics from file
        
        fid: string with file name to read from or an open file object.
        return_comments: If true, the return value is (y, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = file(fid, 'r')
        
        line = fid.readline()
        # Strip out the comments
        comments = []
        while line.startswith('#'):
            comments.append(line[1:].strip())
            line = fid.readline()
        
        # Read the order, model type of the data
        line_spl = line.split()
        order = int(line_spl[0])
        model_type = line_spl[1]
        num_pops = int(line_spl[2])
        if len(line_spl) > 3:
            # get the pop_ids
            pop_ids = line_spl[3:]
        else:
            pop_ids = None
        
        data = numpy.fromstring(fid.readline().strip(), sep=' ')

        maskline = fid.readline().strip()
        mask = numpy.fromstring(maskline, sep=' ')

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        y = LDstats(data, mask, order=order, num_pops=num_pops, pop_ids=pop_ids)

        if not return_comments:
            return y
        else:
            return y,comments
        
        
    
    def to_file(self, fid, precision=16, comment_lines=[]):
        """
        Write LD statistics to file.
        
        fid: string with file name to write to or an open file object.
        precision: precision with which to write out entries of the LD stats. 
                   (They are formated via %.<p>g, where <p> is the precision.)
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
        foldmaskinfo: If False, folding and mask and population label
                      information will not be saved. 

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing an integer giving the order of LD 
              statistics. So this line would be '4' if we have modeled
              the order D^4 system.
            On the *same line*, the string 'onepop' or 'multipop' denoting 
              whether we have a single or multipop object.
            On the *same line*, number of populations (must be '1' if we have
              written 'onepop', and can be any integer for 'multipop').
            On the *same line*, optional strings each containing the population
              labels in quotes separated by spaces, e.g. "pop 1" "pop 2"
            A single line giving the array elements. The order of elements is 
              given by order in Numerics.moment_names...
            A single line giving the elements of the mask in the same order as
              the data line. '1' indicates masked, '0' indicates unmasked.
        """
        # Open the file object.
        newfile = False
        if not hasattr(fid, 'write'):
            newfile = True
            fid = file(fid, 'w')

        # Write comments
        for line in comment_lines:
            fid.write('# ')
            fid.write(line.strip())
            fid.write(os.linesep)

        # Write out the order of the LDstats
        fid.write('%i ' % self.order)
        
        # Write out model type (onepop or multipop) and number of populations
        if len(self.data) == 5:
            fid.write('onepop %i' % self.num_pops)
        else:
            fid.write('multipop %i' % self.num_pops)
        
        if self.pop_ids is not None:
            for label in self.pop_ids:
                fid.write(' "%s"' % label)
        
        fid.write(os.linesep)

        # Write the data to the file
        self.data.tofile(fid, ' ', '%%.%ig' % precision)
        fid.write(os.linesep)

        # Write the mask to the file
        numpy.asarray(self.mask,int).tofile(fid, ' ')
        fid.write(os.linesep)

        # Close file
        if newfile:
            fid.close()

    tofile = to_file
        
    
    # Ensures that when arithmetic is done with LDstats objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    outLDstats = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   order=self.order, num_pops=self.num_pops)
    return outLDstats
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    return self
""" % {'method':method})
    
    
    def integrate(self, nu, tf, dt=0.001, rho=None, theta=0.0008, ism=True, 
                  m=[]):
        """
        Integrates the LD statistics forward in time. The tricky part is 
        combining single population and multi-population integration routines. 
        11/30: For now, this is just single population
        nu: relative population size, may be a function of time, given as a list [nu1, nu2, ...]
        tf: total time to integrate
        dt: integration timestep
        rho: can be a single recombination rate or list of recombination rates 
             (in which case we are integrating a list of LD stats for each rate)
        theta: per base population-scaled mutation rate (4N*mu)
               if we pass [theta1, theta2], differing mutation rates at left and right 
               locus, implemented in the ISM=True model
        ism: if True, we use the infinite sites model, otherwise we use a 
             reversible mutation model (equal forward and reverse mutation 
             rates)
        m: migration matrix (num_pops x num_pops, storing m_ij migration rate
           from i to j
        """
        order = self.order
        num_pops = self.num_pops
        
        if tf == 0.0:
            return
        
        if rho == None:
            print('Remember to specify rho. Default rho set to 0.0')
            rho = 0.0
        
        if hasattr(theta, '__len__') and ism == False:
            raise ValueError("We can only integrate with differing mutation rates at the two loci in the infinite sites model (set ism=True).")
        
        if self.basis == 'pi' and num_pops != 1:
            raise ValueError("Careful: for num_pops > 1, basis must be z. Try swapping basis before splitting.")
        
        if self.basis == 'pi':
            self.data[:] = Numerics.integrate(self.data, nu, tf, rho=rho, 
                                    theta=theta, order=order, dt=dt, ism=ism)
        elif self.basis == 'z':
            # and multipop with same basis
            self.data[:] = Numerics.integrate_multipop(self.data, nu, tf, dt=dt,
                                    rho=rho, theta=theta, m=m, 
                                    num_pops=self.num_pops, ism=ism)
        else:
            raise ValueError("basis is not defined on this LD object")

# Allow LDstats objects to be pickled.
import copy_reg
def LDstats_pickler(y):
    return LDstats_unpickler, (y.data, y.mask, y.order, y.num_pops)
def LDstats_unpickler(data, mask, order, num_pops):
    return LDstats(data, mask, order=order, num_pops=num_pops)
copy_reg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
