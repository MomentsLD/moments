"""
Contains LD statistics object
"""
import logging
logging.basicConfig()
logger = logging.getLogger('LDstats_mod')

import os,sys
import numpy, numpy as np
import copy

import Numerics2, Util

class LDstats2(list):
    """
    Represents linkage disequilibrium statistics, stored in a more accessible
        way, with heterozygosity separated and multiple rho values (LDstats2).
    
    (Only allows equal mutation rates, and all statistics here are order 2 (D^2))
    
    LDstats are represented as a list of statistics over two locus pairs for 
    a given recombination distance.
    
    It has form Y[0] = two locus stats for rho[0], ... Y[n] = two locus stats for rho[n], H
    Each Y[i] is a condensed set of statistics (E[D_i(1-2p_j)(1-2q_k)] = E[D_i(1-2p_k)(1-2q_j)]
    
    The constructor has the format:
        y = moments.LD.LDstats(data, num_pops, pop_ids)
        data: The list of statistics
        num_pops: Number of populations. For one population, higher order 
                  statistics may be computed.
        pop_ids: population names in order that statistics are represented here.
    """
    def __new__(self, data, num_pops=None, pop_ids=None):
        if num_pops == None:
            raise ValueError('Specify number of populations as num_pops=.')
        my_list = super(LDstats2, self).__new__(self, data, num_pops=None, pop_ids=None)
        
        if hasattr(data, 'num_pops'):
            my_list.num_pops = data.num_pops
        else:
            my_list.num_pops = num_pops
        
        if hasattr(data, 'pop_ids'):
            my_list.pop_ids = data.pop_ids
        else:
            my_list.pop_ids = pop_ids
        
        return my_list

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
    

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
        self.num_pops = getattr(obj, 'num_pops', 'unspecified')
        self.pop_ids = getattr(obj, 'pop_ids', 'unspecified')
    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.num_pops = self.num_pops
        result.pop_ids = self.pop_ids
        return result
    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'num_pops'):
            self.num_pops = obj.num_pops
        if hasattr(obj, 'pop_ids'):
            self.pop_ids = obj.pop_ids

    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        ys = np.asanyarray(self[:-1])
        h = self[-1]
        return 'LDstats(%s, %s, num_pops=%s, pop_ids=%s)'\
                % (str(ys), str(h), str(self.num_pops), str(self.pop_ids))

    def names(self):
        if self.num_pops == None:
            raise ValueError('Number of populations must be specified (as stats.num_pops)')
        
        return Util.moment_names(self.num_pops) # returns (ld_stat_names, het_stat_names)
    
    def split(self, pop_to_split):
        """
        y: LDstats object for 
        pop_to_split: index of population to split
        If the populations are labeled [pop1, pop2], with the new pop pop_new, 
        the output statistics would have population order [pop1, pop2, pop_new]
        New population always appended on end. 
        """
        if pop_to_split > self.num_pops:
            raise ValueError("population to split larger than number of pops")

        h = self[-1]
        ys = self[:-1]
        
        h_new = Numerics2.split_h(h, pop_to_split, self.num_pops)
        ys_new = []
        for y in ys:
            ys_new.append(Numerics2.split_ld(y, pop_to_split, self.num_pops))
        
        return LDstats2(ys_new+[h_new], num_pops=self.num_pops+1)
        
            
    def swap_pops(self, pop1, pop2):
        """
        like swapaxes for switching population ordering
        pop1 and pop2 are indexes of 
        """
        if pop1 > self.num_pops or pop2 > self.num_pops or pop1 < 1 or pop2 < 1:
            raise ValueError("Invalid population number specified.")
        if pop1 == pop2:
            return self            

        mom_list = Util.moment_names(self.num_pops)
        h_new = np.zeros(len(mom_list[-1]))

        if len(self) == 2:
            y_new = np.zeros(len(mom_list[0]))
        if len(self) > 2:
            ys_new = [np.zeros(len(mom_list[0])) for i in range(len(self)-1)]
        
        pops_old = list(range(1,self.num_pops+1))
        pops_new = list(range(1,self.num_pops+1))
        pops_new[pop1-1] = pop2
        pops_new[pop2-1] = pop1
        
        d = dict(zip(pops_old, pops_new))
        
        for ii,mom in enumerate(mom_list[-1]):
            pops_mom = [int(p) for p in mom.split('_')[1:]]
            pops_mom_new = [d.get(p) for p in pops_mom]
            mom_new = mom.split('_')[0] + '_' + '_'.join([str(p) for p in pops_mom_new])
            h_new[ii] = self[-1][mom_list[-1].index(Util.map_moment(mom_new))]

        if len(self) > 1:
            for ii,mom in enumerate(mom_list[0]):
                pops_mom = [int(p) for p in mom.split('_')[1:]]
                pops_mom_new = [d.get(p) for p in pops_mom]
                mom_new = mom.split('_')[0] + '_' + '_'.join([str(p) for p in pops_mom_new])
                if len(self) == 2:
                    y_new[ii] = self[0][mom_list[0].index(Util.map_moment(mom_new))]
                elif len(self) > 2:
                    for jj in range(len(self)-1):
                        ys_new[jj][ii] = self[jj][mom_list[0].index(Util.map_moment(mom_new))]
                
        if self.pop_ids is not None:
            current_order = self.pop_ids
            new_order = [self.pop_ids[d[ii]-1] for ii in range(1,self.num_pops+1)]

        if len(self) == 1:
            return LDstats2(h_new, num_pops=self.num_pops, pop_ids=new_order)
        elif len(self) == 2:
            return LDstats2([y_new,h_new], num_pops=self.num_pops, pop_ids=new_order)
        else:
            return LDstats2(ys_new+[h_new], num_pops=self.num_pops, pop_ids=new_order)


    def marginalize(self,pops):
        """
        Marginalize over the LDstats, removing moments for given pops
        pops could be a single population or a list of pops
        assume that we have 
        """
        if self.num_pops == 1:
            print("no populations left.")
            return self
        if hasattr(pops,"__len__") == False:
            pops = [pops]

        # multiple pop indices to marginalize over
        names_from_ld, names_from_h = Util.moment_names(self.num_pops)
        names_to_ld, names_to_h = Util.moment_names(self.num_pops - len(pops))
        y_new = [np.zeros(len(names_to_ld)) for i in range(len(self)-1)] + [np.zeros(len(names_to_h))]
        count = 0
        for mom in names_from_ld:
            mom_pops = [int(p) for p in mom.split('_')[1:]]
            if len(np.intersect1d(pops, mom_pops)) == 0:
                for ii in range(len(y_new)-1):
                    y_new[ii][count] = self[ii][names_from_ld.index(mom)]
                count += 1
        if self.pop_ids == None:
            return LDstats2(y_new, num_pops=self.num_pops-len(pops))
        else:
            new_ids = copy.copy(self.pop_ids)
            for ii in sorted(pops)[::-1]:
                new_ids.pop(ii-1)
            return LDstats2(y_new, num_pops=self.num_pops-len(pops), pop_ids=new_ids)

            
#    def merge(self, f):
#        if self.num_pops == 2:
#            y_new = Numerics.merge_2pop(self,f)
#            return LDstats(y_new, num_pops=1, order=self.order, basis=self.basis)
#        else:
#            raise ValueError("merge function is 2->1 pop.")
    
#    def merge(self, pop1, pop2, f):
#        y_new = Numerics.admix_npops(self, self.num_pops, pop1, pop2, f)
#        y_new = LDstats(y_new, num_pops=self.num_pops+1, order=self.order, basis=self.basis)
#        y_new = y_new.swap_pops(pop1, y_new.num_pops)
#        y_new = y_new.marginalize([pop2,y_new.num_pops])
#        return y_new

#    def admix(self, pop1, pop2, f):
#        """
#        admixture between pop1 and pop2, with fraction f of migrants from pop 1 (1-f from pop 2)
#        returns a new LDstats object with the admixed population in last index
#        """
#        if self.num_pops < 2 or pop1 > self.num_pops or pop2 > self.num_pops:
#            raise ValueError("Something wrong with calling admix.")
#        elif pop1 >= pop2:
#            raise ValueError("pop1 must be less than pop2, and f is fraction from pop1.")
#        else:
#### Numerics.admix_npops probably needs to check for basis
#            y_new = Numerics.admix_npops(self, self.num_pops, pop1, pop2, f)
#            return LDstats(y_new, num_pops=self.num_pops+1, order=self.order, basis=self.basis)
    
#    def pulse_migrate(self, pop_from, pop_to, f):
#        """
#        Pulse migration event, from pop_from to pop_to.
#        After pulse migration event, pop_to is composed of (1-f) from pop_to and
#        f from pop_from.
#        
#        This is equivalent to admix into new, so we'll use the admix function 
#            and then rearrange and marginalize populations
#        """
#        if pop_from < pop_to:
#            y_new = self.admix(pop_from, pop_to, f)
#        elif pop_to < pop_from:
#            y_new = self.admix(pop_to, pop_from, 1-f)
#        
#        y_new = y_new.swap_pops(pop_to, y_new.num_pops)
#        y_new = y_new.marginalize([y_new.num_pops])
#        return y_new
    
#    # Make from_file a static method, so we can use it without an instance.
#    @staticmethod
#    def from_file(fid, return_comments=False):
#        """
#        Read LD statistics from file
#        
#        fid: string with file name to read from or an open file object.
#        return_comments: If true, the return value is (y, comments), where
#                         comments is a list of strings containing the comments
#                         from the file (without #'s).
#        """
#        newfile = False
#        # Try to read from fid. If we can't, assume it's something that we can
#        # use to open a file.
#        if not hasattr(fid, 'read'):
#            newfile = True
#            fid = file(fid, 'r')
#        
#        line = fid.readline()
#        # Strip out the comments
#        comments = []
#        while line.startswith('#'):
#            comments.append(line[1:].strip())
#            line = fid.readline()
#        
#        # Read the order, model type of the data
#        line_spl = line.split()
#        order = int(line_spl[0])
#        model_type = line_spl[1]
#        num_pops = int(line_spl[2])
#        if len(line_spl) > 3:
#            # get the pop_ids
#            pop_ids = line_spl[3:]
#        else:
#            pop_ids = None
#        
#        data = numpy.fromstring(fid.readline().strip(), sep=' ')
#
#        maskline = fid.readline().strip()
#        mask = numpy.fromstring(maskline, sep=' ')
#
#        # If we opened a new file, clean it up.
#        if newfile:
#            fid.close()
#
#        y = LDstats(data, mask, order=order, num_pops=num_pops, pop_ids=pop_ids)
#
#        if not return_comments:
#            return y
#        else:
#            return y,comments
#        
#        
#    
#    def to_file(self, fid, precision=16, comment_lines=[]):
#        """
#        Write LD statistics to file.
#        
#        fid: string with file name to write to or an open file object.
#        precision: precision with which to write out entries of the LD stats. 
#                   (They are formated via %.<p>g, where <p> is the precision.)
#        comment lines: list of strings to be used as comment lines in the header
#                       of the output file.
#        foldmaskinfo: If False, folding and mask and population label
#                      information will not be saved. 
#
#        The file format is:
#            # Any number of comment lines beginning with a '#'
#            A single line containing an integer giving the order of LD 
#              statistics. So this line would be '4' if we have modeled
#              the order D^4 system.
#            On the *same line*, the string 'onepop' or 'multipop' denoting 
#              whether we have a single or multipop object.
#            On the *same line*, number of populations (must be '1' if we have
#              written 'onepop', and can be any integer for 'multipop').
#            On the *same line*, optional strings each containing the population
#              labels in quotes separated by spaces, e.g. "pop 1" "pop 2"
#            A single line giving the array elements. The order of elements is 
#              given by order in Numerics.moment_names...
#            A single line giving the elements of the mask in the same order as
#              the data line. '1' indicates masked, '0' indicates unmasked.
#        """
#        # Open the file object.
#        newfile = False
#        if not hasattr(fid, 'write'):
#            newfile = True
#            fid = file(fid, 'w')
#
#        # Write comments
#        for line in comment_lines:
#            fid.write('# ')
#            fid.write(line.strip())
#            fid.write(os.linesep)
#
#        # Write out the order of the LDstats
#        fid.write('%i ' % self.order)
#        
#        # Write out model type (onepop or multipop) and number of populations
#        if len(self) == 5:
#            fid.write('onepop %i' % self.num_pops)
#        else:
#            fid.write('multipop %i' % self.num_pops)
#        
#        if self.pop_ids is not None:
#            for label in self.pop_ids:
#                fid.write(' "%s"' % label)
#        
#        fid.write(os.linesep)
#
#        # Write the data to the file
#        self.tofile(fid, ' ', '%%.%ig' % precision)
#        fid.write(os.linesep)
#
#        # Write the mask to the file
#        numpy.asarray(self.mask,int).tofile(fid, ' ')
#        fid.write(os.linesep)
#
#        # Close file
#        if newfile:
#            fid.close()
#
#    tofile = to_file
        
    
    # Ensures that when arithmetic is done with LDstats objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        newdata = self.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.%(method)s (other)
        newmask = self.mask
    outLDstats = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   num_pops=self.num_pops, pop_ids=self.pop_ids)
    return outLDstats
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        self.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.%(method)s (other)
    return self
""" % {'method':method})
    
    
    def integrate(self, nu, tf, dt=0.001, rho=None, theta=0.001, m=None, frozen=None):
        """
        Integrates the LD statistics forward in time. The tricky part is 
        combining single population and multi-population integration routines. 
        nu: relative population size, may be a function of time, given as a list [nu1, nu2, ...]
        tf: total time to integrate
        dt: integration timestep
        rho: can be a single recombination rate or list of recombination rates 
             (in which case we are integrating a list of LD stats for each rate)
        theta: per base population-scaled mutation rate (4N*mu)
               if we pass [theta1, theta2], differing mutation rates at left and right 
               locus, implemented in the ISM=True model
        m: migration matrix (num_pops x num_pops, storing m_ij migration rate
           from i to j
        """
        num_pops = self.num_pops
        
        if tf == 0.0:
            return
        
        if rho == None and len(self) > 1:
            print('Rho not set, must specify.')
            return
        elif np.isscalar(rho) and len(self) != 2:
            print('Single rho passed but LD object has additional statistics.')
        elif np.isscalar(rho) == False and len(rho) != len(self)-1:
            print('Mismatch of rhos passed and size of LD object.')
        
        if m is not None and num_pops > 1:
            if np.shape(m) != (num_pops, num_pops):
                raise ValueError("migration matrix incorrectly defined for number of pops.")
        
        self[:] = Numerics2.integrate(self, nu, tf, dt=dt,
                                    rho=rho, theta=theta, m=m,
                                    num_pops=num_pops, frozen=frozen)

# Allow LDstats objects to be pickled.
try:
    import copy_reg
except:
    import copyreg
def LDstats_pickler(y):
    return LDstats_unpickler, (y.data, y.mask, y.num_pops, y.pop_ids)
def LDstats_unpickler(data, mask, num_pops, pop_ids):
    return LDstats(data, mask, num_pops=num_pops, pop_ids=pop_ids)

try:
    copy_reg.pickle(LDstats2, LDstats_pickler, LDstats_unpickler)
except:
    copyreg.pickle(LDstats2, LDstats_pickler, LDstats_unpickler)
