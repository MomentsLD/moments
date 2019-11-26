"""
Contains LD statistics object

XXX I made a stupid decision early on to index populations from 1 instead of 0.
    At some point I need to go through and fix all that, perhaps...

"""
import logging
logging.basicConfig()
logger = logging.getLogger('LDstats_mod')

import os,sys
import numpy, numpy as np
import copy

from . import Numerics, Util

class LDstats(list):
    """
    Represents linkage disequilibrium statistics, stored in a more accessible
        way, with heterozygosity separated and multiple rho values (LDstats).
    
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
        my_list = super(LDstats, self).__new__(self, data, num_pops=None, pop_ids=None)
        
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
    
    def LD(self, pops = None):
        """
        returns LD stats for populations given (if None, returns all)
        """
        if len(self) > 1:
            if pops is not None:
                to_marginalize = list(set(range(1,self.num_pops+1)) - set(pops))
                Y_new = self.marginalize(to_marginalize)
                if len(self) == 2:
                    return Y_new[:-1][0]
                else:
                    return Y_new[:-1]
            else:
                return self[:-1]
    
    def H(self, pops = None):
        if pops is not None:
            to_marginalize = list(set(range(1,self.num_pops+1)) - set(pops))
            Y_new = self.marginalize(to_marginalize)
            return Y_new[-1]
        else:
            return self[-1]
        
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
        
        h_new = Numerics.split_h(h, pop_to_split, self.num_pops)
        ys_new = []
        for y in ys:
            ys_new.append(Numerics.split_ld(y, pop_to_split, self.num_pops))
        
        return LDstats(ys_new+[h_new], num_pops=self.num_pops+1)
        
            
    def swap_pops(self, pop1, pop2):
        """
        like swapaxes for switching population ordering
        pop1 and pop2 are indexes of the two populations to swap, and also swaps
        their population id names in self.pop_ids
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
        else:
            new_order = None

        if len(self) == 1:
            return LDstats([h_new], num_pops=self.num_pops, pop_ids=new_order)
        elif len(self) == 2:
            return LDstats([y_new,h_new], num_pops=self.num_pops, pop_ids=new_order)
        else:
            return LDstats(ys_new+[h_new], num_pops=self.num_pops, pop_ids=new_order)


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
        count = 0
        for mom in names_from_h:
            mom_pops = [int(p) for p in mom.split('_')[1:]]
            if len(np.intersect1d(pops, mom_pops)) == 0:
                y_new[-1][count] = self[-1][names_from_h.index(mom)]
                count += 1
        
        if self.pop_ids == None:
            return LDstats(y_new, num_pops=self.num_pops-len(pops))
        else:
            new_ids = copy.copy(self.pop_ids)
            for ii in sorted(pops)[::-1]:
                new_ids.pop(ii-1)
            return LDstats(y_new, num_pops=self.num_pops-len(pops), pop_ids=new_ids)

    
    ## Admix takes two populations, creates new population with fractions f, 1-f
    ## Merge takes two populations (pop1, pop2) and merges together with fraction
    ## f from population pop1, and (1-f) from pop2
    ## Pulse migrate again takes two populations, pop1 and pop2, and replaces fraction
    ## f from pop2 with pop1
    ## In each case, we use the admix function, which appends a new population on the end
    ## In the case of merge, we use admix and then marginalize pop1 and pop2
    ## in the case of pulse migrate, we use admix, and then swap new pop with pop2, and
    ## then marginalize pop2, so the new population takes the same position in pop_ids 
    ## that pop2 was previously in
    
    def admix(self, pop1, pop2, f, new_pop=None):
        """
        Admixture between pop1 and pop2, given by indexes. f is the fraction 
            contributed by pop1, so pop2 contributes 1-f.
        If new_pop is left as 'None', the admixed population's name is 'Adm'.
            Otherwise, we can set it with new_pop=new_pop_name.
        """
        if self.num_pops < 2 or pop1 > self.num_pops or pop2 > self.num_pops or pop1 < 1 or pop2 < 1:
            raise ValueError("Improper usage of admix (wrong indices?).")
        else:
            Y_new = Numerics.admix(self, self.num_pops, pop1, pop2, f)
            if self.pop_ids is not None:
                if new_pop == None:
                    new_pop_ids = self.pop_ids + ['Adm']
                else:
                    new_pop_ids = self.pop_ids + [new_pop]
            else:
                new_pop_ids = None
            return LDstats(Y_new, num_pops=self.num_pops+1, pop_ids=new_pop_ids)
    
    def merge(self, pop1, pop2, f, new_pop=None):
        """
        Merger of populations pop1 and pop22, with fraction f from pop1 
            and 1-f from pop2.
        Places new population at the end, then marginalizes pop1 and pop2.
        To admix two populations and keep one or both, use pulse migrate or 
            admix, respectively.
        """
        Y_new = self.admix(pop1, pop2, f, new_pop=new_pop)
        Y_new = Y_new.marginalize([pop1,pop2])
        return Y_new
    
    def pulse_migrate(self, pop1, pop2, f):
        """
        Pulse migration/admixure event from pop1 to pop2, with fraction f 
            replacement. We use the admix function above. We want to keep the 
            original population names the same, if they are given in the LDstats
            object, so we use new_pop=self.pop_ids[pop2].
        We admix pop1 and pop2 with fraction f and 1-f, then swap the new
            admixed population with pop2, then marginalize the original pop2.
        """
        if self.pop_ids is not None:
            Y_new = self.admix(pop1, pop2, f, new_pop=self.pop_ids[pop2-1])
        else:
            Y_new = self.admix(pop1, pop2, f)
        Y_new = Y_new.swap_pops(pop2, Y_new.num_pops)
        Y_new = Y_new.marginalize(Y_new.num_pops)
        return Y_new
    
    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, return_comments=False):
        """
        Read LD statistics from file
        
        fid: string with file name to read from or an open file object.
        return_statistics: If true, returns statistics writen to file.
        return_comments: If true, the return value is (y, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = open(fid, 'r')
        
        line = fid.readline()
        # Strip out the comments
        comments = []
        while line.startswith('#'):
            comments.append(line[1:].strip())
            line = fid.readline()
        
        # Read the num pops and pop_ids, if given
        line_spl = line.split()
        num_pops = int(line_spl[0])
        if len(line_spl) > 1:
            # get the pop_ids
            pop_ids = line_spl[1:]
            if num_pops != len(pop_ids):
                print('Warning: num_pops does not match number of pop_ids.')
        else:
            pop_ids = None
        
        # Get the statistic names
        ld_stats = fid.readline().split()
        het_stats = fid.readline().split()
        if ld_stats == ['ALL']:
            ld_stats = Util.moment_names(num_pops)[0]
        if het_stats == ['ALL']:
            het_stats = Util.moment_names(num_pops)[1]
        statistics = [ld_stats, het_stats]
        
        # Get the number of LD statistic rows and read LD data
        num_ld_rows = int(fid.readline().strip())
        data = []
        for r in range(num_ld_rows):
            data.append(numpy.fromstring(fid.readline().strip(), sep=' '))
        
        # Read heterozygosity data
        data.append(numpy.fromstring(fid.readline().strip(), sep=' '))
        
        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        y = LDstats(data, num_pops=num_pops, pop_ids=pop_ids)

        if not return_comments:
            return y, statistics
        else:
            return y, statistics, comments


    def to_file(self, fid, precision=16, statistics='ALL', comment_lines=[]):
        """
        Write LD statistics to file.
        
        fid: string with file name to write to or an open file object.
        precision: precision with which to write out entries of the LD stats. 
                   (They are formated via %.<p>g, where <p> is the precision.)
        statistics: defaults to 'ALL', meaning all statistics are given in the
                    LDstats object. Otherwise, list of two lists, first giving
                    present LD stats, and the second giving present het stats.
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
                       I use comment lines mainly to record the recombination 
                       bins or distances given in the LDstats (something like
                       "'edges = ' + str(r_edges)".

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing an integer giving the number of
              populations.
            On the *same line*, optional, the names of those populations. If
              names are given, there needs to be the same number of pop_ids
              as the integer number of populations. For example, the line could
              be '3 YRI CEU CHB'.
            A single line giving the names of the *LD* statistics, in the order
              they appear for each recombination rate distance or bin.
              Optionally, this line could read ALL, indicating that every 
              statistic in the basis is given, and in the 'correct' order.
            A single line giving the names of the *heterozygosity* statistics,
              in the order they appear in the final row of data. Optionally,
              this line could read ALL.
            A line giving the number of recombination rate bins/distances we 
              have data for (so we know how many to read)
            One line for each row of LD statistics.
            A single line for the heterozygosity statistics.
        """
        
        # if statistics is ALL, check to make sure the lengths are correct
        if statistics != 'ALL':
            ld_stat_names, het_stat_names = statistics
        else:
            ld_stat_names, het_stat_names = Util.moment_names(self.num_pops)
        
        all_correct_length = 1
        for ld_stats in self.LD():
            if len(ld_stats) != len(ld_stat_names):
                all_correct_length = 0
                break
        if len(self.H()) != len(het_stat_names):
            all_correct_length = 0
        
        if all_correct_length == 0:
            raise ValueError('Length of data arrays does not match expected \
                              length of statistics.')
        
        
        # Open the file object.
        newfile = False
        if not hasattr(fid, 'write'):
            newfile = True
            fid = open(fid, 'w')

        # Write comments
        for line in comment_lines:
            fid.write('# ')
            fid.write(line.strip())
            fid.write(os.linesep)

        # Write out the number of populations and pop_ids if given
        fid.write('%i' % self.num_pops)
        if self.pop_ids is not None:
            for pop in self.pop_ids:
                fid.write(' %s' % pop)
        fid.write(os.linesep)
        
        # Write out LD statistics
        if statistics == 'ALL':
            fid.write(statistics)
        else:
            for stat in statistics[0]:
                fid.write('%s ' % stat)
        fid.write(os.linesep)
        
        # Write out het statistics
        if statistics == 'ALL':
            fid.write(statistics)
        else:
            for stat in statistics[1]:
                fid.write('%s ' % stat)
        fid.write(os.linesep)

        # Write the LD data to the file
        fid.write('%i' % len(self.LD()))
        fid.write(os.linesep)
        for ld_stats in self.LD():
            for stat in ld_stats:
                fid.write('%%.%ig ' % precision % stat)
            fid.write(os.linesep)
        
        # Write the het data to the file
        for stat in self.H():
            fid.write('%%.%ig ' % precision % stat)
        
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
    
    
    def integrate(self, nu, tf, dt=0.001, rho=None, theta=0.001, m=None, selfing=None, frozen=None):
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
        selfing: list of selfing probabilities, same length as nu.
        frozen: list of True and False same length as nu. True implies that a lineage
                is frozen (as in ancient samples). False integrates as normal.
        """
        num_pops = self.num_pops
        
        if tf == 0.0:
            return
        
        if rho is None and len(self) > 1:
            print('Rho not set, must specify.')
            return
        elif rho is not None and np.isscalar(rho) and len(self) != 2:
            print('Single rho passed but LD object has additional statistics.')
            return
        elif rho is not None and np.isscalar(rho) == False and len(rho) != len(self)-1:
            print('Mismatch of rhos passed and size of LD object.')
            return
        
        if rho is not None and np.isscalar(rho) == False and len(rho) == 1:
            rho = rho[0]
        
        if callable(nu):
            if len(nu(0)) != num_pops:
                raise ValueError("len of pop size function must equal number of pops.")
        else:
            if len(nu) != num_pops:
                raise ValueError("len of pop sizes must equal number of pops.")
        
        if m is not None and num_pops > 1:
            if np.shape(m) != (num_pops, num_pops):
                raise ValueError("migration matrix incorrectly defined for number of pops.")
        
        if frozen is not None:
            if len(frozen) != num_pops:
                raise ValueError("frozen must have same length as number of pops.")
        
        if selfing is not None:
            if len(selfing) != num_pops:
                raise ValueError("selfing must have same length as number of pops.")
        
        # enforce minimum 10 time steps per integration
        if tf < dt*10:
            dt_adj = tf / 10
        else:
            dt_adj = dt * 1.0
        
        self[:] = Numerics.integrate(self[:], nu, tf, dt=dt_adj,
                                    rho=rho, theta=theta, m=m,
                                    num_pops=num_pops, 
                                    selfing=selfing, frozen=frozen)

# Allow LDstats objects to be pickled.
try:
    import copy_reg
except:
    import copyreg
def LDstats_pickler(y):
    return LDstats_unpickler, (y, y.num_pops, y.pop_ids)
def LDstats_unpickler(data, num_pops, pop_ids):
    return LDstats(data, num_pops=num_pops, pop_ids=pop_ids)

try:
    copy_reg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
except:
    copyreg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
