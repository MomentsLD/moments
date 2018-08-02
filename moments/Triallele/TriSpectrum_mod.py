"""
Contains triallele spectrum object
"""
import logging
logging.basicConfig()
logger = logging.getLogger('TriSpectrum_mod')

import os
import numpy, numpy as np
import moments.Triallele.Numerics
from moments.Triallele.Integration import integrate_cn

class TriSpectrum(numpy.ma.masked_array):
    """
    Represents a triallelic frequency spectrum.
    
    Spectra are represented ...
    
    The constructor has the format:
        fs = moments.Triallele.TriSpectrum(data, mask, mask_infeasible,
                                data_folded_major, data_folded_ancestral)
        
        data: The frequency spectrum data
        mask: An optional array of the same size as data. 'True' entries in this array
              are masked in the TriSpectrum. These represent missing data categories,
              or invalid entries in the array.
        data_folded_major: If True, it is assumed that the input data is folded 
                           for the major and minor derived alleles
        data_folded_ancestral: If True, it is assumed that the input data is folded
                               to account for uncertainty in the ancestral state. Note
                               that if True, data_folded_major must also be True.
        check_folding_major: If True and data_folded_ancestral=True, the data and
                             mask will be checked to ensure they are consistent
        check_folding_ancestral: If True and data_folded_ancestral=True, the data and
                                 mask will be checked to ensure they are consistent
    """
    def __new__(subtype, data, mask=numpy.ma.nomask, finite_genome=False, 
                mask_infeasible=True,
                mask_fixed = True,
                data_folded_major=None, check_folding_major=True,
                data_folded_ancestral=None, check_folding_ancestral=True,
                dtype=float, copy=True, fill_value=numpy.nan, keep_mask=True,
                shrink=True):
        if finite_genome == True:
            mask_fixed = False
            #print('If we simulate under the finite genome model, we do not mask fixed sites')
        data = numpy.asanyarray(data)
        
        if mask is numpy.ma.nomask:
            mask = numpy.ma.make_mask_none(data.shape)
        
        subarr = numpy.ma.masked_array(data, mask=mask, dtype=dtype, copy=copy,
                                       fill_value=fill_value, keep_mask=True, 
                                       shrink=True)
        subarr = subarr.view(subtype)
        if hasattr(data, 'folded_major'):
            if data_folded_major is None or data_folded_major == data.folded_major:
                subarr.folded_major = data.folded_major
            elif data_folded_major != data.folded_major:
                raise ValueError('Data does not have same major/minor folding status as '
                                 'was called for in Spectrum constructor.')
        elif data_folded_major is not None:
            subarr.folded_major = data_folded_major
        else:
            subarr.folded_major = False
        
        if hasattr(data, 'folded_ancestral'):
            if data_folded_ancestral is None or data_folded_ancestral == data.folded_ancestral:
                subarr.folded_ancestral = data.folded_ancestral
            elif data_folded_ancestral != data.folded_ancestral:
                raise ValueError('Data does not have same ancestral folding status as '
                                 'was called for in Spectrum constructor.')
        elif data_folded_ancestral is not None:
            subarr.folded_ancestral = data_folded_ancestral
        else:
            subarr.folded_ancestral = False
        
        if mask_infeasible:
            subarr.mask_infeasible()
        
        if mask_fixed:
            subarr.mask_fixed()
        
        return subarr
    
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        numpy.ma.masked_array.__array_finalize__(self, obj)
        self.folded_major = getattr(obj, 'folded_major', 'unspecified')
        self.folded_ancestral = getattr(obj, 'folded_ancestral', 'unspecified')
    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = numpy.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.folded_major = self.folded_major
        result.folded_ancestral = self.folded_ancestral
        return result
    def _update_from(self, obj):
        numpy.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'folded_major'):
            self.folded_major = obj.folded_major
        if hasattr(obj, 'folded_ancestral'):
            self.folded_ancestral = obj.folded_ancestral
    # masked_array has priority 15.
    __array_priority__ = 20
    
    def __repr__(self):
        return 'TriSpectrum(%s, folded_major=%s, folded_ancestral=%s)'\
                % (str(self), str(self.folded_major), str(self.folded_ancestral))
    
    def mask_infeasible(self):
        """
        Mask any infeasible entries.
        """
        for ii in range(len(self)):
            self.mask[ii,len(self)-ii:] = True
    
    def mask_fixed(self):
        """
        Mask entries that are not triallelic
        """
        self.mask[0,0] = True
        for ii in range(len(self)):
            self.mask[ii,len(self)-ii-1] = True
        
    def unfold(self):
        if not self.folded_major:
            raise ValueError('Input Spectrum is not folded.')
        data = self.data
        unfolded = TriSpectrum(data, mask_infeasible=True)
        return unfolded

    def _get_sample_size(self):
        return numpy.asarray(self.shape)[0] - 1
    sample_size = property(_get_sample_size)

    def _get_sample_sizes(self):
        return [self._get_sample_size()]
    sample_sizes = property(_get_sample_sizes)
    
    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, mask_infeasible=True, return_comments=False):
        """
        Read frequency spectrum from file.

        fid: string with file name to read from or an open file object.
        mask_infeasible: If True, mask the infeasible entries in the triallelic spectrum.
        return_comments: If true, the return value is (fs, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).

        See to_file method for details on the file format.
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

        # Read the shape of the data
        shape,folded_major,folded_ancestral = line.split()
        shape = [int(shape)+1,int(shape)+1]

        data = numpy.fromstring(fid.readline().strip(), 
                                count=numpy.product(shape), sep=' ')
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        maskline = fid.readline().strip()
        mask = numpy.fromstring(maskline, 
                                count=numpy.product(shape), sep=' ')
        mask = mask.reshape(*shape)
        
        if folded_major == 'folded_major':
            folded_major = True
        else:
            folded_major = False
        if folded_ancestral == 'folded_ancestral':
            folded_ancestral = True
        else:
            folded_ancestral = False

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        fs = TriSpectrum(data, mask, mask_infeasible, data_folded_ancestral=folded_ancestral,
                      data_folded_major=folded_major)
        if not return_comments:
            return fs
        else:
            return fs,comments
    
    def to_file(self, fid, precision=16, comment_lines=[], foldmaskinfo=True):
        """
        Write frequency spectrum to file.
    
        fid: string with file name to write to or an open file object.
        precision: precision with which to write out entries of the SFS. (They 
                   are formated via %.<p>g, where <p> is the precision.)
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
        foldmaskinfo: If False, folding and mask and population label
                      information will not be saved.

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing N integers giving the dimensions of the fs
              array. So this line would be '5 5 3' for an SFS that was 5x5x3.
              (That would be 4x4x2 *samples*.)
            On the *same line*, the string 'folded_major' or 'unfolded_major' 
              denoting the folding status of the array
            On the *same line*, the string 'folded_ancestral' or 'unfolded_ancestral' 
              denoting the folding status of the array
            A single line giving the array elements. The order of elements is 
              e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
            A single line giving the elements of the mask in the same order as
              the data line. '1' indicates masked, '0' indicates unmasked.
        """
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

        # Write out the shape of the fs
        fid.write('{0} '.format(self.sample_size))

        if foldmaskinfo:
            if not self.folded_major:
                fid.write('unfolded_major ')
            else:
                fid.write('folded_major ')
            if not self.folded_ancestral:
                fid.write('unfolded_ancestral ')
            else:
                fid.write('folded_ancestral ')
        
        fid.write(os.linesep)

        # Write the data to the file
        self.data.tofile(fid, ' ', '%%.%ig' % precision)
        fid.write(os.linesep)

        if foldmaskinfo:
            # Write the mask to the file
            numpy.asarray(self.mask,int).tofile(fid, ' ')
            fid.write(os.linesep)

        # Close file
        if newfile:
            fid.close()

    tofile = to_file
    
    def fold_major(self):
        if self.folded_major:
            raise ValueError('Input Spectrum is already folded.')
        folded = self + np.transpose(self)
        for ii in range(len(folded)):
            folded[ii,ii] /= 2
        folded.mask[0,:] = True
        folded.mask[:,0] = True
        for ii in range(len(folded)):
            folded[ii,ii+1:] = 0
            folded.mask[ii,ii+1:] = True
            folded.mask[ii,len(self)-1-ii:] = True
        
        folded.folded_major = True
        folded.folded_ancestral = self.folded_ancestral
        return folded

    def log(self):
        """
        Return the natural logarithm of the entries of the frequency spectrum.

        Only necessary because numpy.ma.log now fails to propagate extra
        attributes after numpy 1.10.
        """
        logfs = numpy.ma.log(self)
        logfs.folded_major = self.folded_major
        logfs.folded_ancestral = self.folded_ancestral
        return logfs
    
    def fold_ancestral(self):
        if self.folded_ancestral:
            raise ValueError('Input Spectrum is already folded.')
        folded = 0*self
        ns = len(folded)-1
        for ii in range(ns):
            for jj in range(ns):
                kk = ns-ii-jj
                if self.mask[ii,jj] == True:
                    continue
                elif ii <= kk and jj <= kk:
                    if ii >= jj:
                        folded[ii,jj] += self[ii,jj]
                    else:
                        folded[jj,ii] += self[ii,jj]
                elif ii > kk and jj <= kk:
                    folded[kk,jj] += self[ii,jj]
                elif ii <= kk and jj > kk:
                    folded[kk,ii] += self[ii,jj]
                else: # ii > kk and jj > kk
                    if ii >= jj:
                        folded[jj,kk] += self[ii,jj]
                    else:
                        folded[ii,kk] += self[ii,jj]
        # mask if not a valid entry for ancestrally folded spectrum
        for ii in range(ns):
            for jj in range(ns):
                kk = ns-ii-jj
                if not (kk>=ii>=jj):
                    folded.mask[ii,jj] = True
        
        folded.folded_major = True
        folded.folded_ancestral = True
        return folded
    
    def S(self):
        """
        Segregating sites
        """
        return self.sum()

    def project(self, ns, finite_genome=False):
        """
        Project to smaller sample size.
        ns: Sample size for new spectrum.
        """
        ### Take from numerics and put here - projection cache might stay in Numerics
        if finite_genome == False: # projection doesn't reach edges
            data = moments.Triallele.Numerics.project(self, ns)
            output = TriSpectrum(data, mask_infeasible=True)
            if self[-1,0].mask:
                output.mask_fixed()
            return output
        else: # all entries are projected, total sum of spectrum is maintained
#            output = moments.Triallele.Numerics.project_fg(self, ns)
#            if self.mask_infeasible == True:
#                output.mask_infeasible()
#            return output
            pass # still to implement
            
    
    def pi(self):
        """
        Estimated expected number of pairwise differences between two samples from the population
        at loci that are triallelic
        """
        pass
    
    # Ensures that when arithmetic is done with TriSpectrum objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, numpy.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = numpy.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    outfs = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   mask_infeasible=False, 
                                   data_folded_major=self.folded_major,
                                   data_folded_ancestral=self.folded_ancestral)
    return outfs
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, numpy.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = numpy.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    return self
""" % {'method':method})

    def _check_other_folding(self, other):
        """
        Ensure other Spectrum has same .folded status
        """
        if isinstance(other, self.__class__)\
           and (other.folded_major != self.folded_major
                or other.folded_ancestral != self.folded_ancestral):
            raise ValueError('Cannot operate with a folded Spectrum and an '
                             'unfolded one.')
    
    def integrate(self, nu, tf, dt=0.001, gammas=None, theta=1.0):
        """
        Method to simulate the triallelic fs forward in time.
        This integration scheme takes advantage of scipy's sparse methods.
        nu: population effective sizes as positive value or callable function
        tf: integration time in genetics units
        dt_fac: time step for integration
        gammas: Population size scaled selection coefficients [sAA, sA0, sBB, sB0, sAB]
                See documentation for definition and use
        theta: Population size scale mutation parameter
        """
        if gammas == None:
            gammas = [0,0,0,0,0]
        
        self.data[:] = integrate_cn(self.data, nu, tf, dt=dt, gammas=gammas, theta=theta)
        
        return self
    
# Allow TriSpectrum objects to be pickled. 
# See http://effbot.org/librarybook/copy-reg.htm
import copy_reg
def TriSpectrum_pickler(fs):
    # Collect all the info necessary to save the state of a TriSpectrum
    return TriSpectrum_unpickler, (fs.data, fs.mask, fs.folded_major,
                                   fs.folded_ancestral)
def TriSpectrum_unpickler(data, mask, folded_major, folded_ancestral):
    # Use that info to recreate the TriSpectrum
    return TriSpectrum(data, mask, mask_infeasible=False,
                       data_folded_major=folded_major,
                       data_folded_ancestral=folded_ancestral)
copy_reg.pickle(TriSpectrum, TriSpectrum_pickler, TriSpectrum_unpickler)
