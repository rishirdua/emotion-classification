# Copyright (c) 2006, National ICT Australia
# All rights reserved.
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the 'License'); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an 'AS IS' basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# Authors: Le Song (lesong@it.usyd.edu.au) and Alex Smola
# (alex.smola@nicta.com.au)
# Created: (20/10/2006)
# Last Updated: (dd/mm/yyyy)
#

##\package elefant.kernels.vector
# This module contains kernel classes for vectorial data.
#
# The CVectorKernel class provides common interface for all kernel classes
# for vectorial data. This kernel should never be instantiated as well.
# All other kernel classes are derived from the CVectorKernel class. The
# hierarchy for the the classes in this package is as follows:
# --CKernel (abstract)
# ------CVectorKernel (abstract)
# ----------CDeltaKernel
# ----------CLinearKernel
# ----------CDotProductKernel (abstract)
# --------------CPolynomialKernel
# --------------CTanhKernel
# --------------CExpKernel
# ----------CRBFKernel (abstract)
# --------------CGaussKernel
# --------------CLaplaceKernel
# --------------CInvDisKernel
# --------------CInvSqDisKernel
# --------------CBesselKernel
#

__version__ = '$Revision: $' 
# $Source$

import numpy
import numpy.random as random
from generic import CKernel
from my_exceptions import CElefantConstraintException

## Generic kernel class for vectorial data
#
# This kernel provide common interface for all kernels operating on
# vectorial data. This interface includes the following key kernel
# manipulations (functions):
# --Dot(x1, x2): $K(x1, x2)$
# --Expand(x1, x2, alpha): $sum_r K(x1_i,x2_r) \times alpha2_r$
# --Tensor(x1, y1, x2, y2): $K(x1_i,x2_j) \times (y1_i \times y1_j)$
# --TensorExpand(x1, y1, x2, y2, alpha2):
# $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$
# --Remember(x): Remember data x
# --Forget(x): Remove remembered data x
# To design a specific kernel, simply overload these methods. The generic
# kernel itself should never be instantiated, although methods
# Remember and Forget are implemented in this class. The Remember method
# stores the inner product of an input vector. 
#
class CVectorKernel(CKernel):
    def __init__( self, blocksize = 128 ):
        CKernel.__init__( self, blocksize )
        self._name = 'Vector kernel'
        ## @var _cacheKernel
        # Cache that store the base part for the kernel matrix.
        # This cache facilates the incremental and decremental
        # computational of the kernel matrix.
        #
        self._cacheKernel = {}
        ## @var _typicalParam
        # Typical parameter for the kernel. Many kernels do not
        # have parameters, such linear kernel. In this case, set
        # zero as the typical parameter. This variable will be
        # usefull when optimizing the kernel matrix with respect
        # to the kernel matrix.
        #
        self._typicalParam = numpy.array([0])

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #        
    def K(self, x1, x2):
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented' 

    ## Compute the kernel between the data points in x1 and those in x2.
    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points. 
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        raise NotImplementedError, \
              'CVectorKernel.Dot in abstract class is not implemented' 

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None):
        raise NotImplementedError, \
              'CVectorKernel.Expand in abstract class is not implemented' 

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2. It returns a matrix
    # with entry $(ij)$ equal to $K(x1_i,x2_j) \times (y1_i \times y1_j)$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #    
    def Tensor(self, x1, y1, x2, y2, index1=None, index2=None):
        raise NotImplementedError, \
              'CVectorKernel.Tensor in abstract class is not implemented' 

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2, and final multiply
    # the resulting matrix by alpha2. It returns a matrix with entry $(ij)$
    # equal to $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def TensorExpand(self, x1, y1, x2, y2, alpha2, index1=None, index2=None, \
                     output=None):
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented'
    
    ## Remember x by computing the inner product of the data points contained
    # in x, storing them in the cache and indexing them by the id of
    # x. If x have already been remembered,
    # the old stored information is simply overwritten.
    # @param x [read] The data to be remembered.
    #
    def Remember(self, x):
        # default behavior
        assert x is not None, 'x is None'
        assert len(x.shape) == 2, 'x is not a matrix'
        self._cacheData[id(x)] = (x**2).sum(axis=1)
        
    ## Remove the remembered data x. If x is not given, then all remembered
    # data in the cache is removed. If x has never been remembered, then
    # this function does nothing and return False instead. 
    # @param x [read] The data to be removed.
    #
    def Forget(self, x=None):
        # default behavior
        if x is not None:
            assert len(x.shape) == 2, 'Argument 1 is has wrong shape'
            if self._cacheData.has_key(id(x)) is False:
                return False
            else:
                del self._cacheData[id(x)]
        else:
            self._cacheData.clear()
            
        return True

    ## Method that operates on the base part x of a kernel.
    # The derived classes overload this method to generate new
    # kernels.
    # @param x Base part of the kernel.
    #
    def Kappa(self, x):
        # default behavior
        return x

    ## Gradient of the kernel with respect to the kernel
    # parameter evaluated in the base part x of a kernel.
    # The derived classes overload this method to generate new
    # gradients of the kernels.
    # @param x Base part of the kernel.
    #
    def KappaGrad(self, x):
        # default behavior
        return numpy.zeros(x.shape)

    ## Function that set the parameter of the kernel.
    # If the derived classes have parameters, overload this
    # method to set the parameters.
    # @param param Parameters to be set.
    #
    def SetParam(self, param):
        # default behavior        
        pass

    ## Clear the base part of the kernel computed for data x.
    # If x is not given, then all remembered data in the cache is removed.
    # If x has never been remembered, then this function does nothing
    # and return False instead.
    # @param x [read] The data whose base part is to be removed from the cache.
    #
    def ClearCacheKernel(self, x=None):
        # default behavior        
        if x is not None:
            assert len(x.shape) == 2, "Argument 1 has wrong shape"
            if self._cacheKernel.has_key(id(x)) is False:
                return False
            else:
                del self._cacheKernel[id(x)]
        else:
            self._cacheKernel.clear()

        return True

    ## Create the cache for the base part of the kernel computed for
    # data x, and index them by the id of x. If x have already been
    # remembered, the old stored information is simply overwritten.
    # Overload this method to store different base part for different
    # kernels.
    # @param x [read] The data whose base part is to be cached.
    #
    def CreateCacheKernel(self, x):
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented'

    ## Dot product of x with itself with the cached base part of the kernel.
    # Overload this method to use the base part differently for different
    # kernel. If param is given, the kernel matrix is computed using
    # the given parameter and the current base part. Otherwise, the old
    # parameters are used.
    # @param x The data set.
    # @param param The new parameters.
    # @param output The output buffer.
    #
    def DotCacheKernel(self, x, param=None, output=None):
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented'

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2. Overload this method to define the decrement of the base
    # part differently for different kernels. Note that this method
    # updates the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    #
    def DecCacheKernel(self, x1, x2):       
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented'

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2, and return the resulting kernel matrix. If param is given,
    # the kernel matrix is computed using the given parameter and the
    # current base part. Otherwise, the old parameters are used. Overload
    # this method to have different behavior for different kernel. Note
    # that this method does NOT change the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    # @param param The new parameters.
    #
    def DecDotCacheKernel(self, x1, x2, param=None, output=None):
        raise NotImplementedError, \
              'CVectorKernel.K in abstract class is not implemented'

    ## Gradient of the kernel matrix with respect to the kernel parameter.
    # If param is given, the kernel matrix is computed using the given
    # parameter and the  current base part. Otherwise, the old parameters
    # are used. Overload this method to have different behavior.
    # @param x The data set for the kernel matrix.
    # @param param The kernel parameters.
    #
    def GradDotCacheKernel(self, x, param=None, output=None):
        # default behavior        
        assert len(x.shape)==2, "Argument 1 has wrong shape"
        assert self._cacheKernel.has_key(id(x)) == True, \
               "Argument 1 has not been cached"
    
        if param is not None:
            self.SetParam(param)

        n = x.shape[0]
        output = numpy.zeros((n,n), numpy.float64)
        return output
        
#------------------------------------------------------------------------------
    
## Linear kernel class
#
# The methods are implemente efficiently by computing the resulting
# matrices block by block.
#
class CLinearKernel(CVectorKernel):
    def __init__(self, blocksize=128):
        CVectorKernel.__init__(self, blocksize)
        self._name = 'Linear kernel'

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        assert len(x1.squeeze().shape) == 1, 'x1 is not a vector'
        assert len(x2.squeeze().shape) == 1, 'x2 is not a vector'
        return (x1.squeeze()*x2.squeeze()).sum()

    ## Compute the kernel between the data points in x1 and those in x2.
    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points. 
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and Argument 2 have different dimensions'
        
        if index1 is not None:
            x1 = x1[index1,]
        if index2 is not None:
            x2 = x2[index2,]

        # number of data points in x1.
        n1 = x1.shape[0]
        # number of data points in x2.
        n2 = x2.shape[0]
        # number of blocks.
        nb = n1 / self._blocksize

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)
            
        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.dot(x1, numpy.transpose(x2))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0        
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
                
        return output    

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert len(alpha2.shape) == 2, 'Argument 3 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and 2 has different dimesions'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 2 and 3 has different number of data points'
        
        if index1 is not None:
            x1 = x1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            alpha2 = alpha2[index2,]

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]            

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)           

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.dot(numpy.dot(x1, numpy.transpose(x2)), alpha2)
                return output
            
        # blocking         
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.dot(numpy.transpose(numpy.dot(x2,numpy.transpose(x1[lower_limit:upper_limit,]))), alpha2)
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.dot(numpy.transpose(numpy.dot(x2,numpy.transpose(x1[lower_limit:n1,]))), alpha2)
            
        return output   

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2. It returns a matrix
    # with entry $(ij)$ equal to $K(x1_i,x2_j) \times (y1_i \times y1_j)$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Tensor(self, x1, y1, x2, y2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 has different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 2 and 3 has different dimensions'

        if index1 is not None:
            x1 = x1[index1,]
            y1 = y1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            y2 = y2[index2,]

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = x2.shape[0]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.transpose(y1[:,0]*numpy.transpose(y2[:,0]*numpy.dot(x1, numpy.transpose(x2))))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = y2[:,0]*numpy.transpose(y1[lower_limit:upper_limit,0]*numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = y2[:,0]*numpy.transpose(y1[lower_limit:n1,0]*numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
                
        return output            

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2, and final multiply
    # the resulting matrix by alpha2. It returns a matrix with entry $(ij)$
    # equal to $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def TensorExpand(self, x1, y1, x2, y2, alpha2, index1=None, index2=None, output=None):  
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 have different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 3 and 4 have different dimensions'
        assert len(alpha2.shape) == 2, 'Argument 5 has wrong shape'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 3 and 5 have different number of data points'
        
        if index1 is not None:
            x1 = x1[index1,]
            y1 = y1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            y2 = y2[index2,]
            alpha2 = alpha2[index2,]

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.transpose(y1[:,0] * numpy.transpose(numpy.dot(y2[:,0]*numpy.dot(x1,numpy.transpose(x2)), alpha2)))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.dot(y2[:,0]*numpy.transpose(y1[lower_limit:upper_limit,0] * numpy.dot(x2,numpy.transpose(x1[lower_limit:upper_limit,]))), alpha2)
            lower_limit = upper_limit            
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.dot(y2[:,0]*numpy.transpose(y1[lower_limit:n1,0] * numpy.dot(x2,numpy.transpose(x1[lower_limit:n1,]))), alpha2)

        return output

    ## Create the cache for the base part of the kernel computed for
    # data x, and index them by the id of x. If x have already been
    # remembered, the old stored information is simply overwritten.
    # @param x [read] The data whose base part is to be cached.
    #
    def CreateCacheKernel(self, x):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        n = x.shape[0]
        nb = n / self._blocksize

        # create the cache space
        if self._cacheKernel.has_key(id(x)):
            self.ClearCacheKernel(x)
        tmpCacheKernel = numpy.zeros((n,n), numpy.float64)
        self._cacheKernel[id(x)] = tmpCacheKernel

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            tmpCacheKernel[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n:
            tmpCacheKernel[lower_limit:n,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:n,])))

        return True

    ## Dot product of x with itself with the cached base part of the kernel.
    # If param is given, the kernel matrix is computed using
    # the given parameter and the current base part. Otherwise, the old
    # parameters are used.
    # @param x The data set.
    # @param param The new parameters.
    # @param output The output buffer.
    #
    def DotCacheKernel(self, x, param=None, output=None):
        assert len(x.shape)==2, 'Argument 1 has wrong shape'
        assert self._cacheKernel.has_key(id(x)) == True, \
               'Argument 1 has not been cached'
        
        n = x.shape[0]
        nb = n / self._blocksize  
        tmpCacheKernel = self._cacheKernel[id(x)]

        # set parameters.
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = tmpCacheKernel[lower_limit:upper_limit,]
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = tmpCacheKernel[lower_limit:n,]
            
        return output   

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2. Note that this method updates the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    #
    def DecCacheKernel(self, x1, x2):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'

        n = x1.shape[0]
        nb = n / self._blocksize
        tmpCacheKernel = self._cacheKernel[id(x1)]

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            tmpCacheKernel[lower_limit:upper_limit,] = tmpCacheKernel[lower_limit:upper_limit,] \
                                                      - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n:
            tmpCacheKernel[lower_limit:n,] = tmpCacheKernel[lower_limit:n,] \
                                                   - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))
            
        return True

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2, and return the resulting kernel matrix. If param is given,
    # the kernel matrix is computed using the given parameter and the
    # current base part. Otherwise, the old parameters are used. Note
    # that this method does NOT change the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    # @param param The new parameters.
    #
    def DecDotCacheKernel(self, x1, x2, param=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'
        
        n = x1.shape[0]
        nb = n / self._blocksize    
        tmpCacheKernel = self._cacheKernel[id(x1)]

        # set parameters.    
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = tmpCacheKernel[lower_limit:upper_limit,] \
                                            - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))              
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = tmpCacheKernel[lower_limit:n,] \
                                         - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))

        return output  

#------------------------------------------------------------------------------
    
## Dot Product kernel class
#
# All kernels that are function of the dot product between the two
# data points, ie. $K(x1,x2)=Kappa(x1^\top x2). 
# The methods are implemente efficiently by computing the resulting
# matrices block by block. All kernel classes derived from CDotProductKernel
# differ only in their choice of Kappa function.
#
class CDotProductKernel(CVectorKernel):
    def __init__(self, blocksize=128):
        CVectorKernel.__init__(self, blocksize)
        self._name = 'Dot Product kernel'

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        assert len(x1.squeeze().shape) == 1, 'x1 is not a vector'
        assert len(x2.squeeze().shape) == 1, 'x2 is not a vector'        
        return (x1.squeeze()*x2.squeeze()).sum()

    ## Compute the kernel between the data points in x1 and those in x2.
    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points. 
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and Argument 2 have different dimensions'
        
        if index1 is not None:
            x1 = x1[index1,]
        if index2 is not None:
            x2 = x2[index2,]

        # number of data points in x1.
        n1 = x1.shape[0]
        # number of data points in x2.
        n2 = x2.shape[0]
        # number of blocks.
        nb = n1 / self._blocksize

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = self.Kappa(numpy.dot(x1, numpy.transpose(x2)))
                return output

        # blocking        
        lower_limit = 0
        upper_limit = 0        
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = self.Kappa(numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,]))))
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = self.Kappa(numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,]))))
                
        return output    

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert len(alpha2.shape) == 2, 'Argument 3 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and 2 has different dimesions'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 2 and 3 has different number of data points'
        
        if index1 is not None:
            x1 = x1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            alpha2 = alpha2[index2,]

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is not None:
            output = output
        else:
            output = numpy.zeros((n1,n2), numpy.float64)           

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.dot(self.Kappa(numpy.dot(x1, numpy.transpose(x2))), alpha2)
                return output            

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.dot(self.Kappa(numpy.transpose(numpy.dot(x2,numpy.transpose(x1[lower_limit:upper_limit,])))), alpha2)
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.dot(self.Kappa(numpy.transpose(numpy.dot(x2,numpy.transpose(x1[lower_limit:n1,])))), alpha2)
            
        return output   

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2. It returns a matrix
    # with entry $(ij)$ equal to $K(x1_i,x2_j) \times (y1_i \times y1_j)$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Tensor(self, x1, y1, x2, y2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 has different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 2 and 3 has different dimensions'

        if index1 is not None:
            x1 = x1[index1,]
            y1 = y1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            y2 = y2[index2,]
            
        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = x2.shape[0]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.transpose(y1[:,0]*numpy.transpose(y2[:,0]*self.Kappa(numpy.dot(x1, numpy.transpose(x2)))))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = y2[:,0]*numpy.transpose(y1[lower_limit:upper_limit,0]*self.Kappa(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,]))))
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = y2[:,0]*numpy.transpose(y1[lower_limit:n1,0]*self.Kappa(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,]))))
                
        return output            

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2, and final multiply
    # the resulting matrix by alpha2. It returns a matrix with entry $(ij)$
    # equal to $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def TensorExpand(self, x1, y1, x2, y2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 have different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 3 and 4 have different dimensions'
        assert len(alpha2.shape) == 2, 'Argument 5 has wrong shape'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 3 and 5 have different number of data points'
        
        if index1 is not None:
            x1 = x1[index1,]
            y1 = y1[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            y2 = y2[index2,]
            alpha2 = alpha2[index2,]
    
        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.transpose(y1[:,0] * numpy.transpose(numpy.dot(y2[:,0]*self.Kappa(numpy.dot(x1,numpy.transpose(x2))), alpha2)))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.dot(y2[:,0]*numpy.transpose(y1[lower_limit:upper_limit,0] * self.Kappa(numpy.dot(x2,numpy.transpose(x1[lower_limit:upper_limit,])))), alpha2)
            lower_limit = upper_limit            
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.dot(y2[:,0]*numpy.transpose(y1[lower_limit:n1,0] * self.Kappa(numpy.dot(x2,numpy.transpose(x1[lower_limit:n1,])))), alpha2)

        return output

    ## Create the cache for the base part of the kernel computed for
    # data x, and index them by the id of x. If x have already been
    # remembered, the old stored information is simply overwritten.
    # @param x [read] The data whose base part is to be cached.
    #
    def CreateCacheKernel(self, x):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        n = x.shape[0]
        nb = n / self._blocksize

        # create the cache space
        if self._cacheKernel.has_key(id(x)):
            self.ClearCacheKernel(x)
        tmpCacheKernel = numpy.zeros((n,n), numpy.float64)
        self._cacheKernel[id(x)] = tmpCacheKernel

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            tmpCacheKernel[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n:
            tmpCacheKernel[lower_limit:n,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:n,])))

        return True

    ## Dot product of x with itself with the cached base part of the kernel.
    # If param is given, the kernel matrix is computed using
    # the given parameter and the current base part. Otherwise, the old
    # parameters are used.
    # @param x The data set.
    # @param param The new parameters.
    # @param output The output buffer.
    #
    def DotCacheKernel(self, x, param=None, output=None):
        assert len(x.shape)==2, 'Argument 1 has wrong shape'
        assert self._cacheKernel.has_key(id(x)) == True, \
               'Argument 1 has not been cached'
        
        n = x.shape[0]
        nb = n / self._blocksize  
        tmpCacheKernel = self._cacheKernel[id(x)]

        # set parameters.
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = self.Kappa(tmpCacheKernel[lower_limit:upper_limit,])
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = self.Kappa(tmpCacheKernel[lower_limit:n,])

        return output

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2. Note that this method updates the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    #
    def DecCacheKernel(self, x1, x2):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'

        n = x1.shape[0]
        nb = n / self._blocksize
        tmpCacheKernel = self._cacheKernel[id(x1)]

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            tmpCacheKernel[lower_limit:upper_limit,] = tmpCacheKernel[lower_limit:upper_limit,] \
                                                      - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))
            lower_limit = upper_limit
        if lower_limit <= n:
            tmpCacheKernel[lower_limit:n,] = tmpCacheKernel[lower_limit:n,] \
                                                   - numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))
            
        return True

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2, and return the resulting kernel matrix. If param is given,
    # the kernel matrix is computed using the given parameter and the
    # current base part. Otherwise, the old parameters are used. Note
    # that this method does NOT change the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    # @param param The new parameters.
    #
    def DecDotCacheKernel(self, x1, x2, param=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'
        
        n = x1.shape[0]
        nb = n / self._blocksize    
        tmpCacheKernel = self._cacheKernel[id(x1)]

        # set parameters.    
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))
            output[lower_limit:upper_limit,] = self.Kappa(tmpCacheKernel[lower_limit:upper_limit,] \
                                                       - output[lower_limit:upper_limit,])                                                       
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))
            output[lower_limit:n,] = self.Kappa(tmpCacheKernel[lower_limit:n,] \
                                                    - output[lower_limit:n,])

        return output   

    ## Gradient of the kernel matrix with respect to the kernel parameter.
    # If param is given, the kernel matrix is computed using the given
    # parameter and the  current base part. Otherwise, the old parameters
    # are used.
    # @param x The data set for the kernel matrix.
    # @param param The kernel parameters.
    #
    def GradDotCacheKernel(self, x, param=None, output=None):
        assert len(x.shape) == 2, "Argument 1 has wrong shape"
        assert self._cacheKernel.has_key(id(x)) == True, \
               "Argument 1 has not been cached"

        n = x.shape[0]
        nb = n / self._blocksize
        tmpCacheKernel = self._cacheKernel[id(x)]

        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = self.KappaGrad(tmpCacheKernel[lower_limit:upper_limit,])
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = self.KappaGrad(tmpCacheKernel[lower_limit:n,])

        return output
    
## Polynomial kernel
#
# Kernel of the form: $K(x1,x2)=(scale (x1^\top x2) + offset)^degree$.
# This is implemented by the Kappa function.
#
class CPolynomialKernel(CDotProductKernel):
    def __init__(self, degree=2, offset=1.0, scale=1.0, blocksize=128):
        CDotProductKernel.__init__(self, blocksize)
        self._name = 'Polynomial kernel'
        self._scale = scale
        self._offset = offset
        self._degree = degree

    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return (self._scale * x + self._offset) ** self._degree 

## Tanh kernel
#
# Kernel of the form : $K(x1,x2)=\tanh(scale (x1^\top x2) + offset)$.
# This is implemented by the Kappa function.
#
class CTanhKernel(CDotProductKernel):
    def __init__(self, offset=1.0, scale=1.0, blocksize=128):
        CDotProductKernel.__init__(self, blocksize)
        self._name = 'Tahn kernel'
        self._scale = scale
        self._offset = offset

    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return numpy.tanh(self._scale * x + self._offset)

## Exponential kernel
#
# Kernel of the form: $K(x1,x2)=\exp(scale (x1^\top x2) + offset)
# This is implemented by the Kappa function.
#
class CExpKernel(CDotProductKernel):
    def __init__(self, offset=0, scale=1.0, blocksize=128):
        CDotProductKernel.__init__(self, blocksize)
        self._name = 'Expential kernel'
        self._scale = scale
        self._offset = offset

    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #        
    def Kappa(self, x):
        return numpy.exp(self._scale * x + self._offset)

#------------------------------------------------------------------------------
    
## RBF kernels. 
# 
# Distance-based kernel, where Gauss, Laplace and Bessel kernels are derived.
# This class of kernels has the form: $K(x1,x2)=f(\|x1-x2\|)$
# The methods are implemente efficiently by computing the resulting
# matrices block by block. All kernel classes derived from CRBFKernel
# differ only in their choice of Kappa function.
#
class CRBFKernel(CVectorKernel):
    def __init__(self, blocksize=128):
        CVectorKernel.__init__(self, blocksize=blocksize)
        self._name = 'RBF kernel'

    ## Method that operates on the base part of a kernel.
    # The derived classes overload this method to generate new
    # kernels.
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return x

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        ##assert len(x1.squeeze().shape) == 1, 'x1 is not a vector'
        ##assert len(x2.squeeze().shape) == 1, 'x2 is not a vector'
        
        v = (x1 - x2).squeeze()
        return self.Kappa(numpy.sum(v*v))

    ## Compute the square distance using inner products. It computes
    # the distance by following formular:
    # $(x1-x2)^2 = x1^2 - 2 x1 x2 + x2^2$
    # @param dest [write] The destination where the results are stored.
    # @param x1_x2 [read] The inner product matrix between x1 and x2.
    # @param dot_x1 [read] The inner product of the data in x1.
    # @param dot_x2 [read] The inner product of the data in x2.
    #
    def KappaSqDis(self, dest, x1_x2, dot_x1, dot_x2):
        dest[:,] = self.Kappa(- 2.0 * x1_x2 + numpy.add.outer(dot_x1, dot_x2))
        
    ## Compute the kernel between the data points in x1 and those in x2.
    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points. 
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and Argument 2 have different dimensions'

        # retrieve remembered data from the cache.        
        flg1 = False
        if self._cacheData.has_key(id(x1)):
            dot_x1 = self._cacheData[id(x1)]
            flg1 = True
        flg2 = False
        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
            flg2 = True

        if index1 is not None:
            x1 = x1[index1,]
            if flg1 is True:
                dot_x1 = dot_x1[index1,]
            else:
                dot_x1 = numpy.sum(x1**2, 1)
        else:
            if flg1 is False:
                dot_x1 = numpy.sum(x1**2, 1)
                
        if index2 is not None:
            x2 = x2[index2,]            
            if flg2 is True:
                dot_x2 = dot_x2[index2,]
            else:
                dot_x2 = numpy.sum(x2**2, 1)                
        else:
            if flg2 is False:
                dot_x2 = numpy.sum(x2**2, 1)                

        # number of data points in x1.
        n1 = x1.shape[0]
        # number of data points in x2.
        n2 = x2.shape[0]
        # number of blocks.
        nb = n1 / self._blocksize

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)
            
        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.dot(x1, numpy.transpose(x2))
                self.KappaSqDis(output, output, dot_x1, dot_x2)
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            self.KappaSqDis(output[lower_limit:upper_limit,], output[lower_limit:upper_limit,], dot_x1[lower_limit:upper_limit], dot_x2)           
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
            self.KappaSqDis(output[lower_limit:n1,], output[lower_limit:n1,], dot_x1[lower_limit:n1], dot_x2)
                
        return output  

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert len(alpha2.shape) == 2, 'Argument 3 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and 2 has different dimesions'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 2 and 3 has different number of data points'

        # retrieve remembered data from cache.
        flg1 = False
        if self._cacheData.has_key(id(x1)):
            dot_x1 = self._cacheData[id(x1)]
            flg1 = True
        flg2 = False
        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
            flg2 = True

        if index1 is not None:
            x1 = x1[index1,]
            if flg1 is True:
                dot_x1 = dot_x1[index1,]
            else:
                dot_x1 = numpy.sum(x1**2, 1)
        else:
            if flg1 is False:
                dot_x1 = numpy.sum(x1**2, 1)
        if index2 is not None:
            x2 = x2[index2,]
            if flg2 is True:
                dot_x2 = dot_x2[index2,]
            else:
                dot_x2 = numpy.sum(x2**2, 1)
            alpha2 = alpha2[index2,]
        else:
            if flg2 is False:
                dot_x2 = numpy.sum(x2**2, 1)

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases.
        if index2 is not None:
            if len(index2) <= self._blocksize:
                x1_x2 = numpy.dot(x1, numpy.transpose(x2))
                self.KappaSqDis(x1_x2, x1_x2, dot_x1, dot_x2)
                output = numpy.dot(x1_x2,alpha2)
                return output

        # blocking.
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:upper_limit], dot_x2)
            output[lower_limit:upper_limit,] = numpy.dot(x1_x2, alpha2)
            lower_limit = upper_limit           
        if lower_limit <= n1:
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:n1], dot_x2)
            output[lower_limit:n1,] = numpy.dot(x1_x2, alpha2)
            
        return output   

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2. It returns a matrix
    # with entry $(ij)$ equal to $K(x1_i,x2_j) \times (y1_i \times y1_j)$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Tensor(self, x1, y1, x2, y2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 has different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 2 and 3 has different dimensions'

        # retrieve remembered data from the cache
        flg1 = False
        if self._cacheData.has_key(id(x1)):
            dot_x1 = self._cacheData[id(x1)]
            flg1 = True
        flg2 = False
        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
            flg2 = True

        if index1 is not None:
            x1 = x1[index1,]
            if flg1 is True:
                dot_x1 = dot_x1[index1,]
            else:
                dot_x1 = numpy.sum(x1**2, 1)
            y1 = y1[index1,]
        else:
            if flg1 is False:
                dot_x1 = numpy.sum(x1**2, 1)
        if index2 is not None:
            x2 = x2[index2,]
            if flg2 is True:
                dot_x2 = dot_x2[index2,]
            else:
                dot_x2 = numpy.sum(x2**2, 1)
            y2 = y2[index2,]
        else:
            if flg2 is False:
                dot_x2 = numpy.sum(x2**2, 1)
        
        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = x2.shape[0]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                x1_x2 = numpy.dot(x1, numpy.transpose(x2))
                self.KappaSqDis(x1_x2, x1_x2, dot_x1, dot_x2)
                output = numpy.transpose(y1[:,0] * numpy.transpose(y2[:,0] * x1_x2))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:upper_limit], dot_x2)
            output[lower_limit:upper_limit,] = numpy.transpose(y1[lower_limit:upper_limit,0] * numpy.transpose(y2[:,0] * x1_x2))
            lower_limit = upper_limit
        if lower_limit <= n1:
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:n1], dot_x2)
            output[lower_limit:n1,] = numpy.transpose(y1[lower_limit:n1,0] * numpy.transpose(y2[:,0] * x1_x2))

        return output

    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix elementwiesely by the
    # the outer-product matrix between y1 and y2, and final multiply
    # the resulting matrix by alpha2. It returns a matrix with entry $(ij)$
    # equal to $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param y1 [read] The first set of labels.
    # @param x2 [read] The second set of data points.
    # @param y2 [read] The second set of labels.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def TensorExpand(self, x1, y1, x2, y2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y1.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == y1.shape[0], \
               'Argument 1 and 2 have different dimensions'
        assert len(x2.shape) == 2, 'Argument 3 has wrong shape'
        assert len(y2.shape) == 2, 'Argument 4 has wrong shape'
        assert x2.shape[0] == y2.shape[0], \
               'Argument 3 and 4 have different dimensions'
        assert len(alpha2.shape) == 2, 'Argument 5 has wrong shape'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 3 and 5 have different number of data points'

        # retrieve remembered data from the cache
        flg1 = False
        if self._cacheData.has_key(id(x1)):
            dot_x1 = self._cacheData[id(x1)]
            flg1 = True
        flg2 = False
        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
            flg2 = True

        if index1 is not None:
            x1 = x1[index1,]
            if flg1 is True:
                dot_x1 = dot_x1[index1,]
            else:
                dot_x1 = numpy.sum(x1**2, 1)
            y1 = y1[index1,]
        else:
            if flg1 is False:
                dot_x1 = numpy.sum(x1**2, 1)
                
        if index2 is not None:
            x2 = x2[index2,]
            if flg2 is True:
                dot_x2 = dot_x2[index2,]
            else:
                dot_x2 = numpy.sum(x2**2, 1)
            y2 = y2[index2,]
            alpha2 = alpha2[index2,]
        else:
            if flg2 is False:
                dot_x2 = numpy.sum(x2**2, 1)

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                x1_x2 = numpy.dot(x1, numpy.transpose(x2))
                self.KappaSqDis(x1_x2, x1_x2, dot_x1, dot_x2)
                output = numpy.transpose(y1[:,0] * numpy.transpose(numpy.dot(y2[:,0]*x1_x2, alpha2)))
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:upper_limit,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:upper_limit], dot_x2)
            output[lower_limit:upper_limit,] = numpy.transpose(y1[lower_limit:upper_limit,0] * numpy.transpose(numpy.dot(y2[:,0]*x1_x2, alpha2)))
            lower_limit = upper_limit
        if lower_limit <= n1:
            x1_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x1[lower_limit:n1,])))
            self.KappaSqDis(x1_x2, x1_x2, dot_x1[lower_limit:n1], dot_x2)
            output[lower_limit:n1,] = numpy.transpose(y1[lower_limit:n1,0] * numpy.transpose(numpy.dot(y2[:,0]*x1_x2, alpha2)))

        return output

    ## Create the cache for the base part of the kernel computed for
    # data x, and index them by the id of x. If x have already been
    # remembered, the old stored information is simply overwritten.
    # @param x [read] The data whose base part is to be cached.
    #
    def CreateCacheKernel(self, x):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        n = x.shape[0]
        nb = n / self._blocksize

        # create the cache space
        if self._cacheKernel.has_key(id(x)):
            self.ClearCacheKernel(x)
        tmpCacheKernel = numpy.zeros((n,n), numpy.float64)
        self._cacheKernel[id(x)] = tmpCacheKernel

        if self._cacheData.has_key(id(x)):
            dot_x = self._cacheData[id(x)]
        else:
            dot_x = numpy.sum(x**2,1)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            tmpCacheKernel[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:upper_limit,])))
            tmpCacheKernel[lower_limit:upper_limit,] = numpy.add.outer(dot_x[lower_limit:upper_limit], dot_x) \
                                                     - 2*tmpCacheKernel[lower_limit:upper_limit,]
            lower_limit = upper_limit
        if lower_limit <= n:
            tmpCacheKernel[lower_limit:n,] = numpy.transpose(numpy.dot(x, numpy.transpose(x[lower_limit:n,])))
            tmpCacheKernel[lower_limit:n,] = numpy.add.outer(dot_x[lower_limit:n], dot_x) \
                                                  - 2*tmpCacheKernel[lower_limit:n,]

        return True

    ## Dot product of x with itself with the cached base part of the kernel.
    # If param is given, the kernel matrix is computed using
    # the given parameter and the current base part. Otherwise, the old
    # parameters are used.
    # @param x The data set.
    # @param param The new parameters.
    # @param output The output buffer.
    #
    def DotCacheKernel(self, x, param=None, output=None):
        assert len(x.shape)==2, 'Argument 1 has wrong shape'
        assert self._cacheKernel.has_key(id(x)) == True, \
               'Argument 1 has not been cached'
        
        n = x.shape[0]
        nb = n / self._blocksize  
        tmpCacheKernel = self._cacheKernel[id(x)]

        # set parameters.
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = self.Kappa(tmpCacheKernel[lower_limit:upper_limit,])
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = self.Kappa(tmpCacheKernel[lower_limit:n,])

        return output

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2. Note that this method updates the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    #
    def DecCacheKernel(self, x1, x2):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'

        n = x1.shape[0]
        nb = n / self._blocksize
        tmpCacheKernel = self._cacheKernel[id(x1)]

        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
        else:
            dot_x2 = numpy.sum(x2**2,1) 

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            x2_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))
            tmpCacheKernel[lower_limit:upper_limit,] = tmpCacheKernel[lower_limit:upper_limit,] \
                                                     + 2*x2_x2 \
                                                     - numpy.add.outer(dot_x2[lower_limit:upper_limit], dot_x2)
            lower_limit = upper_limit
        if lower_limit <= n:
            x2_x2 = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))
            tmpCacheKernel[lower_limit:n,] = tmpCacheKernel[lower_limit:n,] \
                                                  + 2*x2_x2 \
                                                  - numpy.add.outer(dot_x2[lower_limit:n], dot_x2)       

        return True

    ## Decrement the base part of the kernel for x1 stored in the cache
    # by x2, and return the resulting kernel matrix. If param is given,
    # the kernel matrix is computed using the given parameter and the
    # current base part. Otherwise, the old parameters are used. Note
    # that this method does NOT change the cache for the kernel part.
    # @param x1 The data set whose base part has been cached.
    # @param x2 The data set who is to be decremented from x1.
    # @param param The new parameters.
    #
    def DecDotCacheKernel(self, x1, x2, param=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and 2 have different number of data points'
        assert self._cacheKernel.has_key(id(x1)) == True, \
               'Argument 1 has not been cached'
        
        n = x1.shape[0]
        nb = n / self._blocksize    
        tmpCacheKernel = self._cacheKernel[id(x1)]

        # set parameters.    
        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        if self._cacheData.has_key(id(x2)):
            dot_x2 = self._cacheData[id(x2)]
        else:
            dot_x2 = numpy.sum(x2*x2,1)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:upper_limit,])))
            output[lower_limit:upper_limit,] = self.Kappa(tmpCacheKernel[lower_limit:upper_limit,] \
                                                       + 2*output[lower_limit:upper_limit,] \
                                                       - numpy.add.outer(dot_x2[lower_limit:upper_limit], dot_x2))
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = numpy.transpose(numpy.dot(x2, numpy.transpose(x2[lower_limit:n,])))
            output[lower_limit:n,] = self.Kappa(tmpCacheKernel[lower_limit:n,] \
                                                    + 2*output[lower_limit:n,] \
                                                    - numpy.add.outer(dot_x2[lower_limit:n], dot_x2))

        return output   

    ## Gradient of the kernel matrix with respect to the kernel parameter.
    # If param is given, the kernel matrix is computed using the given
    # parameter and the  current base part. Otherwise, the old parameters
    # are used.
    # @param x The data set for the kernel matrix.
    # @param param The kernel parameters.
    #
    def GradDotCacheKernel(self, x, param=None, output=None):
        assert len(x.shape) == 2, "Argument 1 has wrong shape"
        assert self._cacheKernel.has_key(id(x)) == True, \
               "Argument 1 has not been cached"

        n = x.shape[0]
        nb = n / self._blocksize
        tmpCacheKernel = self._cacheKernel[id(x)]

        if param is not None:
            self.SetParam(param)

        if output is None:
            output = numpy.zeros((n,n), numpy.float64)

        # blocking
        lower_limit = 0
        upper_limit = 0
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = self.KappaGrad(tmpCacheKernel[lower_limit:upper_limit,])
            lower_limit = upper_limit
        if lower_limit <= n:
            output[lower_limit:n,] = self.KappaGrad(tmpCacheKernel[lower_limit:n,])

        return output
    

## Gauss kernel
#
# Kernel of the form: $K(x1,x2)=\exp(-scale*\|x1-x2\|^2)$.
# This is implemented by the Kappa function.
#
class CGaussKernel(CRBFKernel):
    def __init__(self, scale=1.0, blocksize=128):
        CRBFKernel.__init__(self, blocksize)
        self._name = 'Gauss kernel'
        if scale < 0.0:
            raise CElefantConstraintException(scale, "param must not be negative")        
        self._scale = scale
        self._typicalParam = 10**(numpy.arange(10)-7.0)       

    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return numpy.exp(-self._scale * x)

    ## Gradient of the kernel with respect to the kernel
    # parameter evaluated at the base part x of the kernel.
    # @param x Base part of the kernel.
    #
    def KappaGrad(self, x):
        return -x * numpy.exp(-self._scale * x)

    ## Function that set the parameter of the kernel.
    # @param param Parameters to be set.
    #
    def SetParam(self, param):
        if param < 0.0:
            raise CElefantConstraintException(param, "param must not be negative")
        self._scale = param

## Laplace Kernel
#
# Kernel of the form: $K(x1,x2)=\exp(-scale*\|x1-x2\|)$.
# This is implemented by the Kappa function.
#
class CLaplaceKernel(CRBFKernel):
    def __init__(self, scale=1.0, blocksize=128):
        CRBFKernel.__init__(self, blocksize)
        self._name = 'Laplace kernel'
        if scale < 0.0:
            raise CElefantConstraintException(scale, "param must not be negative")
        self._scale = scale
        self._typicalParam = 10**(numpy.arange(10)-7.0)         
        
    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return numpy.exp(-self._scale * numpy.sqrt(x.clip(min = 0.0, max = numpy.inf)))

    ## Gradient of the kernel with respect to the kernel
    # parameter evaluated at the base part x of the kernel.
    # @param x Base part of the kernel.
    #
    def KappaGrad(self, x):
        tmp = numpy.sqrt(x.clip(min = 0.0, max = numpy.inf))
        return -tmp * numpy.exp(-self._scale * tmp)

    ## Function that set the parameter of the kernel.
    # @param param Parameters to be set.
    #        
    def SetParam(self, param):
        #  Enforce nonnegativity of parameter
        # assert param >= 0.0, 'param must not be negative'
        if param < 0.0:
            raise CElefantConstraintException(param, "param must not be negative")
        self._scale = param

## Inverse square distance kernel
#
# Kernel of the form: $K(x1,x2)=\frac{1}{\|x1-x2\|^2}
# This is implemented by the Kappa function. Note that
# if the distance between x1 and x2 is very small, the
# kernel is computed as zero.
#
class CInvSqDisKernel(CRBFKernel):
    def __init__(self, blocksize=128, epsilon = 1.0e-6):
        CRBFKernel.__init__(self, blocksize)
        self._name = 'Inverse Square Distance kernel'
        self._epsilon = epsilon
        
        
    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return 1.0 / (x + self._epsilon)
    
## Inverse distance kernel
#
# Kernel of the form: $K(x1,x2)=\frac{1}{\|x1-x2\|}
# This is implemented by the Kappa function. Note that
# if the distance between x1 and x2 is very small, the
# kernel is computed as zero.
#
class CInvDisKernel(CRBFKernel):
    def __init__(self, blocksize=128, epsilon = 1.0e-6):
        CRBFKernel.__init__(self, blocksize)
        self._name = 'Inverse Distance kernel'
        self._epsilon = epsilon

    ## Method that operates on the base part of the kernel to
    # generate the new kernel
    # @param x Base part of the kernel matrix.
    #
    def Kappa(self, x):
        return 1.0/numpy.sqrt(x + self._epsilon)

#------------------------------------------------------------------------------
    
## Delta kernel class
#
# The methods are implemente efficiently by computing the resulting
# matrices block by block. Currently this class only works for the
# case where the inputs are integer.
#
class CDeltaKernel(CVectorKernel):
    def __init__(self, blocksize=128):
        CVectorKernel.__init__(self, blocksize)
        self._name = 'Delta kernel'

    ## Compute the kernel between two data points x1 and x2.
    # It returns a value 1 if x1 and x2 are equal, otherwise 0.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        assert len(x1.squeeze().shape) == 1, 'x1 is not a vector'
        assert len(x2.squeeze().shape) == 1, 'x2 is not a vector'
        return numpy.equal.outer(x1.squeeze(), x2.squeeze()).astype(numpy.int)

    ## Compute the kernel between the data points in x1 and those in x2.
    # It returns a matrix with entry $(ij)$ equal to $K(x1_i, x1_j)$.
    # If index1/index2 is
    # specified, only those data points in x1/x2 with indices corresponding
    # to index1/index2 are used to compute the kernel matrix. Furthermore,
    # if output is specified, the provided buffer is used explicitly to
    # store the kernel matrix.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param index1 [read] The indices into the first set of data points. 
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Dot(self, x1, x2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert x1.shape[0] == x2.shape[0], \
               'Argument 1 and Argument 2 have different dimensions'
        
        if index1 is not None:
            x1 = x1[index1,]
        if index2 is not None:
            x2 = x2[index2,]

        # number of data points in x1.
        n1 = x1.shape[0]
        # number of data points in x2.
        n2 = x2.shape[0]
        # number of blocks.
        nb = n1 / self._blocksize

        if output is None:
            output = numpy.zeros((n1,n2), numpy.int)
            
        # handle special cases:
        if index2 is not None:
            if len(index2) <= self._blocksize:
                output = numpy.equal.outer(x1.squeeze(), x2.squeeze()).astype(numpy.int)
                return output

        # blocking
        lower_limit = 0
        upper_limit = 0        
        for i in range(nb):
            upper_limit = upper_limit + self._blocksize
            output[lower_limit:upper_limit,] = numpy.equal.outer(x1[lower_limit:upper_limit,].squeeze(), x2.squeeze()).astype(numpy.int)
            lower_limit = upper_limit
        if lower_limit <= n1:
            output[lower_limit:n1,] = numpy.equal.outer(x1[lower_limit:n1,].squeeze(), x2.squeeze()).astype(numpy.int)
                
        return output    
    
#------------------------------------------------------------------------------
    
## Joint kernel class for k(x,x')l(y,y')
#
# The methods are implemente efficiently by computing the resulting
# matrices block by block. Currently this class only works for the
# case where the inputs are integer.
#
class CJointKernel(CVectorKernel):
    def __init__(self, y, xkernel=CLinearKernel(), ykernel=CDeltaKernel(), blocksize=128):
        CVectorKernel.__init__(self, blocksize)
        self._name = 'Joint kernel'
        self.xkernel = xkernel
        self.ykernel = ykernel
        # remember the kernel matrix on y.
        assert y is not None, 'y has be to provided for the initialization'
        self.kyy = self.ykernel.Dot(y,y)
            
    ## Compute the kernel between the data points in x1 and those in x2,
    # then multiply the resulting kernel matrix by alpha2.
    # It returns a matrix with entry $(ij)$ equal to
    # $sum_r K(x1_i,x2_r) \times alpha2_r$.
    # Other parameters are defined similarly as those in Dot.
    # @param x1 [read] The first set of data points.
    # @param x2 [read] The second set of data points.
    # @param alpha2 [read] The set of coefficients.
    # @param index1 [read] The indices into the first set of data points.
    # @param index2 [read] The indices into the second set of data points.
    # @param output [write] The buffer where the output matrix is written into.
    #
    def Expand(self, x1, x2, alpha2, index1=None, index2=None, output=None):
        assert len(x1.shape) == 2, 'Argument 1 has wrong shape'
        assert len(x2.shape) == 2, 'Argument 2 has wrong shape'
        assert len(alpha2.shape) == 2, 'Argument 3 has wrong shape'
        assert x1.shape[1] == x2.shape[1], \
               'Argument 1 and 2 has different dimesions'
        assert x2.shape[0] == alpha2.shape[0], \
               'Argument 2 and 3 has different number of data points'

        L = self.kyy
        if index1 is not None:
            x1 = x1[index1,]
            L = L[index1,]
        if index2 is not None:
            x2 = x2[index2,]
            alpha2 = alpha2[index2,]
            L = L[:,index2]        

        n1 = x1.shape[0]
        nb = n1 / self._blocksize
        n2 = alpha2.shape[1]

        if output is None:
            output = numpy.zeros((n1,n2), numpy.float64)

        output = numpy.transpose(numpy.dot(L, self.xkernel.Expand(x1, x2, alpha2)))
        return output

if __name__== '__main__':
    n = 5000
    x = numpy.random.rand(n,1)
    y = (numpy.random.rand(n,1)*10).astype(numpy.int)
    alpha = numpy.random.rand(n,4)
    import time
    t1 = time.clock()
    kernel = CJointKernel(y)
    K = kernel.Expand(x, x, alpha)
    t2 = time.clock()
    print K.shape, t2-t1
