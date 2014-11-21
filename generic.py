# Copyright (c) 2006, National ICT Australia
# All rights reserved.
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# Authors: Le Song (lesong@it.usyd.edu.au) and Alex Smola
# (alex.smola@nicta.com.au)
# Created: (20/10/2006)
# Last Updated: (dd/mm/yyyy)
#

##\package elefant.kernels.generic
# This module contains generic class for kernels
#
# The CKernel class provides common interface for all kernel classes. Note
# that it should never be instantiated.
#

__version__ = "$Revision: $" 
# $Source$ 

import numpy
import numpy.random as random

## Generic kernel class
#
# This kernel provide common interface for all kernels. This interface
# includes the following key kernel manipulations (functions):
# --Dot(x1, x2): $K(x1, x2)$
# --Expand(x1, x2, alpha): $sum_r K(x1_i,x2_r) \times alpha2_r$
# --Tensor(x1, y1, x2, y2): $K(x1_i,x2_j) \times (y1_i \times y1_j)$
# --TensorExpand(x1, y1, x2, y2, alpha2):
# $sum_r K(x1_i,x2_r) \times (y1_i \times y1_r) \times alpha2_r$
# --Remember(x): Remember data x
# --Forget(x): Remove remembered data x
# To design a specific kernel, simply overload these methods. The generic
# kernel itself should never be instantiated. 
#
class CKernel(object):
    def __init__(self, blocksize=128):
        ## @var _blocksize
        # Parameter that determines the size of each block when computing the
        # kernel matrix in blocks. Properly blocking the kernel matrix during
        # computation improves the speed.
        #
        self._blocksize = blocksize
        ## @var _name
        # Name of the kernel.
        #
        self._name = "Generic kernel"
        ## @var _cacheData
        # Cache that stores data that have appeared before.
        #
        self._cacheData = {}
        
    def __str__(self):
        return self._name
    
    def __repr__(self):
        return "Kernel object of type '" + self._name + "'"

    ## Compute the kernel between two data points x1 and x2.
    # It returns a scale value of dot product between x1 and x2.
    # @param x1 [read] The first data point.
    # @param x2 [read] The second data point.
    #
    def K(self, x1, x2):
        raise NotImplementedError, \
              'CKernel.K in abstract class is not implemented'
    
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
              'CKernel.Dot in abstract class is not implemented' 

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
        raise NotImplementedError, \
              'CKernel.Expand in abstract class is not implemented' 

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
        raise NotImplementedError, \
              'CKernel.Tensor in abstract class is not implemented' 

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
              'CKernel.TensorExpand in abstract class is not implemented'
    
    ## Remember the data by performing necessary preprossing on
    # the data, storing it in the cache and indexing it by the id of
    # the data. The preprocessing can be defined differently for
    # different classes. If the data have already been remembered,
    # the old stored information is simply overwritten.
    # @param x [read] The data to be remembered.
    #
    def Remember(self, x):
        raise NotImplementedError, \
              'CKernel.Remember in abstract class is not implemented' 

    ## Remove a remembered data from the cache. If x is not given, then
    # all the data remembered in the cache  will be removed. If a given
    # x is not remembered beforehand, False is returned; otherwise, True
    # is returned. 
    # @param x [read] The data to be removed.
    #
    def Forget(self, x=None):
        raise NotImplementedError, \
              'CKernel.Forget in abstract class is not implemented' 
   
