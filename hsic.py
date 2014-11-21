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
# Authors: Le Song (lesong@it.usyd.edu.au)
# Created: (20/10/2006)
# Last Updated: (dd/mm/yyyy)
#

##\package elefant.fselection.hsic
# This module perform computation related to Hilber-Schmidt Independence
# Criterion. Hilber-Schmidt Independence Criterion is short for HSIC.
#
# HISC is defined as $HSIC=\frac{1}{m}Tr(KHLH)$, where $kMat$ and $lMat$
# are the kernel matrices for the data and the labels respectively.
# $H=I-\frac{1}{m}\delta_{ij}$, where $m$ is the number of data points,
# is the centering matrix. The unbiased estimator of HSIC is computed as
# $HSIC=\frac{1}{m(m-3)}\left[Tr(KL)+\frac{1}{(m-1)(m-2)}1^\top K11^\top L1
# -\frac{2}{m-2}1^\top KL1\right]. For more theorectical underpinning
# of HSIC, see the following reference:
#
# Gretton, A., O. Bousquet, A. Smola and B. Schoelkopf: Measuring
# Statistical Dependence with Hilbert-Schmidt Norms. Algorithmic
# Learning Theory: 16th International Conference, ALT 2005, 63-78, 2005.
# 

__version__ = '$Revision: $' 
# $Source$

import numpy
import vector
from setdiag0 import setdiag0

## Class that perform computation related to HSIC.
#
# It contains function that computes biased and unbiased HSIC, part of HSIC
# necessary for faster its faster computation, and functions that enable
# an optimization on HSIC with respect to the kernel parameters.
#
class CHSIC(object):
    def __init__(self):
        pass

    ## Compute HLH give the labels.
    # @param y The labels.
    # @param kernely The kernel on the labels, default to linear kernel.
    #
    def ComputeHLH(self, y, kernely=vector.CLinearKernel()):
        ny = y.shape
        if len(ny) > 1:
            lMat = kernely.Dot(y, y)
        else:
            lMat = numpy.outerproduct(y, y)

        sL = numpy.sum(lMat, axis=1)
        ssL = numpy.sum(sL)
        # hlhMat
        return lMat - numpy.add.outer(sL, sL)/ny[0] + ssL/(ny[0]*ny[0])

    ## Compute the biased estimator of HSIC.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data, default to linear kernel.
    # @param kernely The kernel on the labels, default to linear kernel.
    #
    def BiasedHSIC(self, x, y, kernelx=vector.CLinearKernel(), \
                   kernely=vector.CLinearKernel()):
        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
               "Argument 1 and 2 have different number of data points"

        if len(nx) > 1:
            kMat = kernelx.Dot(x, x)
        else:
            kMat = numpy.outerproduct(x, x)

        hlhMat = ComputeHLH(y, kernely)
        return numpy.sum(numpy.sum(kMat * hlhMat)) / ((nx[0]-1)*(nx[0]-1))

    ## Objective of the biased HSIC when performing optimization over
    # the kernel parameters.
    # @param param The kernel parameters.
    # @param x The data.
    # @param kernelx The kernel on the data.
    # @param hlhMat The HLH matrix on the labels.
    #
    def ObjBiasedHSIC(self, param, x, kernelx, hlhMat):
        nx = x.shape
        kMat = kernelx.DotCacheKernel(x, param)
        return -numpy.sum(numpy.sum(kMat * hlhMat)) / ((nx[0]-1)*(nx[0]-1))

    ## Gradient of the objective of the biased HSIC when performing
    # optimization over the kernel parameters.
    # @param param The kernel parameters.
    # @param x The data.
    # @param kernelx The kernel on the data.
    # @param hlhMat The HLH matrix on the labels.
    #
    def GradBiasedHISC(self, param, x, kernelx, hlhMat):
        nx = x.shape
        kMat = kernelx.GradDotCacheKernel(x, param)
        return -numpy.sum(numpy.sum(kMat * hlhMat)) / ((nx[0]-1)*(nx[0]-1))

    ## Fast computation of the biased HSIC when the kernel matrix
    # for the data and the HLH matrix for the labels are already
    # computed.
    # @param kMat The kernel matrix for the data.
    # @param hlhMat The HLH matrix for the labels.
    #
    def BiasedHSICFast(self, kMat, hlhMat):
        nx = kMat.shape
        assert (kMat.shape == hlhMat.shape), \
               "Argument 1 and 2 have different shapes"

        return (kMat * hlhMat).sum() / ((nx[0]-1)*(nx[0]-1))

    ## Fast computation of the biased HSIC when the kernel matrix
    # for the labels can be decomposed into HLH = y * y' and the
    # rank of y is low
    # @param kMat The kernel matrix for the data.
    # @param y The HLH = y * y' for the labels.
    #
    def BiasedHSICFast2(self, kMat, y):
        nx = kMat.shape
        assert (kMat.shape[0] == y.shape[0]), \
               "Argument 1 and 2 have different shapes"

        return numpy.dot(y.T, numpy.dot(kMat, y)).trace() / ((nx[0]-1)*(nx[0]-1))

    ## Fast computation of the biased HSIC when the kernel matrix
    # for the data K can be decomposed into K = x * x' and that 
    # for the labels can be decomposed into HLH = y * y' and the
    # rank of y is low (this will be useful after incomplete cholesky
    # factorization
    # @param x The K = x * x' for the data.
    # @param y The HLH = y * y' for the labels.
    #
    def BiasedHSICFast3(self, x, y):
        nx = x.shape
        assert (x.shape[0] == y.shape[0]), \
               "Argument 1 and 2 have different shapes"

        return (numpy.dot(x.T, y)**2).sum() / ((nx[0]-1)*(nx[0]-1))   
    

    ## Compute the UNbiased estimator of HSIC.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data, default to linear kernel.
    # @param kernely The kernel on the labels, default to linear kernel.
    #
    def UnBiasedHSIC(self, x, y, kernelx=vector.CLinearKernel(), \
                     kernely=vector.CLinearKernel()):
        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
               "Argument 1 and 2 have different number of data points"

        kMat = kernelx.Dot(x,x)
        setdiag0(kMat)

        lMat = kernely.Dot(y,y)
        setdiag0(lMat)

        sK = kMat.sum(axis=1)
        ssK = sK.sum()
        sL = lMat.sum(axis=1)
        ssL = sL.sum()

        return ( kMat.__imul__(lMat).sum() + \
                 (ssK*ssL)/((nx[0]-1)*(nx[0]-2)) - \
                 2 * sK.__imul__(sL).sum() / (nx[0]-2) \
                 ) / (nx[0]*(nx[0]-3))

    ## Objective of the UNbiased HSIC when performing optimization over
    # the kernel parameters.
    # @param param The kernel parameters.
    # @param x The data.
    # @param kernelx The kernel on the data.
    # @param lMat The kernel matrix of the label.
    # @param sL The vector of the sum of each row of lMat.
    # @param ssL The vector of the sum of all entries in lMat.
    #
    def ObjUnBiasedHSIC(self, param, x, kernelx, lMat, sL, ssL):
        nx = x.shape    
        kMat = kernelx.DotCacheKernel(x, param)
        sK = numpy.sum(kMat, axis=1)
        ssK = numpy.sum(sK)

        return -( numpy.sum(numpy.sum(kMat*lMat)) \
                  + (ssK*ssL)/((nx[0]-1)*(nx[0]-2)) \
                  - 2*numpy.sum(sK*sL)/(nx[0]-2) \
                  ) / (nx[0]*(nx[0]-3))


    ## Gradient of the objective of the UNbiased HSIC when performing
    # optimization over the kernel parameters.
    # @param param The kernel parameters.
    # @param x The data.
    # @param kernelx The kernel on the data.
    # @param lMat The kernel matrix of the label.
    # @param sL The vector of the sum of each row of lMat.
    # @param ssL The vector of the sum of all entries in lMat.
    #
    def GradUnBiasedHSIC(self, param, x, kernelx, lMat, sL, ssL):
        nx = x.shape
        kMat = kernelx.GradDotCacheKernel(x, param)
        sK = numpy.sum(kMat, axis=1)
        ssK = numpy.sum(sK)

        return -( numpy.sum(numpy.sum(kMat*lMat)) \
                  + (ssK*ssL)/((nx[0]-1)*(nx[0]-2)) \
                  - 2*numpy.sum(sK*sL)/(nx[0]-2) \
                  ) / (nx[0]*(nx[0]-3))

    ## Fast computation of the biased HSIC when the kernel matrix
    # for the data and the HLH matrix for the labels are already
    # computed.
    # @param kMat The kernel matrix for the data.
    # @param lMat The kernel matrix of the label.
    # @param sL The vector of the sum of each row of lMat.
    # @param ssL The vector of the sum of all entries in lMat.
    #
    def UnBiasedHSICFast(self, kMat, lMat, sL, ssL):
        nx = kMat.shape
        assert (kMat.shape == lMat.shape), \
               "Argument 1 and 2 have different shapes"

        sK = numpy.sum(kMat, axis=1)
        ssK = numpy.sum(sK)

        return ( numpy.sum(numpy.sum(kMat*lMat)) \
                 + (ssK*ssL)/((nx[0]-1)*(nx[0]-2)) \
                 - 2*numpy.sum(sK*sL)/(nx[0]-2) \
                 ) / (nx[0]*(nx[0]-3))

## Normalize each dimension of the data separately to zero mean and unit
# standard deviation.
# @param data [read\write] The data to be normalized. Each row is a
# datum and each column a dimension.
#
def normalize(data):
    m = data.mean(axis=0)
    s = data.std(axis=0)
    data.__isub__(m).__itruediv__(s)

## Center the kernel matrix in the feature space.
# @param k [read\write] The kernel matrix to be centered. 
#
def center(k):
    n = k.shape
    assert n[0] == n[1], 'k must be symmetric and positive semidefinite'    
    mk = k.mean(axis=1)
    mk.shape = (n[0], 1)
    mmk = mk.mean()
    k.__isub__(mk).__isub__(mk.T).__iadd__(mmk)

