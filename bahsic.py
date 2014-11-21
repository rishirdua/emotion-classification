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

##\package elefant.fselection.bahsic
# This module perform backward elimination for feature selection
# using HSIC (BAHSIC).
#
# The algorithm proceeds recursively, eliminating the least
# relevant features and adding them to the eliminated list
# in each iteration. For more theoretical underpinning see the
# following reference for more information:
#
# Le Song, Justin Bedo, Karsten M. Borgwardt, Arthur Gretton
# and Alex Smola. The BAHSIC family of gene selection algorithms.
#

__version__ = '$Revision: $' 
# $Source$

import numpy
from scipy import optimize

import vector
from hsic import CHSIC
from setdiag0 import setdiag0


## Class that perform backward elimination for feature selection (BAHSIC).
#
# It has two version of BAHSIC: one without optimization over the kernel
# parameters and one with optimization over the kernel parameters.
#
class CBAHSIC(object):
    def __init__(self):
        pass

    ## BAHSIC with optimization over the kernel parameters.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data.
    # @param kernely The kernel on the labels.
    # @param flg3 The number of desired features.
    # @param flg4 The proportion of features eleminated in each iteration.
    #
    def BAHSICOpt(self, x, y, kernelx, kernely, flg3, flg4):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y.shape) == 2, 'Argument 2 has wrong shape'
        assert x.shape[0] == y.shape[0], \
               'Argument 1 and 2 have different number of data points'
                       
        print '--initializing...'
        hsic = CHSIC()
        
        L = kernely.Dot(y, y)
        setdiag0(L)
        sL = numpy.sum(L, axis=1)
        ssL = numpy.sum(sL)

        n = x.shape
        eliminatedI = []
        selectedI = set(numpy.arange(n[1]))

        kernelx.CreateCacheKernel(x)
        sga = kernelx._typicalParam
        sgaN = sga.shape
        sgaN = sgaN[0]

        while True:        
            selectedI = selectedI - set(eliminatedI)
            sI = numpy.array([j for j in selectedI])
            m = len(sI)

            print m
            if (m == 1):
                eliminatedI.append(selectedI.pop())
                break

            sgaMat = []
            hsicMat = []
            for k in range(sgaN):
                ## bfgs in scipy is not working here
                retval = optimize.fmin_cg(hsic.ObjUnBiasedHSIC, \
                                          sga[[k],].ravel(), \
                                          hsic.GradUnBiasedHSIC,\
                                          args=[x, kernelx, L, sL, ssL], \
                                          gtol=1e-6, maxiter=100, \
                                          full_output=True, disp=False)
                sgaMat.append(retval[0])
                hsicMat.append(retval[1])
                    
            k = numpy.argmin(hsicMat)
            sga0 = sgaMat[k]
            
            objj = []
            for j in selectedI:
                K = kernelx.DecDotCacheKernel(x, x[:,[j]], sga0)
                setdiag0(K)
                objj.append(hsic.UnBiasedHSICFast(K, L, sL, ssL))

            if m > flg3:
                maxj = numpy.argsort(objj)
                num = int(flg4 * m)+1
                if m - num <= flg3:
                    num = m - flg3
                maxj = maxj[m:m-num-1:-1]
            else:
                maxj = numpy.array([numpy.argmax(objj)])
                
            j = numpy.take(sI,maxj)
            eliminatedI.extend(j)
            kernelx.DecCacheKernel(x, x[:,j])

        kernelx.ClearCacheKernel(x)
        return eliminatedI

    ## BAHSIC without optimization over the kernel parameters.
    # @param x The data.
    # @param y The labels.
    # @param kernelx The kernel on the data.
    # @param kernely The kernel on the labels.
    # @param flg3 The number of desired features.
    # @param flg4 The proportion of features eleminated in each iteration.
    #
    def BAHSICRaw(self, x, y, kernelx, kernely, flg3, flg4):
        assert len(x.shape) == 2, 'Argument 1 has wrong shape'
        assert len(y.shape) == 2, 'Argument 2 has wrong shape'
        assert x.shape[0] == y.shape[0], \
               'Argument 1 and 2 have different number of data points'       

        print '--initializing...'
        hsic = CHSIC()

        L = kernely.Dot(y, y)
        setdiag0(L)

        sL = numpy.sum(L, axis=1)
        ssL = numpy.sum(sL)

        n = x.shape
        eliminatedI = []
        selectedI = set(numpy.arange(n[1]))

        kernelx.CreateCacheKernel(x)

        while True:
            selectedI = selectedI - set(eliminatedI)
            sI = numpy.array([j for j in selectedI])
            m = len(sI)

            print m
            if (m == 1):
                eliminatedI.append(selectedI.pop())
                break

            objj = []
            for j in selectedI:
                K = kernelx.DecDotCacheKernel(x, x[:,[j]])
                setdiag0(K)
                objj.append(hsic.UnBiasedHSICFast(K, L, sL, ssL))

            if m > flg3:
                maxj = numpy.argsort(objj)
                num = int(flg4 * m)+1
                if m-num <= flg3:
                    num = m - flg3
                maxj = maxj[m:m-num-1:-1]
            else:
                maxj = numpy.array([numpy.argmax(objj)])

            j = numpy.take(sI,maxj)
            eliminatedI.extend(j)
            kernelx.DecCacheKernel(x, x[:,j])

        kernelx.ClearCacheKernel(x)
        return eliminatedI
