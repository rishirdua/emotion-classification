#!/usr/bin/env python

# Copyright (c) 2004 National ICT Australia --- All Rights Reserved
# THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF SML.NICTA
# The copyright notice above does not evidence any
# actual or intended publication of this work.
#
# Authors:      Le Song
# Last changed: 02/08/2006 (Christfried Webers)

import numpy

def setdiag0(K):
    """Set the diagonal entries of a square matrix to 0
    """
    n = K.shape[0]
    numpy.put(K, numpy.arange(n) * (n + 1), 0.0)
