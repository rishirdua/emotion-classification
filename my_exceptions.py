# Copyright (c) 2007, National ICT Australia
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
# Authors: Christfried Webers
# Created: (09/10/2007)
# Last Updated: 
#

## Exception classes for the Elefant project

class CElefantException(Exception):
    """Base class for exceptions in Elefant."""
    pass


class CElefantConstraintException(CElefantException):
    """Exception raised for constraint violation.
    
       Attributes:
            value   -- input value violating constrained
            message -- explanation of the error
    """
    
    def __init__(self, value, message):
        self.value = value
        self.message = message
        
    