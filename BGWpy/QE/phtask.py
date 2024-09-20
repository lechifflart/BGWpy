from __future__ import print_function
import os

from numpy import array
from ..core import fortran_str
from ..core import Namelist, Writable, Card
from .qetask import BaseQePhTask

from ..config import flavors

# Public
__all__ = ['QePhInput']

class QePhInput(Writable):

    def __init__(self, **kwargs):
        
        super(QePhInput, self).__init__(**kwargs)
        
        self.title_line = str()
        self.inputph = Namelist('inputph')
        self.xq = list()
        self.qpointsspecs = Card('QPOINTSSPECS', '')
        self.atom = list()
        
        if 'variables' in kwargs:
            self.set_variables(kwargs['variables'])
    
    def _iswavevector(self):
        """True if ldisp != .true. and qplot != .true."""
        not_ldisp = self.inputph.ldisp != True
        not_qplot = self.inputph.qplot != True
        return not_ldisp and not_qplot
    
    def _isqplot(self):
        """True if qplot == .true."""
        return self.inputph.qplot == True
    
    def _isnattodo(self):
        """True if nat_todo has been specified"""
        return self.inputph.nat_todo != 0
    
    def set_variables(self, variables):
        """
        Use a nested dictionary to set variables.
        The items in the variables dictionary should
        be dictionaries for namelist input variables,
        and lists for card input variables.
        In case of card input variables, the first item of the list
        must correspond to the option.

        Example:

        pwscfinput.set_variables({
            'control' : {
                'verbosity' : 'high',
                'nstep' : 1,
                },
            'system' : {
                'nbnd' : 10,
                },
            'electrons' : {
                'conv_thr' : 1e-6,
                },
            'cell_parameters' : ['angstrom',
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.,
                ],
            'atomic_species' : ['',
                'Ga', 69.723, 'path/to/Ga/pseudo',
                'As', 74.921, 'path/to/As/pseudo',
                ],
            })
        """
        for key, val in variables.items():

            if key not in dir(self):
                continue
            obj = getattr(self, key)

            if isinstance(obj, Namelist):
                obj.update(val)
            elif isinstance(obj, Card):
                obj.option = val[0]
                while obj:
                    obj.pop()
                obj.extend(val[1:])


    def __str__(self):
        
        S  = ''
        S += fortran_str(self.title_line)
        S += str(self.inputph)
        
        if self._iswavevector():
            S += fortran_str(self.xq)
        elif self._isqplot():
            S += str(self.qpointsspecs)
        
        if self._isnattodo():
            S += fortran_str(self.atom)
        
        return S


# Daan; Base structure copied from QE2BGW task.
class QePhTask(BaseQePhTask):
    """Phonon calculation."""

    _TASK_NAME = 'PHonon'

    _input_fname = 'ph.in'
    _output_fname = 'ph.out'

    def __init__(self, dirname, **kwargs):
        """
        Arguments
        ---------

        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.


        Keyword arguments
        -----------------
        (All mandatory unless specified otherwise)

        prefix : str
            Prepended to input/output filenames; must be the same
            used in the calculation of unperturbed system.
        
        Properties
        ----------
        
        """
        
        kwargs.setdefault('runscript_fname', 'ph.run.sh')
        
        super(QePhTask, self).__init__(dirname, **kwargs)
        
        self.title_line = kwargs.pop('title_line')
        
        # Maybe let the defaults be handled by PW2BGWInput
        
        # Input file
        defaults = dict(
            title_line  = self.title_line,
        )
        
        variables = dict()
        for key, value in defaults.items():
            variables[key] = kwargs.get(key, value)
        
        self.input = QePhInput(prefix=self.prefix, **variables)
        self.input.fname = self._input_fname

        # Run script
        self.runscript['PH'] = 'ph.x'
        self.runscript.append('$MPIRUN $PH $PHFLAGS -in {} &> {}'.format(
                              self._input_fname, self._output_fname))
    
    # _wfn_fname = 'wfn.cplx'
    # @property
    # def wfn_fname(self):
    #     return os.path.join(self.dirname, self._wfn_fname)
    
    # @wfn_fname.setter
    # def wfn_fname(self, value):
    #     self._wfn_fname = value
    #     self.input['wfng_file'] = value
