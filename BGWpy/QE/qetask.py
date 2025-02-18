from __future__ import print_function

import os

from ..config import flavors
from ..core.util import exec_from_dir
from ..core import MPITask, IOTask
from ..DFT import DFTTask

# Public
__all__ = ['QeDFTTask','BaseQePhTask']

class QeTask(IOTask):
    """Base class for Quantum Espresso calculations."""

    _TAG_JOB_COMPLETED = 'JOB DONE'
    _use_hdf5_qe = flavors['use_hdf5_qe']

    def __init__(self, dirname, **kwargs):
        """
        Arguments
        ---------

        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.

        Keyword arguments
        -----------------

        See also:
            BGWpy.DFT.DFTTask

        """

        super(QeTask, self).__init__(dirname, **kwargs)
        
        self.prefix = kwargs['prefix']
        self.savedir = self.prefix + '.save'
        self.linked_savedir = kwargs.get('linked_savedir', None)

    def exec_from_savedir(self):
        original = os.path.realpath(os.curdir)
        if os.path.realpath(original) == os.path.realpath(self.dirname):
            return exec_from_dir(self.savedir)
        return exec_from_dir(os.path.join(self.dirname, self.savedir))

    def write(self):
        if self.linked_savedir:
            self.update_link(self.linked_savedir, self.savedir)
        
        super(QeTask, self).write()
        
        with self.exec_from_dirname():
            self.input.write()
            if self.linked_savedir: pass
            elif not os.path.exists(self.savedir):
                os.makedirs(self.savedir, exist_ok=True)
        
            

class QeDFTTask(DFTTask, QeTask):
    """Base class for Quantum Espresso calculations."""

    def __init__(self, dirname, **kwargs):
        """
        Arguments
        ---------

        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.

        Keyword arguments
        -----------------

        See also:
            BGWpy.DFT.DFTTask

        """

        super(QeDFTTask, self).__init__(dirname, **kwargs)

        self.runscript['PW'] = kwargs.get('PW', 'pw.x')
        self.runscript['PWFLAGS'] = kwargs.get('PWFLAGS', ' ')
        
    def write(self):
        self.check_pseudos()
        super(QeDFTTask, self).write()
    
    # Yikes! I have to recopy the property. python3 would be so much better...
    @property
    def pseudo_dir(self):
        return self._pseudo_dir

    @pseudo_dir.setter
    def pseudo_dir(self, value):
        if os.path.realpath(value) == value.rstrip(os.path.sep):
            self._pseudo_dir = value
        else:
            self._pseudo_dir = os.path.relpath(value, self.dirname)
        if 'input' in dir(self):
            if 'control' in dir(self.input):
                self.input.control['pseudo_dir'] = self._pseudo_dir

class BaseQePhTask(MPITask, QeTask):
    """Base class for Quantum Espresso phonon calculations."""
    
    def __init__(self, dirname, **kwargs):
        """
        Arguments
        ---------

        dirname : str
            Directory in which the files are written and the code is executed.
            Will be created if needed.
        
        Keyword arguments
        -----------------
        
        """
        super(BaseQePhTask, self).__init__(dirname, **kwargs)
        
        # TODO : task parity : move these lines to its own class BGWpy/Phonon/PhTask.py
        # See BGWpy/DFT/dfttask.py for reference of how this is handled.
        # vvv
        self.flavor = kwargs.pop('flavor',  'qe')
        if self.is_flavor_QE:
            self.version = kwargs.pop('version',  6)
        # ^^^

        self.runscript['PH'] = kwargs.get('PH', 'ph.x')
        self.runscript['PHFLAGS'] = kwargs.get('PHFLAGS', ' ')

    @property
    def is_flavor_QE(self):
        return any([tag in self.flavor.lower() for tag in ['qe', 'espresso']])
