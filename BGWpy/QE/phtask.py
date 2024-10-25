from __future__ import print_function

from ..core import fortran_str
from ..core import Namelist, Writable, Card
from .qetask import BaseQePhTask


# Public
__all__ = ['QePhInput', 'QePhTask']

class QePhInput(Writable):

    def __init__(self, fname, *args, **kwargs):
        
        super(QePhInput, self).__init__(fname)
        
        self.title_line = str()
        self.inputph = Namelist('inputph')
        self.xq = list()
        self.qpointsspecs = Card('', '', quotes=False)
        self.atom = list()
        # Default settings
        defaults = dict(
            title_line = '',
            xq = list(),
            inputph = dict(
                ldisp = False,
                qplot = False,
                nat_todo = 0),
            atom = list(),
        )
        # Ensure all default values are set!
        # If kwargs specifies an inputph dict with ldisp, qplot or nat_todo missing
        # the checks in __str__ would fail as they would not be set in self.inputph
        self.set_variables(defaults)
        
        # Override default from kwargs
        variables = defaults
        for key, value in defaults.items():
            variables[key] = kwargs.get(key, value)
        # Set amass(i)
        if 'amass' in kwargs:
            amass = kwargs['amass']
            for [index, mass] in amass:
                key = 'amass({0})'.format(int(index))
                variables['inputph'][key] = mass
        # Other inputph
        keys = ['qplot', 'ldisp', 'nq1','nq2','nq3', 
                'asr', 'nogg', 'tr2_ph', 'fildyn', 'reduce_io',
                'electron_phonon']
        for key in keys:
            if key in kwargs:
                variables['inputph'][key] = kwargs.get(key)
        self.set_variables(variables)
        
        # Set qPointsSpecs
        if self._isqplot():
            # Add number of qpoints to qPointsSpecs
            nqs = len( self.xq )
            self.qpointsspecs.append(nqs)
            # default value of 1 for the weights.
            nq = kwargs.get('nq',[1]*nqs)
            if len(nq) != nqs:
                raise ValueError('Not enough weights supplied in nq.')
            # Add points and weights to qPointsSpecs
            for qpoint, w in zip(self.xq, nq):
                self.qpointsspecs.append(list(qpoint) + [int(w)])
        
        # Alternative method for entering variables.
        if 'variables' in kwargs:
            self.set_variables(kwargs['variables'])
        
    def _iswavevector(self):
        """True if ldisp != .true. and qplot != .true."""
        not_ldisp = self.inputph['ldisp'] != True
        not_qplot = self.inputph['qplot'] != True
        return not_ldisp and not_qplot
    
    def _isqplot(self):
        """True if qplot == .true."""
        return self.inputph['qplot'] == True
    
    def _isnattodo(self):
        """True if nat_todo has been specified"""
        return self.inputph['nat_todo'] != 0
    
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
            else:
                setattr(self, key, val)
    
    def __str__(self):
        
        S  = ''
        S += fortran_str(self.title_line, False) + '\n'
        S += str(self.inputph)
        
        if self._iswavevector():
            S += fortran_str(self.xq) + '\n'
        elif self._isqplot():
            S += str(self.qpointsspecs)
        
        if self._isnattodo():
            S += fortran_str(self.atom) + '\n'
        
        return S
    
    # Daan ; Input sanitization could be globally implemented using a Line-of-input object
    # instead of just str. This would mean not having to do this property/setter per str variable.
    _title_line = str()
    @property
    def title_line(self):
        return self._title_line
    @title_line.setter
    def title_line(self, value):
        self._title_line = value.strip().replace('\n','')


# Daan ; Base structure copied from QE2BGW task.
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
        ldisp : bool (False), optional
            Use a wave-vector grid displaced by half a grid step
            in each direction - meaningful only when ldisp is .true.
            When this option is set, the q2r.x code cannot be used.
        qplot : bool (False), optional
            If .true. a list of q points is read from input.
        nat_todo : int (0), optional
            Choose the subset of atoms to be used in the linear response
            calculation.
        xq : list(3), float, it depends
            The phonon wavevector, in units of 2pi/a0
            (a0 = lattice parameter).
            Not used if ldisp==True or qplot==True
        xq : list(N,3), float, it depends
            q-point coordinates; used only with ldisp==True and qplot==True.
            The phonon wavevector, in units of 2pi/a0 (a0 = lattice parameter).
        nq : list(1), int, it depends
            The weight of the q-point; the meaning of nq depends on 
            the flags q2d and q_in_band_form.
        Properties
        ----------
        
        """
        
        kwargs.setdefault('runscript_fname', 'ph.run.sh')
        
        super(QePhTask, self).__init__(dirname, **kwargs)
        
        # Construct input
        inp = QePhInput('input_ph', **kwargs)
        
        # Set mandatory
        inp.inputph.update(
            prefix = self.prefix
        )
        # store input
        self.input = inp
        
        # input filename
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
