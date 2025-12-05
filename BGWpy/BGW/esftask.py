from __future__ import print_function
import os

from .bgwtask import BGWTask
from .inputs import ESFInput
from ..core import BasicInputFile, Namelist

__all__ = ['ESFTask']


class ESFTask(BGWTask):
    """Excited-state forces workflow helper."""

    _TASK_NAME = 'ExcitedStateForces'
    _input_fname = 'forces.inp'
    _dynmat_input_fname = 'dynmat.inp'
    _output_fname = 'esf.out'
    _TAG_JOB_COMPLETED = ''

    def __init__(self, dirname, structure, prefix, eqp_fname, exciton_fname,
                 el_ph_dir, dynmat_fname, scf_out_fname=None,
                 excited_forces_script='excited_forces.py',
                 rand_disp_script=None, harmonic_script=None,
                 first_iteration=False, iexc=1, dynmat_variables=None,
                 extra_lines=None, extra_variables=None,
                 log_results=True, log_fname='esf_log.txt',
                 atoms_fname='atoms', atoms_info_fname='Atoms_info',
                 forces_input_fname=None, dynmat_input_fname=None,
                 dynmat_output_fname='dynmat.out',
                 python_executable='python', **kwargs):
        """
        Build a job that runs dynmat, excited_forces and optional extrapolation.

        Required parameters mirror the manual commands previously used in
        workflow_esf.py so this task can be dropped in alongside other BGW
        tasks.
        """

        super(ESFTask, self).__init__(dirname, **kwargs)

        self.structure = structure
        self.prefix = prefix
        self.atoms_fname = atoms_fname
        self.atoms_info_fname = atoms_info_fname
        self.log_results = log_results
        self.log_fname = log_fname
        self.excited_forces_script = excited_forces_script
        self.rand_disp_script = rand_disp_script
        self.harmonic_script = harmonic_script
        self.first_iteration = first_iteration
        self.python_executable = python_executable

        self.eqp_dest = kwargs.get('eqp_dest', 'eqp.dat')
        self.exciton_dest = kwargs.get('exciton_dest', 'eigenvectors.h5')
        self.el_ph_dest = kwargs.get('el_ph_dest', f'{prefix}.phsave')
        self.dynmat_dest = kwargs.get('dynmat_dest', 'dyn')
        self.scf_out_dest = kwargs.get('scf_out_dest', 'out')

        self._dynmat_input_fname = dynmat_input_fname or self._dynmat_input_fname
        self.dynmat_output_fname = dynmat_output_fname

        # Input file for excited_forces.py
        self.input = ESFInput(
            iexc,
            self.eqp_dest,
            self.exciton_dest,
            self._format_elph_dir(self.el_ph_dest),
            *(extra_lines or []),
            **(extra_variables or {})
        )
        self.input.fname = forces_input_fname or self._input_fname

        # Input file for dynmat.x
        dynmat_vars = {'fildyn': self.dynmat_dest, 'asr': 'simple', 'fileig': 'eigvecs'}
        if dynmat_variables:
            dynmat_vars.update(dynmat_variables)
        self.dynmat_variables = dynmat_vars
        dynmat_keywords = str(Namelist('input', **self.dynmat_variables)).strip().splitlines()
        self.dynmat_input = BasicInputFile(
            fname=self._dynmat_input_fname,
            keywords=dynmat_keywords,
        )

        # Links
        self.eqp_fname = eqp_fname
        self.exciton_fname = exciton_fname
        self.el_ph_dir = el_ph_dir
        self.dynmat_fname = dynmat_fname
        self.scf_out_fname = scf_out_fname

        # Run script
        self.runscript['DYNMAT'] = 'dynmat.x'
        self.runscript['EXCITED_FORCES'] = self.excited_forces_script
        if self.rand_disp_script:
            self.runscript['RANDOM_DISP'] = self.rand_disp_script
        if self.harmonic_script:
            self.runscript['HARMONIC'] = self.harmonic_script

        self.runscript.append('$DYNMAT -inp {} > {}'.format(self.dynmat_input.fname,
                                                            self.dynmat_output_fname))
        self.runscript.append('{} $EXCITED_FORCES > {}'.format(self.python_executable,
                                                               self._output_fname))

        if self.first_iteration and self.rand_disp_script:
            self.runscript.append('{} $RANDOM_DISP > rdft.log'.format(self.python_executable))
        elif (not self.first_iteration) and self.harmonic_script:
            self.runscript.append('{} $HARMONIC > harmonic.log'.format(self.python_executable))

        if self.log_results:
            self.runscript.extend(self._log_commands())

    @property
    def eqp_fname(self):
        return self._eqp_fname

    @eqp_fname.setter
    def eqp_fname(self, value):
        self._eqp_fname = value
        self.update_link(value, self.eqp_dest)

    @property
    def exciton_fname(self):
        return self._exciton_fname

    @exciton_fname.setter
    def exciton_fname(self, value):
        self._exciton_fname = value
        self.update_link(value, self.exciton_dest)

    @property
    def el_ph_dir(self):
        return self._el_ph_dir

    @el_ph_dir.setter
    def el_ph_dir(self, value):
        self._el_ph_dir = value
        self.update_link(value, self.el_ph_dest)

    @property
    def dynmat_fname(self):
        return self._dynmat_fname

    @dynmat_fname.setter
    def dynmat_fname(self, value):
        self._dynmat_fname = value
        self.update_link(value, self.dynmat_dest)

    @property
    def scf_out_fname(self):
        return self._scf_out_fname

    @scf_out_fname.setter
    def scf_out_fname(self, value):
        self._scf_out_fname = value
        if value:
            self.update_link(value, self.scf_out_dest)
        else:
            self.remove_link(self.scf_out_dest)

    def write(self):
        # Ensure directory exists before any writes
        os.makedirs(self.dirname, exist_ok=True)

        super(ESFTask, self).write()
        with self.exec_from_dirname():
            self.input.write()
            self.dynmat_input.write()
            self._write_atoms_files()

    def _write_atoms_files(self):
        atoms_path = self.atoms_fname
        atoms_info_path = self.atoms_info_fname

        with open(atoms_path, 'w') as f_atoms:
            for specie in self.structure.species:
                f_atoms.write(f'{specie.symbol} {float(specie.atomic_mass)}\n')

        with open(atoms_info_path, 'w') as f_atoms_info:
            for coord, specie in zip(self.structure.cart_coords, self.structure.species):
                f_atoms_info.write(f'{specie.symbol} {float(specie.atomic_mass)} {coord[0]} {coord[1]} {coord[2]}\n')

    def _format_elph_dir(self, dirname):
        return dirname if dirname.endswith('/') else dirname + '/'

    def _log_commands(self):
        commands = [
            f'esf_log={self.log_fname}',
            'echo "[[entry]]" >> $esf_log',
            'echo "[structure]" >> $esf_log',
            "find ../ -name 'scf.*.json' | sort -V | tail -n 1 >> $esf_log",
            'echo "[energy]" >> $esf_log',
        ]
        if self.scf_out_fname:
            commands.append(f'grep ! {self.scf_out_dest} >> $esf_log')
        commands.extend([
            f'grep Omega {self._output_fname} >> $esf_log',
            'echo "[exc forces]" >> $esf_log',
            'cat forces_cart.out >> $esf_log',
        ])
        if self.scf_out_fname:
            commands.extend([
                'echo "[scf forces]" >> $esf_log',
                f"grep -Pzo '(?s)Forces.*?Total force' {self.scf_out_dest} >> $esf_log",
            ])
        commands.append('echo >> $esf_log')
        return commands
