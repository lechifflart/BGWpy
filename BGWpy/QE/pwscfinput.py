
from ..core import fortran_str
from ..core import Writable, Namelist, Card
import warnings
import numpy as np

class PWscfInput(Writable):

    _structure = None 
    _pseudos = list()

    def __init__(self, **kwargs):

        super(PWscfInput, self).__init__(**kwargs)

        self.control = Namelist('control')
        self.system = Namelist('system')
        self.electrons = Namelist('electrons')
        self.ions = Namelist('ions')
        self.cell = Namelist('cell')
        self.atomic_species = Card('ATOMIC_SPECIES', '')
        self.atomic_positions = Card('ATOMIC_POSITIONS','crystal',quotes=False)
        self.k_points = Card('K_POINTS', '')
        self.cell_parameters = Card('CELL_PARAMETERS', 'angstrom')
        self.constraints = Card('CONSTRAINTS', '')
        self.occupations = Card('OCCUPATIONS', '')
        self.atomic_forces = Card('ATOMIC_FORCES', '')

        if 'variables' in kwargs:
            self.set_variables(kwargs['variables'])

    def _isrelax(self):
        """True if this is a relaxation calculation."""
        calculation = str(self.control.get('calculation')).lower()
        return calculation in ('relax', 'md', 'vc-relax', 'vc-md' )

    def _isvc(self):
        """True if the volume of the cell can change."""
        calculation = str(self.control.get('calculation')).lower()
        return calculation in ('vc-relax', 'vc-md' )

    def _isfreebrav(self):
        """True if the bravais lattis is free."""
        ibrav = self.system.get('ibrav')
        if ibrav is None:
            return False
        return int(ibrav) == 0

    def _isconstrained(self):
        """True if the ion dynamics is constrained."""
        return self.ions.get('ion_dynamics') in ('verlet', 'damp')

    def _is_manual_occ(self):
        """True if the occupation is specified manually."""
        return str(self.system.get('occupations','')).lower() == 'from_input'

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

        # Perform checks
        self.check_pseudos_names()

        S = ''
        S += str(self.control)
        S += str(self.system)
        S += str(self.electrons)

        if self.ions or self._isrelax():
            S += str(self.ions)

        if self.cell or self._isvc():
            S += str(self.cell)

        if self.cell_parameters or self._isfreebrav():
            S += str(self.cell_parameters)

        # This card is special, since only parts of it need quotes.
        #S += str(self.atomic_species)
        S += '{} {}\n'.format(self.atomic_species.name,
                              self.atomic_species.option)
        for val in self.atomic_species:
            S += '   {} {}\n'.format(fortran_str(val[0], quotes=False),
                                     fortran_str(val[1:], quotes=True))


        S += str(self.atomic_positions)

        if self.constraints or self._isconstrained():
            S += str(self.constraints)

        if self.occupations or self._is_manual_occ():
            S += str(self.occupations)

        if self.atomic_forces:
            S += str(self.atomic_forces)

        S += str(self.k_points)

        return S

    def set_kpoints_crystal(self, kpts, wtks, kpts_option):
        self.k_points.option = kpts_option
        if kpts_option in ['crystal', 'crystal_b']:
            self.k_points.append(len(kpts))
            for k, w in zip(kpts, wtks):
                self.k_points.append(list(k) + [w])
        elif kpts_option == 'automatic':
            self.k_points.append(list(kpts) + list(wtks))

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        """A pymatgen.Structure object."""
        self._structure = structure

        # Set cell parameters
        self.cell_parameters.option = 'angstrom'
        #for vec in structure.lattice_vectors():
        for vec in structure.lattice.matrix:
            self.cell_parameters.append(np.round(vec, 8))
        
        # Set atomic label and mass
        species_label = sorted(set(structure.labels), key=structure.labels.index)
        for label in species_label:
            species = structure.species[ structure.labels.index(label) ]
            # Quantum Espresso does not allow for labels of length > 3
            if len(label) > 3:
                warnings.warn('Label length of atom `{label}` cannot exceed 3 characters!'.format(label))
            self.atomic_species.append([label, float(species.atomic_mass), ''])
        
        if self.pseudos:
            for ii, pseudo in enumerate(self.pseudos):
                self.atomic_species[ii][2] = pseudo
        
        # Set system specifications
        self.system['ibrav'] = 0
        self.system['nat'] = structure.num_sites
        self.system['ntyp'] = len(species_label)

        
        
        # Set atomic positions
        self.atomic_positions.option = 'crystal'
        for site, label in zip(structure.sites, structure.labels):
            frac_coords = list(site.frac_coords)
            for i in range(3):
                if abs(frac_coords[i]) > .5:
                    frac_coords[i] += -1. * np.sign(frac_coords[i])
            self.atomic_positions.append([label] + frac_coords)

    @property
    def pseudos(self):
        return self._pseudos

    @pseudos.setter
    def pseudos(self, pseudos):
        self._pseudos = pseudos
        if self.atomic_species:
            if len(self.atomic_species) != len(pseudos):
                warnings.warn('Number of labelled atoms does not match number of pseudos!')
            for [ii, pseudo], _ in zip(enumerate(pseudos), self.atomic_species):
                self.atomic_species[ii][2] = pseudo

    def check_pseudos_names(self):
        species_info = [ len(row)==3 for row in self.atomic_species ]
        if not any(species_info):
            raise Exception('Cannot check pseudos if they have not been set!')
        if any(species_info) and not all(species_info):
            raise Exception('Only some pseudos have been set!')
        for symbol, _, pseudo in self.atomic_species:
            if symbol.lower() not in pseudo.lower():
                warnings.warn('Suspicious pseudo name for atom {}: {}'.format(symbol, pseudo))
