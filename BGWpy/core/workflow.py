# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:54:44 2024

Template workflow script that creates tasks in the BGWpy method.

@author: Daan Holleman and Fabian Lie
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message="networkx backend defined more than once: nx-loopback")

from os.path import basename, abspath, join as pjoin
from BGWpy import (
    Structure,
    Workflow,
    QePhTask,
    QeScfTask,
    QeWfnTask,
    Qe2BgwTask,
    EpsilonTask,
    SigmaTask,
    KernelTask,
    AbsorptionTask,
    ESFTask,
)
import numpy as np
from glob import glob
import os

workflow = Workflow(dirname='.')
WDIR = basename(abspath('.'))
# %% USER CONFIGURATION (edit here for new materials)
MATERIAL = dict(
    prefix='LiF',
    pseudo_dir='/home/flie/pseudo',
    pseudos=['Li.upf','F.upf'],
    valence_bands_per_prim=5,       # occupied bands per primitive cell
    conv_nbnd_per_prim=90,          # converged nbands per primitive cell (before supercell scaling)
    ecutwfc=90,
    ecutsig=12,
    ecuteps=12,
    primitive_atoms=2,              # atoms in the primitive cell; used to deduce supercell factor, for general use, put same as amount                                    # of atoms in unit cell
    spin_orbit=True,
    spin_orbit_phonons=True
)

BSE_COUNTS = dict(                  # bands included in kernel and absorption
    nbnd_val_per_prim=5,
    nbnd_cond_per_prim=10,
    nbnd_val_fi_per_prim=5,
    nbnd_cond_fi_per_prim=5,
)

GRIDS = dict(
    qshift=[.001, .001, .001],
    kshift=[0., 0., 0.],
    ngkpt_scf=[3, 3, 3],
    ngkpt_fi=[3, 3, 3],
    ngkpt_co=[3, 3, 3],
    fft=[48, 48, 48],
)

# User-editable run options
RUN_OPTIONS = dict(
    use_operator='use_momentum',                 # velocity, momentum, bands
    screening_type='screening_semiconductor',    # semiconductor, metal, graphene
    sym_coarse_grid='no_symmetries_coarse_grid',
    sym_fine_grid='no_symmetries_fine_grid',
)

# Change this to the repository of the excited_state_forces location
loc_excited_forces = '/home/flie/excited_state_forces/main/excited_forces.py'
loc_rand_disp_finite_temp = '/home/flie/excited_state_forces/main/rand_disp_finite_temp.py'
loc_harmonic_extrapolation = '/home/flie/excited_state_forces/main/harmonic_extrapolation.py'

fn_newton = 'esf_files/displacements_Newton_method.dat'
fn_rdft = 'esf_files/atomic_disp_rand_displacements'

QE_LIBRARY_PATH = '/projects/0/nwo20035/ravindra/CPU/QE-7.2/lib:/projects/0/nwo20035/ravindra/CPU/libxc-6.1.0/lib'
BGW_LIBRARY_PATH = '/projects/0/nwo20035/ravindra/CPU/BGW-4.0/lib'
BGW_BIN_PATH = '/projects/0/nwo20035/ravindra/CPU/BGW-4.0/bin'

RESOURCE_GROUPS = {
    # Quantum ESPRESSO pw.x tasks
    'qe': {'time': '1:00:00', 'nodes': 1, 'ntasks_per_node': 64},
    # pw2bgw converter tasks (set separately from pw.x)
    'pw2bgw': {'time': '1:00:00', 'nodes': 1, 'ntasks_per_node': 32},
    # BerkeleyGW tasks
    'bgw': {'time': '2:00:00', 'nodes': 2, 'ntasks_per_node': 64},
}

# Optional per-job tweaks, overrides specific settings from RESOURCE_GROUPS, add a tag here with settings you want, and then add that tag to resources_kwargs at the actual task
JOB_RESOURCE_OVERRIDES = {
    'esf': {'time': '30:00'},       #esf inherits from the qe resource group if not specified
}

QE_WLM_BASE = [
    '--quiet',
    '-o log_%j.out',
    '-e error_%j.err',
    '--partition rome',
    '--mail-type=ALL',
]

BGW_WLM_BASE = [
    '--quiet',
    '--exclusive',
    '-o log_%j.out',
    '-e error_%j.err',
    '--partition rome',
]

QE_HEADER_BASE = [
    '# load modules',
    'module purge',
    'module load 2023',
    'module load QuantumESPRESSO/7.2-foss-2023a',
    '',
    'ulimit -s unlimited',
    '',
    f'export LD_LIBRARY_PATH={QE_LIBRARY_PATH}:$LD_LIBRARY_PATH',
    '',
    'export OMP_NUM_THREADS=1',
]

BGW_HEADER_BASE = [
    '# load modules',
    'module load 2023',
    'module load intel/2023a',
    'module load HDF5/1.14.0-iimpi-2023a',
    '',
    f'export LD_LIBRARY_PATH={BGW_LIBRARY_PATH}:$LD_LIBRARY_PATH',
    '',
    f'pathBGW={BGW_BIN_PATH}',
    'export PATH=$pathBGW:$PATH',
    'export I_MPI_COLL_SHM=0',
]


def qe_header_lines(extra=None):
    """Return Quantum ESPRESSO header commands followed by optional extras."""
    lines = list(QE_HEADER_BASE)
    if extra:
        lines.extend(extra)
    return lines


def bgw_header_lines(extra=None):
    """Return BerkeleyGW header commands followed by optional extras."""
    lines = list(BGW_HEADER_BASE)
    if extra:
        lines.extend(extra)
    return lines


def qe_wlm_lines(ntasks_per_node, extra=None):
    """Return sbatch lines tailored for QE tasks."""
    lines = [f'--ntasks-per-node {ntasks_per_node}', *QE_WLM_BASE]
    if extra:
        lines.extend(extra)
    return lines


def bgw_wlm_lines(ntasks_per_node, extra=None):
    """Return sbatch lines tailored for BGW tasks."""
    lines = [f'--ntasks-per-node {ntasks_per_node}', *BGW_WLM_BASE]
    if extra:
        lines.extend(extra)
    return lines


def resources_for(label, family='qe'):
    """
    Return the resource dict (time, nodes, ntasks_per_node) for a job label,
    merging family defaults with optional overrides.
    """
    base = dict(RESOURCE_GROUPS[family])
    base.update(JOB_RESOURCE_OVERRIDES.get(label, {}))
    return base


def resources_kwargs(label, family='qe'):
    """
    Return kwargs for mpirun_time/nodes/extra_WLM_lines for a given job label.
    """
    res = resources_for(label, family)
    if family in ('qe', 'pw2bgw'):
        wlm_fn = qe_wlm_lines
    elif family == 'bgw':
        wlm_fn = bgw_wlm_lines
    else:
        raise KeyError(f"Unknown resource family '{family}'")
    return dict(
        mpirun_time=res['time'],
        mpirun_nodes=res['nodes'],
        extra_WLM_lines=wlm_fn(res['ntasks_per_node'])
    )


def split_block(text):
    """Return a clean list of lines from a multi-line string."""
    return text.strip().splitlines()


def supercell_factor(structure):
    """Estimate supercell factor from number of atoms and primitive atom count."""
    return max(1, len(structure.species) // MATERIAL['primitive_atoms'])


def build_fallback_structure():
    """Load a fallback Structure from structure.json when scf.json is absent."""
    try:
        return Structure.from_file('structure.json')
    except Exception as exc:
        raise FileNotFoundError(
            "Fallback structure file 'structure.json'"
            "is missing or invalid."
        ) from exc


def load_structure():
    try:
        return Structure.from_file('scf.json'), False
    except Exception:
        return build_fallback_structure(), True


def load_displacements(cart_coords):
    """Load displacements from Newton and RDTF helpers if available."""
    def load_newton():
        try:
            data = np.genfromtxt(fn_newton)
            displacement = data[:, [1, 2, 3]] * 1.0  # first column is indices
            os.rename(fn_newton, fn_newton + '.old')
            print('NEWTON [\u2713]')
            return displacement
        except Exception as exc:
            print(exc)
            print('NEWTON [X]')
            return np.zeros_like(cart_coords)

    def load_rdft():
        try:
            coords = np.genfromtxt(fn_rdft)
            disp = coords[:, [1, 2, 3]]  # first column is element
            os.rename(fn_rdft, fn_rdft + '.old')
            print('RDFT   [\u2713]')
            return disp
        except Exception:
            print('RDFT   [X]')
            return cart_coords.copy()

    return load_newton(), load_rdft()


structure, first_iteration = load_structure()
cart_coords = structure.cart_coords
displacements, disp_coords = load_displacements(cart_coords)

new_cart_coords = disp_coords + displacements

lattice = structure.lattice
species = structure.species
structure = Structure(lattice, species, new_cart_coords, coords_are_cartesian=True)

json_files = glob('scf.*.json')
if json_files:
    iteration = max([int(fn.split('.')[1]) for fn in json_files]) + 1
else:
    iteration = 0
structure.to(f'scf.{iteration:03d}.json', fmt='json') # For visualisation
structure.to('scf.json', fmt='json') # For loading in next run

SCV = supercell_factor(structure)

prefix = MATERIAL['prefix']
pseudo_dir = MATERIAL['pseudo_dir']
pseudos = MATERIAL['pseudos']

soc = MATERIAL['spin_orbit']
socph = MATERIAL['spin_orbit_phonons']

## BANDS
valence = MATERIAL['valence_bands_per_prim']
conv_nbnd = MATERIAL['conv_nbnd_per_prim']
nbnd_val     = SCV * BSE_COUNTS['nbnd_val_per_prim']
nbnd_cond    = SCV * BSE_COUNTS['nbnd_cond_per_prim']
nbnd_val_fi  = SCV * BSE_COUNTS['nbnd_val_fi_per_prim']
nbnd_cond_fi = SCV * BSE_COUNTS['nbnd_cond_fi_per_prim']
# sigma
band_index_min = SCV * valence - nbnd_val + 1
band_index_max = SCV * valence + nbnd_cond
# wfn_co; wfn_fi
min_nbnd = valence * SCV + nbnd_cond + 1
nbnd    = conv_nbnd * SCV
nbnd_fi = valence * SCV + nbnd_cond_fi + 1
# feedback
if min_nbnd >= nbnd:
    raise ValueError('Not enough bands included in coarse grid!')
print(f'coarse: {nbnd}\nfine: \t{nbnd_fi}')
print(f'kernel: {nbnd_val}, {nbnd_cond}')

## ENERGIES
# scf; sigma; epsilon
ecutwfc = MATERIAL['ecutwfc']
ecfixed = MATERIAL['ecutwfc']
ecutsig = MATERIAL['ecutsig']
ecuteps = MATERIAL['ecuteps']

## SETTINGS
use_operator    = RUN_OPTIONS['use_operator']
screening_type  = RUN_OPTIONS['screening_type']
sym_coarse_grid = RUN_OPTIONS['sym_coarse_grid']
sym_fine_grid   = RUN_OPTIONS['sym_fine_grid']

## GRIDS
qshift = GRIDS['qshift']
kshift = GRIDS['kshift']
ngkpt_scf = GRIDS['ngkpt_scf']
ngkpt_fi  = GRIDS['ngkpt_fi']
ngkpt_co  = GRIDS['ngkpt_co']
fft       = GRIDS['fft']

# %% HELPER FUNCTIONS
# removes wavefunction files after running pw2bgw
remove_wfc = lambda task : task.runscript.append(f'rm -f "{prefix}.wfc*"')
# specify jobnames; change head to something recognisable / job specific
name = lambda label : f'{label}-{WDIR}' # e.g.: name(scf) -> LiF-D0-scf



# %% KEYWORD ARGUMENTS
## COMMON ARGUMENTS
kwargs = dict(
    structure = structure,
    prefix = prefix,
    pseudo_dir = pseudo_dir,
    pseudos = pseudos,
    symkpt = False
)
# use symmetry in kpoint generation
kwargs_sym = dict(
    symkpt = True
)
## QUANTUM ESPRESSO
kwargs_qe_base = dict(
    # WFN parameters
    fft = fft,
    ecutwfc = ecutwfc,
    extra_header_lines = qe_header_lines(),
)
# kwargs_qe_big = {**kwargs_qe_base, **dict(mpirun_nodes=2, nproc_per_node=64)}


## BERKELEY-GW
kwargs_bgw_base = dict(
    # Bands used to build the BSE Hamiltonian
    nbnd_val_fi  = nbnd_val_fi,
    nbnd_cond_fi = nbnd_cond_fi,
    nbnd_val_co  = nbnd_val,
    nbnd_cond_co = nbnd_cond,
    nbnd_val = nbnd_val,
    nbnd_cond = nbnd_cond,
    mpirun = 'mpirun',
    extra_header_lines = bgw_header_lines(),
)

# Override for phonon
kwargs_ph = {
    **kwargs_qe_base, 
    **dict(
        electron_phonon = 'simple',
        verbosity = 'high',
        title_line = 'phonon_calc',
        fildyn = 'dyn',
        fildvscf = 'dvscf',
    )
}
kwargs_ph.pop('extra_header_lines')
# Override for pw2bgw
kwargs_pp = dict(**kwargs_qe_base)

## VARIABLES
spinorbit = dict(
    system = dict(
        noncolin = soc,
        lspinorb = soc,
    )
)
variables_co = dict(
    control = dict(
        verbosity = 'high'
    ),
    system = dict(
        nosym = True
    ),
    electrons = dict(
        electron_maxstep = 200,
        diagonalization = 'rmm-davidson',
        diago_rmm_conv = True,
    )
)
variables_fi = dict(
    system = dict(
        occupations = 'smearing',
        smearing = 'gaussian',
        degauss = 0.001,
        nbnd = nbnd_fi,
        nosym = True,
        ecfixed = ecfixed,
        noncolin = socph,
        lspinorb = socph,
    ),
    control = dict(
        tprnfor = True
    )
)



# %% TASKS

# ============================================================================
#                              QUANTUM ESPRESSO
# ============================================================================

# ==========================================
# SCF :: compute ground-state charge density
scftask = QeScfTask(
    dirname = pjoin(workflow.dirname,'scf'),
    mpirun_jobname = name('scf'),
    kshift = [0,0,0],
    autokpt = True,
    ngkpt = ngkpt_scf,
    variables = spinorbit,
    **{**kwargs, **kwargs_qe_base, **kwargs_sym, **resources_kwargs('scf', 'qe')}
)

# ==============================================================
# WFN_FI :: wavefunctions and eigenvalues on fine k-shifted grid
wfntask_fi = QeScfTask(
    dirname = pjoin(workflow.dirname, 'wfn_fi'),
    mpirun_jobname = name('wfn_fi'),
    ngkpt = ngkpt_fi,
    variables = variables_fi,
    **{**kwargs, **kwargs_qe_base, **resources_kwargs('wfn_fi', 'qe')}
)
# convert wfn_fi to BGW
pw2bgwtask_fi = Qe2BgwTask(
    dirname = wfntask_fi.dirname,
    mpirun_jobname = name('wfn_fi-pp'),
    wfn_fname = 'WFN_fi',
    ngkpt = ngkpt_fi,
    dependencies = [wfntask_fi],
    **{**kwargs, **kwargs_pp, **resources_kwargs('wfn_fi-pp', 'pw2bgw')}
)
remove_wfc(pw2bgwtask_fi)

# ============================
# PHONON :: phonon calculation
link_wfn_fi = split_block(f"""\
cd {prefix}.save
ln -nfs ../../wfn_fi/{prefix}.save/*.hdf5 .
ln -nfs ../../wfn_fi/{prefix}.save/*.upf .
cd ..

cp ../wfn_fi/{prefix}.save/data-file-schema.xml ./{prefix}.save/
cp ../wfn_fi/{prefix}.xml .

""")
phr_link = split_block(f"""\
cp -r ../ph0/_ph0 .
cp ../pha/_ph0/{prefix}.phsave/dynmat.1.*.xml ./_ph0/{prefix}.phsave/
cp ../phb/_ph0/{prefix}.phsave/dynmat.1.*.xml ./_ph0/{prefix}.phsave/
cp ../phc/_ph0/{prefix}.phsave/dynmat.1.*.xml ./_ph0/{prefix}.phsave/
cp ../pha/_ph0/{prefix}.phsave/elph.1.*.xml ./_ph0/{prefix}.phsave/
cp ../phb/_ph0/{prefix}.phsave/elph.1.*.xml ./_ph0/{prefix}.phsave/
cp ../phc/_ph0/{prefix}.phsave/elph.1.*.xml ./_ph0/{prefix}.phsave/
""")

nat = len(structure.species)
ph_extra_header_lines = qe_header_lines([''] + link_wfn_fi)

phtask_a = QePhTask(
    dirname = pjoin(workflow.dirname, 'pha'),
    mpirun_jobname = name(f'pha'),
    start_irr = nat*3*0//3+1,
    last_irr = nat*3*1//3,
    xq = [0.,0.,0.],
    nogg = True,                                                                    #True only if you want to do (electronic) gamma only
    extra_header_lines = ph_extra_header_lines,
    dependencies = [pw2bgwtask_fi],
    **{**kwargs, **kwargs_ph, **resources_kwargs('phonon', 'qe')}
)
phtask_b = QePhTask(
    dirname = pjoin(workflow.dirname, 'phb'),
    mpirun_jobname = name(f'phb'),
    start_irr = nat*3*1//3+1,
    last_irr = nat*3*2//3,
    xq = [0.,0.,0.],
    nogg = True,
    extra_header_lines = ph_extra_header_lines,
    dependencies = [pw2bgwtask_fi],
    **{**kwargs, **kwargs_ph, **resources_kwargs('phonon', 'qe')}
)
phtask_c = QePhTask(
    dirname = pjoin(workflow.dirname, 'phc'),
    mpirun_jobname = name(f'phc'),
    start_irr = nat*3*2//3+1,
    last_irr = nat*3*3//3,
    xq = [0.,0.,0.],
    nogg = True,
    extra_header_lines = ph_extra_header_lines,
    dependencies = [pw2bgwtask_fi],
    **{**kwargs, **kwargs_ph, **resources_kwargs('phonon', 'qe')}
)

phtask_0 = QePhTask(
    dirname = pjoin(workflow.dirname, 'ph0'),
    mpirun_jobname = name(f'ph0'),
    start_irr = 0,
    last_irr = 0,
    xq = [0.,0.,0.],
    nogg = True,
    extra_header_lines = ph_extra_header_lines,
    dependencies = [pw2bgwtask_fi],
    **{**kwargs, **kwargs_ph, **resources_kwargs('phonon', 'qe')}
)

phtask_r = QePhTask(
    dirname = pjoin(workflow.dirname,'phr'),
    mpirun_jobname = name(f'phr'),
    recover = True,
    xq = [0.,0.,0.],
    nogg = True,
    extra_header_lines = qe_header_lines([''] + link_wfn_fi + phr_link),
    dependencies = [phtask_0, phtask_a, phtask_b, phtask_c],
    **{**kwargs, **kwargs_ph, **resources_kwargs('phonon', 'qe')}
)

# ====================== #
kwargs['ngkpt'] = ngkpt_co
# ====================== #

# ======================================================
# WFN :: wavefunctions and eigenvalues on a shifted grid
wfntask_ksh = QeWfnTask(
    dirname = pjoin(workflow.dirname,'wfn'),
    mpirun_jobname = name('wfn'),
    charge_density_fname = scftask.charge_density_fname,
    data_file_fname = scftask.data_file_fname,
    kshift = kshift,
    nbnd = nbnd,
    variables = variables_co | spinorbit,
    dependencies = [scftask],
    **{**kwargs, **kwargs_qe_base, **resources_kwargs('wfn', 'qe')}
)

pw2bgwtask_ksh = Qe2BgwTask(
    dirname = wfntask_ksh.dirname,
    mpirun_jobname = name('wfn-pp'),
    wfn_fname = 'WFN',
    kshift = kshift,
    dependencies = [wfntask_ksh],
    **{**kwargs, **kwargs_pp, **resources_kwargs('wfn-pp', 'pw2bgw')}
)
remove_wfc(pw2bgwtask_ksh)

# ==========================================
# WFNQ :: wavefunctions and eigenvalues on k+q-shifted grid
wfntask_qsh = QeWfnTask(
    dirname = pjoin(workflow.dirname, 'wfnq'),
    mpirun_jobname = name('wfnq'),
    charge_density_fname = scftask.charge_density_fname,
    data_file_fname = scftask.data_file_fname,
    kshift = kshift,
    qshift = qshift,
    variables = variables_co | spinorbit,
    dependencies = [scftask],
    **{**kwargs, **kwargs_qe_base, **resources_kwargs('wfnq', 'qe')}
)
# convert wfnq to BGW
pw2bgwtask_qsh = Qe2BgwTask(
    dirname = wfntask_qsh.dirname,
    mpirun_jobname = name('wfnq-pp'),
    wfn_fname = 'WFNq',
    kshift = kshift,
    qshift = qshift,
    dependencies = [wfntask_qsh],
    **{**kwargs, **kwargs_pp, **resources_kwargs('wfnq-pp', 'pw2bgw')}
)
remove_wfc(pw2bgwtask_qsh)

# ============================================================
# WFN_CO :: wavefunctions and eigenvalues on an unshifted grid
#If needed to add something thats in kwags, use kwargs.pop('parameter',None) then re-set it later use kwards['nbnd'] = ...
wfntask_ush = QeWfnTask(
    dirname = pjoin(workflow.dirname,'wfn_co'),
    mpirun_jobname = name('wfn_co'),
    charge_density_fname = scftask.charge_density_fname,
    data_file_fname = scftask.data_file_fname,
    nbnd = nbnd,
    variables = variables_co | spinorbit,
    dependencies = [scftask],
    **{**kwargs, **kwargs_qe_base, **resources_kwargs('wfn_co', 'qe')}
)
pw2bgwtask_ush = Qe2BgwTask(
    dirname = wfntask_ush.dirname,
    mpirun_jobname = name('wfn_co-pp'),
    wfn_fname = 'WFN_co',
    vxc_flag = True,
    rho_fname = 'RHO',
    rhog_flag = True,
    vxc_diag_nmax = nbnd,
    dependencies = [wfntask_ush],
    **{**kwargs, **kwargs_pp, **resources_kwargs('wfn_co-pp', 'pw2bgw')}
)
remove_wfc(pw2bgwtask_ush)



# ============================================================================
#                                    BGW
# ============================================================================

# ================================================================
# EPSILON :: Dielectric matric computation and inversion (epsilon)
epsilontask = EpsilonTask(
    dirname = pjoin(workflow.dirname,'epsilon'),
    mpirun_jobname = name('epsilon'),
    
    qshift = qshift,
    ecuteps = ecuteps,
    
    wfn_fname  = pw2bgwtask_ksh.wfn_fname,
    wfnq_fname = pw2bgwtask_qsh.wfn_fname,
    
    extra_lines = ['frequency_dependence 3','degeneracy_check_override'],
    dependencies = [pw2bgwtask_ksh, pw2bgwtask_qsh],
    **{**kwargs, **kwargs_bgw_base, **resources_kwargs('epsilon', 'bgw')}
)

# ================================
# SIGMA :: Self-energy calculation
sigmatask = SigmaTask(
    dirname = pjoin(workflow.dirname, 'sigma'),
    mpirun_jobname = name('sigma'),
    
    ibnd_min = band_index_min,
    ibnd_max = band_index_max,
    
    wfn_co_fname  = pw2bgwtask_ush.wfn_fname,
    rho_fname     = pw2bgwtask_ush.rho_fname,
    vxc_dat_fname = pw2bgwtask_ush.vxc_dat_fname,
    eps0mat_fname = epsilontask.eps0mat_fname,
    epsmat_fname  = epsilontask.epsmat_fname,
    
    extra_lines = [screening_type, 
                   'frequency_dependence 3',
                   'degeneracy_check_override', 
                   'no_symmetries_q_grid'],
    extra_variables = {'screened_coulomb_cutoff' : ecutsig},
    dependencies = [epsilontask, pw2bgwtask_ush],
    **{**kwargs, **kwargs_bgw_base, **resources_kwargs('sigma', 'bgw')}
)


# ============================
# KERNEL :: Kernel calculation
kerneltask = KernelTask(
    dirname = pjoin(workflow.dirname,'kernel'),
    mpirun_jobname = name('kernel'),
    
    wfn_co_fname  = pw2bgwtask_ush.wfn_fname,
    eps0mat_fname = epsilontask.eps0mat_fname,
    epsmat_fname  = epsilontask.epsmat_fname,
    
    extra_lines = [screening_type, sym_coarse_grid],
    dependencies = [epsilontask, pw2bgwtask_ush],
    **{**kwargs, **kwargs_bgw_base, **resources_kwargs('kernel', 'bgw')}
)

# =========================
# ABSORPTION :: Solving BSE
absorptiontask = AbsorptionTask(
    dirname = pjoin(workflow.dirname,'absorption'),
    mpirun_jobname = name('absorption'),
    
    wfn_co_fname  = pw2bgwtask_ush.wfn_fname,
    wfn_fi_fname  = pw2bgwtask_fi.wfn_fname,
    eps0mat_fname = epsilontask.eps0mat_fname,
    epsmat_fname  = epsilontask.epsmat_fname,
    bsemat_fname  = kerneltask.bsemat_fname,
    eqp_fname     = sigmatask.eqp1_fname,

    use_operator = use_operator,
    extra_variables = {'energy_resolution' : 0.03,
                       'write_eigenvectors' : -1},
    extra_lines = [sym_coarse_grid, sym_fine_grid, screening_type,
                   'eqp_co_corrections',
                   'gaussian_broadening',
                   'degeneracy_check_override',
                   'diagonalization'
                   ],
    dependencies = [pw2bgwtask_fi, kerneltask, sigmatask],
    **{**kwargs, **kwargs_bgw_base, **resources_kwargs('absorption', 'bgw')}
)


# ================================================================
# EXCITED STATE FORCES
esftask = ESFTask(
    dirname = pjoin(workflow.dirname, 'esf_files'),
    mpirun_jobname = name('esf'),
    
    structure = structure,
    prefix = prefix,
    eqp_fname = pjoin(absorptiontask.dirname, 'eqp.dat'),
    exciton_fname = pjoin(absorptiontask.dirname, 'eigenvectors.h5'),
    el_ph_dir = pjoin(phtask_r.dirname, f'_ph0/{prefix}.phsave'),
    dynmat_fname = pjoin(phtask_r.dirname, 'dyn'),
    scf_out_fname = pjoin(wfntask_fi.dirname, 'scf.out'),
    
    excited_forces_script = loc_excited_forces,
    rand_disp_script = loc_rand_disp_finite_temp,
    harmonic_script = loc_harmonic_extrapolation,
    first_iteration = first_iteration,
    
    extra_header_lines = qe_header_lines(),
    dependencies = [absorptiontask, phtask_r],
    **resources_kwargs('esf', 'qe'),
)

# =======================================================
# WORKFLOW OUTPUT
workflow.add_tasks([scftask,
                    wfntask_fi, pw2bgwtask_fi,
                    phtask_a, phtask_b, phtask_c, phtask_0, phtask_r,
                    wfntask_ksh, pw2bgwtask_ksh,
                    wfntask_qsh, pw2bgwtask_qsh,
                    wfntask_ush, pw2bgwtask_ush,
                    epsilontask, 
                    sigmatask,
                    kerneltask,
                    absorptiontask, 
                    esftask],
                    merge=False)


workflow.write()
workflow.write_dependencies()
