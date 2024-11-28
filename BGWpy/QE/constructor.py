from .pwscfinput import PWscfInput


def get_scf_input(prefix, pseudo_dir, pseudos, structure, ecutwfc, kpts, wtks, kpts_option):
    """Construct a Quantum Espresso scf input."""
    inp = PWscfInput()
    
    inp.control.update(
        prefix = prefix,
        pseudo_dir = pseudo_dir,
        calculation = 'scf',
        )
    
    inp.electrons.update(
        electron_maxstep = 100,
        conv_thr = 1.0e-10,
        mixing_mode = 'plain',
        mixing_beta = 0.7,
        mixing_ndim = 8,
        diagonalization = 'rmm-davidson', # Pierre : changed default to rmm-davidson to avoid memory issues
        diago_david_ndim = 2, # Pierre : changed from 4 to 2 but not used with rmm-davidson
        diago_full_acc = True,
        )
    
    inp.system['ecutwfc'] = ecutwfc,
    inp.set_kpoints_crystal(kpts, wtks, kpts_option)
    inp.structure = structure
    inp.pseudos = pseudos

    return inp


def get_bands_input(prefix, pseudo_dir, pseudos, structure, ecutwfc, kpts, wtks, kpts_option, nbnd=None):
    """Construct a Quantum Espresso bands input."""
    inp = get_scf_input(prefix, pseudo_dir, pseudos, structure, ecutwfc, kpts, wtks, kpts_option)
    inp.control['calculation'] = 'bands'
    if nbnd is not None:
        inp.system['nbnd'] = nbnd
    return inp



