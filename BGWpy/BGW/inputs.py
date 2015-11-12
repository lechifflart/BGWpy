
from collections import OrderedDict

from ..core import BasicInputFile


class EpsilonInput(BasicInputFile):

    def __init__(self, ecuteps, nbnd, q0, qpts, *keywords, **variables):

        all_variables = OrderedDict([
            ('epsilon_cutoff' , ecuteps),
            ('number_bands' , nbnd),
            ])

        # band_occupation is deprecated, so I keep it as an optional variable
        nbnd_occ = variables.pop('nbnd_occ', None)
        if nbnd_occ:
            all_variables['band_occupation'] = '{}*1 {}*0'.format(
                                                nbnd_occ, nbnd-nbnd_occ)

        all_variables.update(variables)

        super(EpsilonInput, self).__init__(all_variables, keywords)

        self.q0 = q0
        self.qpts = qpts

    def __str__(self):

        qpt_block = '\nbegin qpoints\n'
        for q0i in self.q0:
            qpt_block += ' {:11.8f}'.format(q0i)
        qpt_block += ' 1.0 1\n'

        for q in self.qpts:
            for qi in q:
                qpt_block += ' {:11.8f}'.format(qi)
            qpt_block += ' 1.0 0\n'
        qpt_block += 'end\n'

        return super(EpsilonInput, self).__str__() + qpt_block


class SigmaInput(BasicInputFile):

    def __init__(self, ecuteps, ecutsigx, nbnd, ibnd_min, ibnd_max, kpts, 
                 *keywords, **variables):

        all_variables = OrderedDict([
            ('screened_coulomb_cutoff' , ecuteps),
            ('bare_coulomb_cutoff' , ecutsigx),
            ('number_bands' , nbnd),
            ('band_index_min' , ibnd_min),
            ('band_index_max' , ibnd_max),
            ])

        # band_occupation is deprecated, so I keep it as an optional variable
        nbnd_occ = variables.pop('nbnd_occ', None)
        if nbnd_occ:
            all_variables['band_occupation'] = '{}*1 {}*0'.format(
                                                nbnd_occ, nbnd-nbnd_occ)

        all_variables.update(variables)

        super(SigmaInput, self).__init__(all_variables, keywords)

        self.kpts = kpts

    def __str__(self):

        kpt_block = '\nbegin kpoints\n'
        for k in self.kpts:
            for ki in k:
                kpt_block += ' {:11.8f}'.format(ki)
            kpt_block += ' 1.0\n'
        kpt_block += 'end\n'

        return super(SigmaInput, self).__str__() + kpt_block


class KernelInput(BasicInputFile):

    def __init__(self, nbnd_val, nbnd_cond, ecuteps, ecutsigx,
                 *keywords, **variables):

        all_variables = OrderedDict([
            ('number_val_bands' , nbnd_val),
            ('number_cond_bands' , nbnd_cond),
            ('screened_coulomb_cutoff' , ecuteps),
            ('bare_coulomb_cutoff' , ecutsigx),
            ])

        all_variables.update(variables)

        super(KernelInput, self).__init__(all_variables, keywords)


class AbsorptionInput(BasicInputFile):

    def __init__(self, nbnd_val_co, nbnd_cond_co, nbnd_val_fi, nbnd_cond_fi,
                 *keywords, **variables):

        all_variables = OrderedDict([
            ('number_val_bands_coarse' , nbnd_val_co),
            ('number_val_bands_fine' , nbnd_val_fi),
            ('number_cond_bands_coarse' , nbnd_cond_co),
            ('number_cond_bands_fine' , nbnd_cond_fi),
            ('energy_resolution' , 0.1),
            ])

        all_variables.update(variables)

        super(AbsorptionInput, self).__init__(all_variables, keywords)

