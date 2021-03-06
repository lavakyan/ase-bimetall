from __future__ import print_function
import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import convert_string_to_fd

from asap3 import EMT
from asap3 import BrennerPotential

class Hybrid(Calculator):
    """Hybrid MM calculator. EMT for particle and Brenner for support. Interaction - ??"""
    implemented_properties = ['energy', 'forces']

    #def __init__(self, np_selection, sp_selection, np_calc = EMT(), sp_calc = , vacuum=None):
    def __init__(self, np_species = ['Pt', 'Cu'], sp_species = ['C'], np_calc = EMT(), sp_calc = BrennerPotential(), inter_calc=None, vacuum=None, txt='-'):
        """ Hybrid calculator, mixing two many-body calculators, i.e. EMT and Brenner.
        The first is good for description of metal nanoparticle,
        and the second is used for carbon support or matrix dynamics.

        The energy is calculated as::

                     _                 _
            E = E   (R    ) + E       (R      ) + ?interaction?
                 EMT  nano     Brenner  Carbon

        parameters:

        vacuum: float or None
            Amount of vacuum to add around QM atoms.  Use None if QM
            calculator doesn't need a box.

        """
        self.np_species = np_species
        self.sp_species = sp_species

        self.np_calc = np_calc
        self.sp_calc = sp_calc
        self.ia_calc = inter_calc
        self.vacuum = vacuum

        self.np_selection = []
        self.sp_selection = []
        self.np_atoms = None
        self.sp_atoms = None
        self.center = None

        #self.name = '{0}+{1}'.format(np_calc.name, sp_calc.name)
        self.name = 'hybrid'

        Calculator.__init__(self)

    def initialize_np(self, atoms):
        constraints = atoms.constraints
        atoms.constraints = []
        self.np_atoms = atoms[self.np_selection]
        atoms.constraints = constraints
        self.np_atoms.pbc = False
        if self.vacuum:
            self.np_atoms.center(vacuum=self.vacuum)
            self.center = self.np_atoms.positions.mean(axis=0)
        self.np_atoms.set_calculator(self.np_calc)

    def initialize_sp(self, atoms):
        constraints = atoms.constraints
        atoms.constraints = []
        self.sp_atoms = atoms[self.sp_selection]
        atoms.constraints = constraints
        self.sp_atoms.pbc = False
        if self.vacuum:
            self.sp_atoms.center(vacuum=self.vacuum)
            self.center = self.sp_atoms.positions.mean(axis=0)
        self.sp_atoms.set_calculator(self.sp_calc)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.np_atoms is None:
            self.initialize_np(atoms)

        if self.sp_atoms is None:
            self.initialize_sp(atoms)

        energy = 0
        forces = np.zeros(shape=(len(atoms), 3))
        ## 1. Nanoparticle
        self.np_atoms.positions = atoms.positions[self.np_selection]
        if self.vacuum:
            self.np_atoms.positions += (self.center -
                                       self.np_atoms.positions.mean(axis=0))
        energy += self.np_calc.get_potential_energy(self.np_atoms)
        np_forces = self.np_calc.get_forces(self.np_atoms)
        if self.vacuum:
            np_forces -= np_forces.mean(axis=0)
        forces[self.np_selection] += np_forces
        ## 2. Support. Copy-Paste. TODO: make loop over subsystems
        self.sp_atoms.positions = atoms.positions[self.sp_selection]
        if self.vacuum:
            self.sp_atoms.positions += (self.center -
                                       self.sp_atoms.positions.mean(axis=0))
        energy += self.sp_calc.get_potential_energy(self.sp_atoms)
        sp_forces = self.sp_calc.get_forces(self.sp_atoms)
        if self.vacuum:
            sp_forces -= sp_forces.mean(axis=0)
        forces[self.sp_selection] += sp_forces
        ## 3. INTERACTION
        if self.ia_calc is not None:
            ienergy, inpforces, ispforces = self.ia_calc.calculate(
                self.np_atoms, self.sp_atoms, (0, 0, 0))
            forces[self.np_selection] += inpforces
            forces[self.sp_selection] += ispforces
            energy += ienergy
        ## 4. return result
        self.results['energy'] = energy
        self.results['forces'] = forces

    def set_atoms(self, atoms):
        """ calculate selections based on NP and support species ingfo """
        if (self.np_species and self.sp_species):   # not empty lists
            symbols = np.array(atoms.get_chemical_symbols())
            self.np_selection = (symbols == '') # all are False
            for spec in self.np_species:
                self.np_selection += (symbols==spec)
            self.sp_selection = (symbols == '') # all are False
            for spec in self.sp_species:
                self.sp_selection += (symbols==spec)
        if (self.sp_selection and self.np_selection):
            #if self.np_atoms is None:
            self.initialize_np(atoms)
            #if self.sp_atoms is None:
            self.initialize_sp(atoms)
        else:
            print('Warning: NO SELECTION!')


#~ class LJInteractions:
    #~ name = 'LJ'
#~
    #~ def __init__(self, parameters):
        #~ """Lennard-Jones type explicit interaction.
        #~ Heavily based on /ase/calculators/qmmm.py version
#~
        #~ parameters: dict
            #~ Mapping from pair of atoms to tuple containing epsilon, sigma
            #~ and cutoff radius for that pair.
#~
        #~ Example::
            #~ lj = LJInteractions({('Ar', 'Ar'): (eps, sigma, cutoff)})
#~
        #~ """
        #~ self.parameters = {}
        #~ for (symbol1, symbol2), (epsilon, sigma, cutoff) in parameters.items():
            #~ Z1 = atomic_numbers[symbol1]
            #~ Z2 = atomic_numbers[symbol2]
            #~ self.parameters[(Z1, Z2)] = epsilon, sigma, cutoff
            #~ self.parameters[(Z2, Z1)] = epsilon, sigma, cutoff
#~
    #~ def calculate(self, qmatoms, mmatoms, shift):
        #~ qmforces = np.zeros_like(qmatoms.positions)
        #~ mmforces = np.zeros_like(mmatoms.positions)
        #~ species = set(mmatoms.numbers)
        #~ energy = 0.0
        #~ for R1, Z1, F1 in zip(qmatoms.positions, qmatoms.numbers, qmforces):
            #~ for Z2 in species:
                #~ if (Z1, Z2) not in self.parameters:
                    #~ continue
                #~ epsilon, sigma, cutoff = self.parameters[(Z1, Z2)]
                #~ mask = (mmatoms.numbers == Z2)
                #~ D = mmatoms.positions[mask] + shift - R1
                #~ wrap(D, mmatoms.cell.diagonal(), mmatoms.pbc)
                #~ d2 = (D**2).sum(1)
                #~ if d2 > cutoff**2:
                    #~ continue
                #~ c6 = (sigma**2 / d2)**3
                #~ c12 = c6**2
                #~ energy += 4 * epsilon * (c12 - c6).sum()
                #~ f = 24 * epsilon * ((2 * c12 - c6) / d2)[:, np.newaxis] * D
                #~ F1 -= f.sum(0)
                #~ mmforces[mask] += f
        #~ return energy, qmforces, mmforces


if __name__ == '__main__':
    from ase import Atoms
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.structure import graphene_nanoribbon
    from ase.visualize import view

    surfaces = [(1,0,0), (1,1,0)]
    layers = [3,2]
    nano = Atoms(FaceCenteredCubic('Pt', surfaces, layers, 4.0))
    symbols = nano.get_chemical_symbols()
    symbols[0] = 'Cu'
    symbols[12] = 'Cu'
    nano.set_chemical_symbols(symbols)
    nano.pop(6)
    support = graphene_nanoribbon(8, 6, type='armchair', saturated=False)
    support.translate([-8,-2,-8])
    nano.extend(support)
    support.translate([0,-3.0,0])
    nano.extend(support)
    nano.center(vacuum=10)
    #~ view(nano)
    calc = Hybrid()
    nano.set_calculator(calc)
    # check subsystems selection
    from ase.io import write
    write('nanoparticle.pdb', calc.np_atoms)
    write('support.pdb', calc.sp_atoms)
    print ("Energy %f eV"% nano.get_potential_energy())

