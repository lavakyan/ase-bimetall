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
    def __init__(self, np_species = ['Pt', 'Cu'], sp_species=['C'], np_calc = EMT(), sp_calc = BrennerPotential(), vacuum=None):
        """ Hybrid calculator, mixing two many-body calculators: EMT and Brenner.
        The first is good ad description of metal nanoparticle,
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

        ## 3. return result
        self.results['energy'] = energy
        self.results['forces'] = forces

    def set_atoms(self, atoms):
        """ calculate selections based on NP and support species ingfo """
        symbols = np.array(atoms.get_chemical_symbols())
        self.np_selection = (symbols == '') # all are False
        for spec in self.np_species:
            self.np_selection += (symbols==spec)
        self.sp_selection = (symbols == '') # all are False
        for spec in self.sp_species:
            self.sp_selection += (symbols==spec)

        #if self.np_atoms is None:
        self.initialize_np(atoms)
        #if self.sp_atoms is None:
        self.initialize_sp(atoms)


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

