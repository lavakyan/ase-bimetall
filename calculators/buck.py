from __future__ import print_function
import numpy as np
from itertools import izip, count
from ase import Atom, Atoms
from ase.data import atomic_numbers
#from ase.io import write
from ase.calculators.calculator import Calculator
from ase import units
#import sys

COUL_COEF = 14.399651725922272 # coefficient in Coulumb energy
                               # units._e**2 / (4*pi*units._eps0) * units.m * units.J


class Buck(Calculator):
    """Buckinham potential calculator """
    implemented_properties = ['energy', 'forces']

    def __init__(self, parameters):
        """ Calculator for Buckinham potential:
          E_B = A exp(-r/rho) - C/r^6
        with Coulumb interaction:
          E_C = q_i * q_j e^2/(4pi eps_0 r^2)
        charges are read from atoms object

        Parameters:
        parameters: dict
            Mapping from pair of atoms to tuple containing A, rho and C
            for that pair. A in eV, rho in Angstr, C in eV/Angstr^2

        Example:
            calc = Buck({('O', 'O'): (A, rho, C)})
        """
        Calculator.__init__(self)
        # copied from ase.calculators.qmmm.LJInteractions
        self.parameters = {}
        for (symbol1, symbol2), (A, rho, C) in parameters.items():
            Z1 = atomic_numbers[symbol1]
            Z2 = atomic_numbers[symbol2]
            self.parameters[(Z1, Z2)] = A, rho, C
            self.parameters[(Z2, Z1)] = A, rho, C


    def calculate(self, atoms, properties, system_changes):
        # call parent method:
        Calculator.calculate(self, atoms, properties, system_changes)

        if not(system_changes is None):
            self.update_atoms()
        species = set(atoms.numbers)
        charges = [atom.charge for atom in atoms]

        energy = 0.0
        forces = np.zeros( (len(atoms), 3) )
        for i, R1, Z1, Q1 in izip(count(), atoms.positions, atoms.numbers, charges):
            for j, R2, Z2, Q2 in izip(count(), atoms.positions, atoms.numbers, charges):
                if j <= i:
                    continue
                # j > i

                if (Z1, Z2) not in self.parameters:
                    continue
                A, rho, C = self.parameters[(Z1, Z2)]
                D = R1 - R2
                #print(D)
                d2 = (D**2).sum()
                if (d2 > 0.001):
                    d = np.sqrt(d2)
                    #print(d)
                    coul = COUL_COEF*Q1*Q2/d
                    Aexp = A*np.exp(-d/rho)   # first term
                    Cr6  = C/d2**3            # second term: C/r^6
                    #~ print('coul\tAexp\tCr6')
                    #~ print(coul, Aexp, Cr6)
                    energy += Aexp - Cr6 + coul
                    # analytical gradient:
                    force = (1/rho*Aexp - 6/d*Cr6 + coul/d) * D/d # D is a vector
                    forces[i, :] +=  force
                    forces[j, :] += -force
                    #print(i, j, Q1, Q2, energy, force)
        if 'energy' in properties:
            self.results['energy'] = energy #* units.kJ / units.mol

        if 'forces' in properties:
            self.results['forces'] = forces

    def update_atoms(self):
        pass

    def set_atoms(self, atoms):
        """ ? """
        self.atoms = atoms        # .copy() ?
        self.update_atoms()


if __name__ == '__main__':
    from ase import Atoms
    atoms = Atoms(symbols='CeO',
          cell=[2,2,5],
          positions=np.array([[ 0.0,  0.0,  0.0],[ 0.0,  0.0,  2.4935832 ]])
    )
    atoms[0].charge = +4
    atoms[1].charge = -2
    #atoms.center(10)
    #~ from ase.visualize import view
    #~ view(atoms)
    calc = Buck( {('Ce', 'O'): (1176.3, 0.381, 0.0)} )
    #calc = Buck( {('Ce', 'O'): (0.0, 0.149, 0.0)} )
    atoms.set_calculator( calc )
    print('Epot = ', atoms.get_potential_energy())
    print('force = ', atoms.get_forces())
    #~ from ase.optimize import BFGS #MDMin
    #~ opt = BFGS(atoms) #MDMin(atoms)
    #~ opt.run(0.01)
    from ase.md.verlet import VelocityVerlet
    dyn = VelocityVerlet(atoms, 0.1, trajectory='test.traj', logfile='-')
    dyn.run(100)
    print('coords = ', atoms.get_positions())
    from ase.visualize import view
    view(atoms)

