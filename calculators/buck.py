import numpy as np
from itertools import count
from ase import Atom, Atoms
from ase.data import atomic_numbers
from ase.calculators.calculator import Calculator
from ase import units


COUL_COEF = 14.399651725922272

def _tup2str(tup):
    """ Convert tuple of atomic symbols to string
    i.e. ('Ce','O') -> 'Ce-O' """
    return '-'.join(tup)

def _str2tup(s):
    """ Parse string and return tuple of atomic symbols
    (not used atm) """
    return tuple(s.split('-'))

class Buck(Calculator):
    """Buckinham potential calculator"""
    implemented_properties = ['energy', 'forces']

    def __init__(self, parameters):
        """
            Calculator for Buckinham potential:
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
        self.parameters = {}
        for (symbol1, symbol2), (A, rho, C) in parameters.items():
            self.parameters[_tup2str((symbol1, symbol2))] = A, rho, C
            self.parameters[_tup2str((symbol2, symbol1))] = A, rho, C

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if not(system_changes is None):
            self.update_atoms()
        species = set(atoms.numbers)
        charges = [atom.charge for atom in atoms]

        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        chems = atoms.get_chemical_symbols()
        for i, R1, symb1, Q1 in zip(count(), atoms.positions, chems, charges):
            for j, R2, symb2, Q2 in zip(count(), atoms.positions, chems, charges):

                if j<=i:
                    continue

                pair = _tup2str((symb1,symb2))

                if pair not in self.parameters:
                    continue

                A, rho, C = self.parameters[pair]
                D = R1 - R2
                d2 = (D**2).sum()
                if (d2 > 0.001):
                    d = np.sqrt(d2)
                    coul = COUL_COEF*Q1*Q2/d
                    Aexp = A*np.exp(-d/rho)
                    Cr6 = C/d2**3
                    energy += Aexp - Cr6 + coul
                    force = (1/rho*Aexp - 6/d*Cr6 + coul/d)*D/d
                    forces[i, :] +=  force
                    forces[j, :] += -force

        if 'energy' in properties:
            self.results['energy'] = energy

        if 'forces' in properties:
            self.results['forces'] = forces

    def check_state(self, atoms, tol=1e-15):
        return True

    def update_atoms(self):
        pass

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.update_atoms()


if __name__ == '__main__':
    from ase import Atoms
    atoms = Atoms(symbols='CeO', cell=[2, 2, 5], positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4935832]]))
    atoms[0].charge = +4
    atoms[1].charge = -2
    calc = Buck({('Ce', 'O'): (1176.3, 0.381, 0.0)})
    #calc = Buck( {('Ce', 'O'): (0.0, 0.149, 0.0)} )
    atoms.set_calculator(calc)
    print('Epot = ', atoms.get_potential_energy())
    print('Force = ', atoms.get_forces())

    from ase.md.verlet import VelocityVerlet
    dyn = VelocityVerlet(atoms, dt=0.1*units.fs, trajectory='test.traj', logfile='-')
    dyn.run(1000)
    print('coodrs = ', atoms.get_positions())
    from ase.visualize import view
    view(atoms)




