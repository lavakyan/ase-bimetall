#!/usr/bin/python

import os, sys
from ase import Atom, Atoms
from ase.calculators.neighborlist import NeighborList

class Test:
    def __init__(self, atoms, log='-'):
        self.atoms = atoms
        self.chems = [] # chemical symbols
        self.CNs = []   # coordination numbers
        if isinstance(log, str):
            if log == '-':
                self.logfile = sys.stdout
            else:
                self.logfile = open(log, 'a')
        else:
            self.logfile = None

    def proc(self):
        atom = Atom('Au', position=[0,0,10])
        atoms.append(atom)

if __name__ == '__main__':
    from ase.cluster.cubic import FaceCenteredCubic
    print('\nTest monoatomic')
    #from ase.cluster.cubic import FaceCenteredCubic
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    max100 = 12
    max110 = 14
    max111 = 15
    a = 4.090  # Ag lattice constant
    layers = [max100, max110, max111]
    atoms = FaceCenteredCubic('Ag', surfaces, layers, latticeconstant=a)
    test = Test(atoms)
    test.proc()
    from ase.visualize import view
    view(atoms)
    print('** Finished **')
