from __future__ import print_function
import numpy as np

import openbabel
from ase import Atom, Atoms
from ase.io import write
from ase.calculators.calculator import Calculator
from ase import units


class OBUFF(Calculator):
    """Open Babel Universal Force Filed (UFF) calculator.
    Using openbabel.py bindings. """
    implemented_properties = ['energy', 'forces']

    def __init__(self, filename='tmp_atoms.xyz'):
        """ Open Babel Universal Force Filed (UFF) calculator.
        Parameters:
        filename: path to a file where cooridnates will be stored to transfer to OB

        Note:
          transfer throught file is slow, but required sicne manual formation
          of atoms system in OB python environment requires explicit notation
          of bonds.. TODO: get away from file-based transfer.
        """
        self.filename = filename
        self.file_format = 'xyz'

        self.mol = None
        self.ff = 0
        openbabel.OBConversion()  # this magic function should be called before FF seaech...
        self.ff = openbabel.OBForceField.FindForceField("UFF")  # TODO: allow to use of other FF ?
        if (self.ff == 0):
            print("Could not find forcefield")
        self.ff.SetLogLevel(openbabel.OBFF_LOGLVL_HIGH)

        Calculator.__init__(self)

    def _shift_OBvec(self, vec, iaxis, d):
        if iaxis == 0:
            x = vec.GetX()
            vec.SetX(x+d)
        elif iaxis == 1:
            y = vec.GetY()
            vec.SetY(y+d)
        elif iaxis == 2:
            z = vec.GetZ()
            vec.SetZ(z+d)
        return vec

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            energy = self.ff.Energy()
            self.results['energy'] = energy * units.kcal / units.mol

        if 'forces' in properties:
            d = 0.001
            N = len(atoms)
            F_ai = np.zeros( (N, 3) )
            for iatom in xrange(N):
                for iaxis in xrange(3):
                    obatom = self.mol.GetAtom( 1 + iatom )
                    vec = obatom.GetVector()
                    obatom.SetVector( self._shift_OBvec(vec, iaxis, d) )
                    self.ff.Setup(self.mol)
                    eplus = self.ff.Energy() * units.kcal/units.mol
                    obatom.SetVector( self._shift_OBvec(vec, iaxis, -2*d) )
                    self.ff.Setup( self.mol )
                    eminus = self.ff.Energy() * units.kcal/units.mol
                    obatom.SetVector( self._shift_OBvec(vec, iaxis, d) ) # putin it back
                    F_ai[iatom, iaxis] = (eminus - eplus) / (2 * d)
            self.results['forces'] = F_ai

    def set_atoms(self, atoms):
        """ put data in OB object """
        self.atoms = atoms        # .copy() ?
        write(self.filename, self.atoms)

        # initialize OBMolecule
        self.mol = openbabel.OBMol()
        print('Atoms set')
        # read atoms from file:
        conv = openbabel.OBConversion()
        #format = conv.FormatFromExt(self.filename)
        conv.SetInAndOutFormats(self.file_format, self.file_format)
        conv.ReadFile(self.mol, self.filename)

        # attach OBmolecule to OBforcefield
        if self.ff.Setup( self.mol ) == 0:
            print("Could not setup forcefield")


if __name__ == '__main__':
    from ase import Atoms
    from ase.structure import molecule
    from ase.visualize import view

    atoms = molecule('CH3')
    calc = OBUFF()
    atoms.set_calculator(calc)
    print ("Energy %f eV"% atoms.get_potential_energy())
    print (atoms.get_forces())

