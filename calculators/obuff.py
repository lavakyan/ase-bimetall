from __future__ import print_function
import numpy as np
import openbabel
from ase import Atom, Atoms
from ase.io import write
from ase.calculators.calculator import Calculator
from ase import units
import sys


class OBUFF(Calculator):
    """Open Babel Universal Force Filed (UFF) calculator.
    Using openbabel.py bindings. """
    implemented_properties = ['energy', 'forces']

    def __init__(self, filename='/tmp/obUFF_atoms.xyz', logfile=None):
        """ Open Babel Universal Force Filed (UFF) calculator.
        Parameters:
        filename: path to a file where cooridnates will be stored to transfer to OB

        Note:
          transfer throught file is slow, but required sicne manual formation
          of atoms system in OB python environment requires explicit notation
          of bonds.. TODO: get away from file-based transfer.
        """
        import ase.parallel  # logging copied from ase.md.logger.MDLogger
        if ase.parallel.rank > 0:
            self.logfile = None  # Only log on master
        if logfile == '-':
            self.logfile = sys.stdout
            self.ownlogfile = False
        elif hasattr(logfile, 'write'):
            self.logfile = logfile
            self.ownlogfile = False
        elif not (logfile is None):
            self.logfile = open(logfile, 'w')
            self.ownlogfile = True
        else:
            self.logfile = None

        self.filename = filename
        self.file_format = 'xyz'  #TODO: auto-detect format?
        self.natoms = 0
        self.nbonds = 0
        #self.nrotat = 0
        #self.ndihed = 0

        self.mol = None
        self.ff = 0
        openbabel.OBConversion()  # this magic function should be called before FF search...
        self.ff = openbabel.OBForceField.FindForceField('UFF')  # TODO: allow to use of other FF ?
        self.name = 'OB_UFF'
        if (self.ff == 0):
            print('Could not find forcefield')
        self.ff.SetLogLevel(openbabel.OBFF_LOGLVL_HIGH)

        Calculator.__init__(self)

    def _shift_OBvec(self, vec, iaxis, d):
        if iaxis == 0:
            x = vec.GetX()
            vec.SetX( x + d )
        elif iaxis == 1:
            y = vec.GetY()
            vec.SetY( y + d )
        elif iaxis == 2:
            z = vec.GetZ()
            vec.SetZ( z + d )
        return vec

    def calculate(self, atoms, properties, system_changes):
        # call parent method:
        Calculator.calculate(self, atoms, properties, system_changes)

        if not(system_changes is None):
            self.update_atoms()

        if 'energy' in properties:
            energy = self.ff.Energy()
            # energy units are kcal/mole according to http://forums.openbabel.org/Energy-units-td1574278.html
            # and in Avogadro they are reported as kJ/mole ...
            self.results['energy'] = energy * units.kcal / units.mol

        if 'forces' in properties:
            d = 0.001
            F_ai = np.zeros( (self.natoms, 3) )
            for iatom in xrange( self.natoms ):
                for iaxis in xrange( 3 ):
                    obatom = self.mol.GetAtom( 1 + iatom )
                    vec = obatom.GetVector()

                    obatom.SetVector( self._shift_OBvec(vec, iaxis, d) )
                    self.ff.Setup( self.mol )
                    eplus = self.ff.Energy() * units.kcal/units.mol

                    obatom.SetVector( self._shift_OBvec(vec, iaxis, -2*d) )
                    self.ff.Setup( self.mol )
                    eminus = self.ff.Energy() * units.kcal/units.mol

                    obatom.SetVector( self._shift_OBvec(vec, iaxis, d) ) # put it back
                    F_ai[iatom, iaxis] = (eminus - eplus) / (2 * d)
            self.results['forces'] = F_ai

    def update_atoms(self):
        write(self.filename, self.atoms)

        # initialize OBMolecule
        self.mol = openbabel.OBMol()
        # read atoms from file:
        conv = openbabel.OBConversion()
        #format = conv.FormatFromExt(self.filename)
        conv.SetInAndOutFormats(self.file_format, self.file_format)
        conv.ReadFile(self.mol, self.filename)

        # attach OBmolecule to OBforcefield
        if self.ff.Setup( self.mol ) == 0:
            print('Could not setup forcefield')

        # log changes in topology:
        natoms = self.mol.NumAtoms()
        if (self.natoms != natoms):
            if self.natoms > 0:
                if not(self.logfile is None):
                    self.logfile.write('WARNING! Changed number of atoms!\n')
                else:
                    print('WARNING: changed number of atoms!\n')
            self.natoms = natoms
            if not(self.logfile is None):
                self.logfile.write('OB> Number of atoms:\t%i\n' % natoms)

        nbonds = self.mol.NumBonds()
        if (self.nbonds != nbonds):
            self.nbonds = nbonds
            if not(self.logfile is None):
                self.logfile.write( 'OB> Number of bonds:\t%i\n' % nbonds)
                self.logfile.flush()

    def set_atoms(self, atoms):
        """ put data in OB object """
        self.atoms = atoms        # .copy() ?
        self.update_atoms()


if __name__ == '__main__':
    from ase import Atoms
    from ase.structure import molecule
    from ase.visualize import view

    atoms = molecule('CH3')
    #~ from ase.io import read
    #~ atoms = read( 'alPtO2 1layer PBE opt.pdb' )
    calc = OBUFF(logfile='-')
    atoms.set_calculator(calc)
    print ('  Energy %f eV'% atoms.get_potential_energy())
    print (atoms.get_forces())

