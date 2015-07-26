"""
  Get geometrical structural properties
  of atom systems, such as:
  * coordination numbers
  * Cowley's short-range order parameter
  [Phys. Rev. 1965, 138, A1384-A1389].
  * QSAR volume, etc
"""

import os, sys
import numpy as np
from math import sqrt
from ase import Atom, Atoms
from ase.calculators.neighborlist import NeighborList

class QSAR:
    def __init__(self, atoms, log='-'):
        self.atoms = atoms.copy()
        self.atoms.center()
        self.chems = [] # chemical symbols
        self.CNs = []   # coordination numbers
        if isinstance(log, str):
            if log == '-':
                self.logfile = sys.stdout
            else:
                self.logfile = open(log, 'a')
        else:
            self.logfile = None

    def monoatomic(self, R1=3, calc_energy=False):
        r"""This routine analyzes atomic structure
        by the calculation of coordination numbers
        in cluster with only one type of atom.

        Parameters
        ----------
        R1: float
            First coordination shell will icnlude all atoms
            with distance less then R1 [Angstrom].
            Default value is 3.
        calc_energy: bool
            Flag used for calculation of potential energy with EMT
            calculator.  The default value is False, so that
            energy is not calculated.

        Returns
        -------
        N: int
            number of atoms in cluster
        R: float
            radius of the cluster
        CN: float
            average coord number
        E: float
            potential energy, -1 if calc_energy is False
        Ncore:
            number of atoms in core region (number of atoms with
            all 12 neighbors)
        CNshell:
            average coordination number for surface atoms only

        Notes
        -----
            The radius of the cluster is roughly determined as
            maximum the distance from the center to most distant atom
            in the cluster.

        Example
        --------
        >>> atoms = FaceCenteredCubic('Ag',
          [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7,8,7], 4.09)
        >>> [N, R, CN] = monoatomic(atoms)
        >>> print "average CN is ", CN
        """
        N = len(self.atoms)
        nl = NeighborList( [0.5 * R1] * N, self_interaction=False, bothways=True )
        nl.build(self.atoms)
        CN = 0
        Ncore = 0
        Nshell = 0
        CNshell = 0  # average CN of surface atoms
        for i in xrange(0, N):
            indeces, offsets = nl.get_neighbors(i)
            CN += len(indeces)
            if len(indeces) < 12:
                Nshell += 1
                CNshell += len(indeces)
            else:
                Ncore += 1
        CN = CN * 1.0 / N
        CNshell = CNshell * 1.0 / Nshell
        #atoms.center()
        R = self.atoms.positions.max() / 2.0
        if calc_energy:
            #from asap3 import EMT
            from ase.calculators.emt import EMT
            atoms.set_calculator(EMT())
            E = atoms.get_potential_energy()
        else:
            E = -1
        #return N, R, CN, E, Ncore, CNshell
        self.N = N
        self.R = R
        self.CN = CN
        self.E = E
        self.Ncore = Ncore
        self.CNshell = CNshell


    def biatomic(self, A, B, R1=3.0, calc_energy=False):
        r"""This routine analyzes atomic structure
        by the calculation of coordination numbers
        in cluster with atoms of two types (A and B).

        Parameters
        ----------
        A: string
            atom type, like 'Ag', 'Pt', etc.
        B: string
            atom type, like 'Ag', 'Pt', etc.
        R1: float
            First coordination shell will icnlude all atoms
            with distance less then R1 [Angstrom].
            Default value is 3.
        calc_energy: bool
            Flag used for calculation of potential energy with EMT
            calculator.  The default value is False, so that
            energy is not calculated.

        Returns
        -------
        N: int
            number of atoms in cluster
        nA:
            number of atoms of type A
        R: float
            radius of the cluster
        CN_AA: float
            average number of atoms A around atom A
        CN_AB: float
            average number of atoms A around atom B
        CN_BB: float
            average number of atoms B around atom B
        CN_BA: float
            average number of atoms B around atom A
        etha: float
            parameter of local ordering, -1 < etha < 1.
            Returns 999 if concentration of one of the
            component is too low.
        E: float
            potential energy
        NAcore:
            number of A atoms in core
        NBcore:
            number of B atoms in core
        CNshellAA:
            average CN of A-A for surface atoms only
        CNshellAB:
            average CN of A-B for surface atoms only
        CNshellBB:
            average CN of B-B for surface atoms only
        CNshellBA:
            average CN of B-A for surface atoms only

        Notes
        -----
            The radius of the cluster is roughly determined as
            maximum the distance from the center to most distant atom
            in the cluster.

        Example
        --------
        >>> atoms = FaceCenteredCubic('Ag',
          [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7,8,7], 4.09)
        >>> atoms = CoreShellFCC(atoms, 'Pt', 'Ag', 0.6, 4.09)
        >>> [N, nA, R, CN_AA, CN_AB, CN_BB, CN_BA, etha] =
          biatomic(atoms, 'Pt', 'Ag')
        >>> print "Short range order parameter: ", etha
        """
        N = len(self.atoms)
        nA = 0
        nB = 0
        for element in self.atoms.get_chemical_symbols():
            if element == A:
                nA += 1
            elif element == B:
                nB += 1
            else:
                raise Exception('Extra element ' + element)
        if (nA + nB != N):
            raise Exception('Number of A (' + str(nA) + ') ' +
              'and B (' + str(nB) + ') artoms mismatch!')
        nl = NeighborList([0.5 * R1] * N, self_interaction=False, bothways=True)
        nl.build(self.atoms)

        # initialize counters:
        CN_AA = 0    # averaged total coord. numbers
        CN_AB = 0
        CN_BB = 0
        CN_BA = 0
        NAcore = 0  # number of atoms in core region
        NBcore = 0
        CNshellAA = 0  # average coord. numbers for surface atoms
        CNshellAB = 0
        CNshellBB = 0
        CNshellBA = 0
        for iatom in xrange(0, N):
            indeces, offsets = nl.get_neighbors(iatom)
            if self.atoms[iatom].symbol == B:
                CN_BB_temp = 0
                CN_BA_temp = 0
                for ii in indeces:
                    if self.atoms[ii].symbol == B:
                        CN_BB_temp += 1
                    else:  # atoms[i].symbol == A :
                        CN_BA_temp += 1
                CN_BB += CN_BB_temp
                CN_BA += CN_BA_temp
                if len(indeces) < 12:
                    # SHELL
                    CNshellBB += CN_BB_temp
                    CNshellBA += CN_BA_temp
                else:
                    # CORE
                    NBcore += 1
            else:  # atoms[iatom].symbol == A :
                CN_AA_temp = 0
                CN_AB_temp = 0
                for i in indeces:
                    if self.atoms[i].symbol == A:
                        CN_AA_temp += 1
                    else:  # atoms[i].symbol==B :
                        CN_AB_temp += 1
                CN_AA += CN_AA_temp
                CN_AB += CN_AB_temp
                if len(indeces) < 12:
                    # SHELL
                    CNshellAA += CN_AA_temp
                    CNshellAB += CN_AB_temp
                else:
                    # CORE
                    NAcore += 1
        # averaging:
        CN_AA = CN_AA * 1.0 / nA
        CN_AB = CN_AB * 1.0 / nA
        CN_BB = CN_BB * 1.0 / nB
        CN_BA = CN_BA * 1.0 / nB
        znam = (nA - NAcore)
        if znam > 0.0001:
            CNshellAA = CNshellAA * 1.0 / znam
            CNshellAB = CNshellAB * 1.0 / znam
        else:
            CNshellAA = 0
            CNshellAB = 0
        znam = (nB - NBcore)
        if znam > 0.0001:
            CNshellBB = CNshellBB * 1.0 / znam
            CNshellBA = CNshellBA * 1.0 / znam
        else:
            CNshellBB = 0
            CNshellBA = 0

        # calc concentrations:
        concB = nB * 1.0 / N
        if concB < 0.0001:
            #print "WARNING! Too low B concentration: ",concB
            etha = 999
        else:
            etha = 1 - CN_AB / (concB * (CN_AA + CN_AB))
        R = self.atoms.positions.max() / 2.0
        if calc_energy:
            #from asap3 import EMT
            from ase.calculators.emt import EMT
            self.atoms.set_calculator(EMT())
            E = self.atoms.get_potential_energy()
        else:
            E = -1
        #return N, nA, R, CN_AA, CN_AB, CN_BB, CN_BA, etha, E, NAcore, \
        #  NBcore, CNshellAA, CNshellAB, CNshellBB, CNshellBA
        self.N = N
        self.nA = nA
        self.R = R
        self.CN_AA = CN_AA
        self.CN_AB = CN_AB
        self.CN_BB = CN_BB
        self.CN_BA = CN_BA
        self.etha = etha
        self.E = E
        self.NAcore = NAcore
        self.NBcore = NBcore
        self.CNshellAA = CNshellAA
        self.CNshellAB = CNshellAB
        self.CNshellBB = CNshellBB
        self.CNshellBA = CNshellBA

    def atom_distances(self, atom_type = 'all'):
        r"""This routine computes the radial distrubution function
        relative of the center of nanoparticle

        Parameters
        ----------
        atom_type: string
            atom type, like 'Ag', 'Pt', etc. Default is 'all'.

        Returns
        -------
        numpy.array() containing the distances of atoms from the center


        Example
        --------
        to be added
        """
        N = 0
        if atom_type == 'all':
            N = len(self.atoms)
        else:
            for i in xrange( len(self.atoms) ):
                if atoms[i].symbol == atom_type:
                    N = N +1
        assert( N>0 )
        xs = atoms.positions[:, 0]
        ys = atoms.positions[:, 1]
        zs = atoms.positions[:, 2]
        # centering at zero
        min_x = min( xs )
        max_x = max( xs )
        min_y = min( ys )
        max_y = max( ys )
        min_z = min( zs )
        max_z = max( zs )
        center_x = (min_x+max_x)/2.0
        center_y = (min_y+max_y)/2.0
        center_z = (min_z+max_z)/2.0
        for i in xrange( len(self.atoms) ):
            xs[i] = xs[i] - center_x
            ys[i] = ys[i] - center_y
            zs[i] = zs[i] - center_z

        Rs = np.zeros( N )
        for i in xrange(N):
            if atom_type == 'all':
                Rs[i] = xs[i]*xs[i]+ys[i]*ys[i]*zs[i]*zs[i]
            else:
                if atoms[i].symbol == atom_type:
                    Rs[i] = xs[i]*xs[i]+ys[i]*ys[i]*zs[i]*zs[i]
            Rs[i] = sqrt( Rs[i] )
        #pause
        return Rs
        #return xs*xs+ys*ys+zs*zs



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
    #from ase.visualize import view
    #view(atoms)
    qsar = QSAR(atoms)

    qsar.monoatomic()
    print('N \t R \t CN \t E \t Ncore \t C \t CNshell')
    print('{}\t{}\t{:.3f}\t{}\t{}\t{:.3f}\t{:.3f}'.format(
      qsar.N, qsar.R, qsar.CN, qsar.E, qsar.Ncore, (float(qsar.Ncore) / qsar.N), qsar.CNshell
    ))

    #exit(0)
    print('\nTest biatomic')
    #import ase

    #from ase import Atoms, Atom
    atoms = FaceCenteredCubic(
      'Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7, 8, 7], 4.09)
    from coreshell import CoreShellFCC

    CoreShellFCC(atoms, 'Pt', 'Ag', ratio=0.2, a_cell=4.09)

    from ase.visualize import view
    view(atoms)
    #N, nA, R, CN_AA, CN_AB, CN_BB, CN_BA, etha, E, NAcore, \
    #  NBcore, CNshellAA, CNshellAB, CNshellBB, CNshellBA \
    qsar = QSAR(atoms)
    qsar.biatomic('Pt', 'Ag')
    print('N = {}'.format(qsar.N))
    print('nA = {}'.format(qsar.nA))
    print('nB = {}'.format((qsar.N - qsar.nA)))
    print('R = {}'.format(qsar.R))
    print('CN_AA = {}'.format(qsar.CN_AA))
    print('CN_AB = {}'.format(qsar.CN_AB))
    print('CN_BB = {}'.format(qsar.CN_BB))
    print('CN_BA = {}'.format(qsar.CN_BA))
    print('etha = {}'.format(qsar.etha))
    print(' E = {}'.format(qsar.E))
    print('NAcore = {}'.format(qsar.NAcore))
    print('NBcore = {}'.format(qsar.NBcore))
    print('CAcore = {}'.format(qsar.NAcore * 1.0 / qsar.nA))
    print('CBcore = {}'.format(qsar.NBcore * 1.0 / (qsar.N - qsar.nA)))
    print('CNshellAA = {}'.format(qsar.CNshellAA))
    print('CNshellAB = {}'.format(qsar.CNshellAB))
    print('CNshellBB = {}'.format(qsar.CNshellBB))
    print('CNshellBA = {}'.format(qsar.CNshellBA))
    qsar_inv = QSAR(atoms)
    qsar_inv.biatomic('Ag', 'Pt')
    assert qsar.N == qsar_inv.N, 'Calculated N is not reflected upon A<->B'
    assert qsar.nA == qsar_inv.N - qsar_inv.nA, 'Calculated nA is not reflected upon A<->B'
    assert qsar.CN_AA == qsar_inv.CN_BB, 'Calculated CN_AA is not reflected upon A<->B'
    assert qsar.CN_AB == qsar_inv.CN_BA, 'Calculated CN_AB is not reflected upon A<->B'
    assert qsar.CN_BB == qsar_inv.CN_AA, 'Calculated CN_BB is not reflected upon A<->B'
    assert qsar.CN_BA == qsar_inv.CN_AB, 'Calculated CN_BA is not reflected upon A<->B'
    assert qsar.CNshellAA == qsar_inv.CNshellBB, \
      'Calculated CNshellAA is not reflected upon A<->B'
    assert qsar.CNshellAB == qsar_inv.CNshellBA, \
      'Calculated CNshellAB is not reflected upon A<->B'
    assert qsar.CNshellBB == qsar_inv.CNshellAA, \
      'Calculated CNshellBB is not reflected upon A<->B'
    assert qsar.CNshellBA == qsar_inv.CNshellAB, \
      'Calculated CNshellBA is not reflected upon A<->B'
    print('** A<->B swap test passed **')
    #raw_input("Press enter")

    print '# Radial distribution in NP'
    print '# All atoms'
    for value in qsar.atom_distances('all'):
         print value
    print '# Ag'
    for value in qsar.atom_distances('Ag'):
         print value
    #print '# Pt'
    #for value in qsar.atom_distances('Pt'):
    #     print value

    print('** Finished **')

