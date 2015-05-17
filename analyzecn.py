"""
  Analyze clusters of atoms considering
  the coordination numbers (CN).
  Provides such parameters as
  averaged first-neighbor coordination number
  and Cowley's short-range order parameter
  [Phys. Rev. 1965, 138, A1384-A1389].
  Two functions (monoatomic and biatomic)
  are designed for study of sinlge component and
  two-component nanoparticles.
"""

#from ase import Atom, Atoms
from ase.calculators.neighborlist import NeighborList


def monoatomic(atoms, R1=3, calc_energy=False):
    r"""This routine analyzes atomic structure
    by the calculation of coordination numbers
    in cluster with only one type of atom.

    Parameters
    ----------
    atoms: ase.Atoms
        ase Atoms object, containing atomic cluster.
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
    N = len(atoms)
    nl = NeighborList([R1 / 2.0] * N, self_interaction=False, bothways=True)
    nl.build(atoms)
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
    atoms.center()
    R = atoms.positions.max() / 2.0
    if calc_energy:
        #from asap3 import EMT
        from ase.calculators.emt import EMT
        atoms.set_calculator(EMT())
        E = atoms.get_potential_energy()
    else:
        E = -1
    return N, R, CN, E, Ncore, CNshell


def biatomic(atoms, A, B, R1=3.0, calc_energy=False):
    r"""This routine analyzes atomic structure
    by the calculation of coordination numbers
    in cluster with atoms of two types (A and B).

    Parameters
    ----------
    atoms: ase.Atoms
        ase Atoms object, containing atomic cluster.
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
    N = len(atoms)
    nA = 0
    nB = 0
    for element in atoms.get_chemical_symbols():
        if element == A:
            nA += 1
        elif element == B:
            nB += 1
        else:
            raise Exception('Extra element ' + element)
    if (nA + nB != N):
        raise Exception('Number of A (' + str(nA) + ') ' +
          'and B (' + str(nB) + ') artoms mismatch!')
    nl = NeighborList([R1 / 2.0] * N, self_interaction=False, bothways=True)
    nl.build(atoms)

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
        if atoms[iatom].symbol == B:
            CN_BB_temp = 0
            CN_BA_temp = 0
            for ii in indeces:
                if atoms[ii].symbol == B:
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
                if atoms[i].symbol == A:
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
    R = atoms.positions.max() / 2.0
    if calc_energy:
        #from asap3 import EMT
        from ase.calculators.emt import EMT
        atoms.set_calculator(EMT())
        E = atoms.get_potential_energy()
    else:
        E = -1
    return N, nA, R, CN_AA, CN_AB, CN_BB, CN_BA, etha, E, NAcore, \
      NBcore, CNshellAA, CNshellAB, CNshellBB, CNshellBA

if __name__ == '__main__':
    print('\nTest biatomic')
    #import ase
    from ase.cluster.cubic import FaceCenteredCubic
    #from ase import Atoms, Atom
    atoms = FaceCenteredCubic(
      'Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7, 8, 7], 4.09)
    #atoms = ase.Atoms(atoms)
    from coreshell import CoreShellFCC
    #from ase.cluster.coreshell import CoreShellFCC
    CoreShellFCC(atoms, 'Pt', 'Ag', ratio=0.2, a_cell=4.09)

    #from ase.visualize import view
    #view(atoms)
    N, nA, R, CN_AA, CN_AB, CN_BB, CN_BA, etha, E, NAcore, \
      NBcore, CNshellAA, CNshellAB, CNshellBB, CNshellBA \
      = biatomic(atoms, 'Pt', 'Ag')
    print('N = {}'.format(N))
    print('nA = {}'.format(nA))
    print('nB = {}'.format((N - nA)))
    print('R = {}'.format(R))
    print('CN_AA = {}'.format(CN_AA))
    print('CN_AB = {}'.format(CN_AB))
    print('CN_BB = {}'.format(CN_BB))
    print('CN_BA = {}'.format(CN_BA))
    print('etha = {}'.format(etha))
    print(' E = {}'.format(E))
    print('NAcore = {}'.format(NAcore))
    print('NBcore = {}'.format(NBcore))
    print('CAcore = {}'.format(NAcore * 1.0 / nA))
    print('CBcore = {}'.format(NBcore * 1.0 / (N - nA)))
    print('CNshellAA = {}'.format(CNshellAA))
    print('CNshellAB = {}'.format(CNshellAB))
    print('CNshellBB = {}'.format(CNshellBB))
    print('CNshellBA = {}'.format(CNshellBA))
    N_, nA_, R_, CN_AA_, CN_AB_, CN_BB_, CN_BA_, etha_, E_, NAcore_, \
      NBcore_, CNshellAA_, CNshellAB_, CNshellBB_, CNshellBA_ \
      = biatomic(atoms, 'Ag', 'Pt')
    assert N == N_, 'Calculated N is not reflected upon A<->B'
    assert nA == N_ - nA_, 'Calculated nA is not reflected upon A<->B'
    assert CN_AA == CN_BB_, 'Calculated CN_AA is not reflected upon A<->B'
    assert CN_AB == CN_BA_, 'Calculated CN_AB is not reflected upon A<->B'
    assert CN_BB == CN_AA_, 'Calculated CN_BB is not reflected upon A<->B'
    assert CN_BA == CN_AB_, 'Calculated CN_BA is not reflected upon A<->B'
    assert CNshellAA == CNshellBB_, \
      'Calculated CNshellAA is not reflected upon A<->B'
    assert CNshellAB == CNshellBA_, \
      'Calculated CNshellAB is not reflected upon A<->B'
    assert CNshellBB == CNshellAA_, \
      'Calculated CNshellBB is not reflected upon A<->B'
    assert CNshellBA == CNshellAB_, \
      'Calculated CNshellBA is not reflected upon A<->B'
    print('** A<->B swap test passed **')
    #raw_input("Press enter")

    print('\nTest monoatomic')
    #from ase.cluster.cubic import FaceCenteredCubic
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    max100 = 12
    max110 = 14
    max111 = 15
    a = 4.090  # Ag lattice constant
    layers = [max100, max110, max111]
    atoms = FaceCenteredCubic('Ag', surfaces, layers, latticeconstant=a)
    N, R, CN, E, Ncore, CNshell = monoatomic(atoms)
    print('N \t R \t CN \t E \t Ncore \t C \t CNshell')
    print('{}\t{}\t{:.3f}\t{}\t{}\t{:.3f}\t{:.3f}'.format(
      N, R, CN, E, Ncore, (float(Ncore) / N), CNshell
    ))
    print('** Finished **')
