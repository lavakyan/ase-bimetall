""" This module contains the function
  to generate core-shell A@B nanoparticle with FCC structure.
"""
import random
import copy
from math import sqrt
from ase.calculators.neighborlist import NeighborList
from ase.cluster.cubic import FaceCenteredCubic

from qsar import QSAR

def sphericalFCC(elem, latticeconstant, nlayers):
    r""" Geneartes spherical cluster of atoms of FCC metal

    Parameters
    ----------
    elem: string
        symbol of chemical element.
    lattice constant: float
        lattice constant in Angstr.
    nlayers:
        number of atomic layers passed to FaceCenteredCubic. Guards the radius of cluster

    Returns
    -------
    ase.Atoms object

    Example
    --------
    >>> atoms = sphericalFCC('Ag', 4.09, 8)
    """
    # 1. generate cubical cluster
    surfaces = [(1, 0, 0)]
    layers = [nlayers]
    atoms = FaceCenteredCubic(elem, surfaces, layers, latticeconstant)
    atoms.center()
    # 2. cut al lextra atom from cube to make it spherical
    Xmin = atoms.positions[:, 0].min()
    Xmax = atoms.positions[:, 0].max()
    C =  (Xmin+Xmax)/2.0
    R = C
    ia = 0
    while ia < len(atoms):
        x2 = (atoms.positions[ia, 0] - C)**2
        y2 = (atoms.positions[ia, 1] - C)**2
        z2 = (atoms.positions[ia, 2] - C)**2
        if (x2 + y2 + z2) > R**2:
            del atoms[ia]
        else:
            ia += 1
    return atoms

def cut_spherical_cluster(atoms, size):
    r""" Cuts spherical cluster from provided atoms object

    Parameters
    ----------
    atoms: ASE.Atoms object
        the original cluster to be cut off
    size: float
        the diameter of resulting cluster, in Angstrom

    Returns
    -------
    ase.Atoms object of resulted cluster

    Example
    --------
    >>> atoms = cut_spherical_cluster(atoms, 10) # 1nm cluster
    """
    atoms = copy.copy(atoms) # keep original atoms unchanged
    atoms.center(0)
    Xmin = atoms.positions[:, 0].min()
    Xmax = atoms.positions[:, 0].max()
    Ymin = atoms.positions[:, 1].min()
    Ymax = atoms.positions[:, 1].max()
    Zmin = atoms.positions[:, 2].min()
    Zmax = atoms.positions[:, 2].max()
    Cx =  (Xmin+Xmax)/2.0
    Cy =  (Ymin+Ymax)/2.0
    Cz =  (Zmin+Zmax)/2.0
    R = size/2.0 # radius of cluster
    ia = 0
    while ia < len(atoms):
        x2 = (atoms.positions[ia, 0] - Cx)**2
        y2 = (atoms.positions[ia, 1] - Cy)**2
        z2 = (atoms.positions[ia, 2] - Cz)**2
        if (x2 + y2 + z2) > R**2:
            del atoms[ia]
        else:
            ia += 1
    return atoms



def CoreShellFCC(atoms, type_a, type_b, ratio, a_cell, n_depth=-1):
    r"""This routine generates cluster with ideal core-shell architecture,
    so that atoms of type_a are placed on the surface
    and atoms of type_b are forming the core of nanoparticle.
    The 'surface' of nanoparticle is defined as atoms
    with unfinished coordination shell.

    Parameters
    ----------
    atoms: ase.Atoms
        ase Atoms object, containing atomic cluster.
    type_a: string
        Symbol of chemical element to be placed on the shell.
    type_b: string
        Symbol of chemical element to be placed in the core.
    ratio: float
        Guards the number of shell atoms, type_a:type_b = ratio:(1-ratio)
    a_cell: float
        Parameter of FCC cell, in Angstrom.
        Required for calculation of neighbor distances in for infinite
        crystal.
    n_depth: int
        Number of layers of the shell formed by atoms ratio.
        Default value -1 is ignored and n_depth is calculated according
        ratio. If n_depth is set then value of ratio is ignored.

    Returns
    -------
        Function returns ASE atoms object which
        contains bimetallic core-shell cluster

    Notes
    -----
        The criterion of the atom beeing on the surface is incompletnes
        of it's coordination shell. For the most outer atoms the first
        coordination shell will be incomplete (coordination number
        is less then 12 for FCC), for the second layer --
        second coordination shell( CN1 + CN2 < 12 + 6) and so on.
        In this algorithm each layer is tagged by the number
        ('depth'), take care if used with other routines
        dealing with tags (add_adsorbate etc).

        First, atoms with unfinished first shell are replaced
        by atoms type_a, then second, and so on.
        The last depth surface layer is replaced by random
        to maintain given ratio value.

    Example
    --------
    >>> atoms = FaceCenteredCubic('Ag',
      [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7,8,7], 4.09)
    >>> atoms = CoreShellFCC(atoms, 'Pt', 'Ag', 0.6, 4.09)
    >>> view(atoms)
    """
    # 0 < ratio < 1
    target_x = ratio
    if n_depth != -1:
        target_x = 1  # needed to label all needed layeres

    def fill_by_tag(atoms, chems, tag):
        """Replaces all atoms within selected layer"""
        for i in xrange(0, len(atoms)):
            if atoms[i].tag == tag:
                chems[i] = type_a
        return
    # coord numbers for FCC:
    coord_nums = [1, 12, 6, 24, 12, 24, 8, 48, 6, 36, 24, 24, 24, 72, 48,
    12, 48, 30, 72, 24]
    # coordination radii obtained from this array as R = sqrt(coord_radii)*a/2
    coord_radii = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30,
    32, 34, 36, 38, 40]

    ## generate FCC cluster ##
    #atoms = FaceCenteredCubic(type_b, surfaces, layers, a_cell)
    n_atoms = len(atoms)
    ## tag layers ##
    positions = [0]  # number of positions in layer
    n_tag = 0    # number of tags to check if there is enought layers
    n_shell = 0  # depth of the shell
    while (n_tag < n_atoms * target_x):
        n_shell += 1
        if (n_depth != -1)and(n_shell > n_depth):
            break
        neiblist = NeighborList(
          [
            a_cell / 2.0 * sqrt(coord_radii[n_shell]) / 2.0 + 0.0001
          ] * n_atoms,
          self_interaction=False, bothways=True
        )
        neiblist.build(atoms)
        for i in xrange(0, n_atoms):
            indeces, offsets = neiblist.get_neighbors(i)
            if (atoms[i].tag == 0):
                if (len(indeces) < sum(coord_nums[1:n_shell + 1])):
                    # coord shell is not full -> atom is on surface!
                    atoms[i].tag = n_shell
                    n_tag += 1
        # save the count of positions at each layer:
        positions.append(n_tag - sum(positions[0:n_shell]))
    ## populate layers ##
    chems = atoms.get_chemical_symbols(reduce=False)
    n_type_a = 0  # number of changes B -> A
    if (n_tag < n_atoms * target_x)and(n_depth == -1):
        # raise exception?
        return None
    else:
        n_filled = n_shell - 1  # number of totally filled layers
        ilayer = 1
        while (ilayer < n_filled + 1):
            fill_by_tag(atoms, chems, ilayer)
            n_type_a += positions[ilayer]
            ilayer += 1
        while (n_type_a < n_atoms * target_x)and(n_depth == -1):
            i = random.randint(0, n_atoms - 1)
            if (atoms[i].tag == n_shell):
                if (chems[i] == type_b):
                    chems[i] = type_a
                    n_type_a += 1
    atoms.set_chemical_symbols(chems)
    ## check number of atoms ##
    checkn_a = 0
    for element in chems:
        if element == type_a:
            checkn_a += 1
    assert n_type_a == checkn_a
    return atoms


def hollowCore(atoms, a_cell, radius=3):
    r"""This routine generates cluster with
    empty core.

    Parameters
    ----------
    atoms: ase.Atoms
        ase Atoms object, containing atomic cluster.
    a_cell: float
        Parameter of FCC cell, in Angstrom.
        Required for calculation of neighbor distances in for infinite
        crystal.
    radius: int
        controlls the size of emty region in the center of cluster.

    Returns
    -------
        Function returns ASE atoms object which
        contains hollow cluster

    Notes
    -----

    Example
    --------
    >>> atoms = FaceCenteredCubic('Ag',
      [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7,8,7], 4.09)
    >>> atoms = hollowCore(atoms, 4.09, 5)
    >>> view(atoms)
    """
    def dist2( P1, P2):
        return (P1[0]-P2[0])**2+(P1[1]-P2[1])**2+(P1[2]-P2[2])**2

    assert radius > 0
    n_atoms = len(atoms)
    Xc = (min(atoms.positions[:,0])+max(atoms.positions[:,0]))/2.0
    Yc = (min(atoms.positions[:,1])+max(atoms.positions[:,1]))/2.0
    Zc = (min(atoms.positions[:,2])+max(atoms.positions[:,2]))/2.0
    center = [Xc, Yc, Zc]
    #atoms.translate([-Xc,-Yc,-Zc])
    i = 0
    while i < len(atoms):
        if dist2(atoms[i].position, center) < radius*radius:
            atoms.pop( i )   # remove atom
        else:
            i += 1
    return atoms



def randomize_biatom(atoms, type_a, type_b, ratio):
    """ replace randomly to acheive target conc """
    n_A = 0
    n_B = 0
    for atom in atoms:
        if atom.symbol == type_a:
            n_A += 1
        elif atom.symbol == type_b:
            n_B += 1
        else:
            raise Error('Extra chemical element %s!'%atom.chemical_symbol)
    #print n_A, n_B
    N = len(atoms)
    #print "conc",  n_A *1.0 / N
    r = random.Random()
    while n_A < ratio*N:  # add A atoms randomly
        index = r.randint(0, N-1)
        if (atoms[index].symbol != type_a):
            #print "changing atom #"+str(index)+" to "+type_a
            #prob = probability(dists[index]/Rmax, p)
            #print p
            if (r.randint(0, 1000) < 500):
                atoms[index].symbol = type_a
                n_A += 1
    return atoms

def randomize_biatom_13(atoms, type_a, type_b, ratio):
    """ replace randomly by clusters of 13 atoms
     to acheive target conc """
    n_A = 0
    n_B = 0
    for atom in atoms:
        if atom.symbol == type_a:
            n_A += 1
        elif atom.symbol == type_b:
            n_B += 1
        else:
            raise Error('Extra chemical element %s!'%atom.chemical_symbol)
    #print n_A, n_B
    N = len(atoms)
    nl = NeighborList([1.5]*N, self_interaction=False, bothways=True)  # 2*1.5=3 Angstr. radius
    nl.build(atoms)
    #print "conc",  n_A *1.0 / N
    r = random.Random()
    while n_A < ratio*N:  # add A atoms randomly
        index = r.randint(0, N-1)
        if (atoms[index].symbol != type_a):
            #print "changing atom #"+str(index)+" to "+type_a
            #if (r.randint(0, 1000) < 500):
            atoms[index].symbol = type_a
            n_A += 1
            indeces, offsets = nl.get_neighbors(index)
            for ia in indeces :
                if (atoms[ia].symbol != type_a)&(n_A < ratio*N):
                    atoms[ia].symbol = type_a
                    n_A += 1
    return atoms

def randomize_userfunc(atoms, new_type, user_func):
    """ replace host atoms randomly by new_type of atom
    by user function of probability distribution.
    Concentration is hidden that function.
    Go throw all atoms one type."""
    #TODO: backup atoms?
    N = len(atoms)
    qsar = QSAR(atoms)
    r = random.Random()
    dists = qsar.atom_distances()
    Rmax = max(dists)
    for i_atom in xrange(N):
        #r.random() - random float in interval [0,1)
        x = dists[i_atom]/Rmax
        if r.random() < user_func(x):
            atoms[i_atom].symbol = new_type

    return atoms

if __name__ == '__main__':
    #
    # test sphericalFCC
    atoms = sphericalFCC('Ag', 4.09, 8)
    #from ase.visualize import view
    #view(atoms)
    #raw_input('press enter')
    #
    from ase.cluster.cubic import FaceCenteredCubic
    atoms = FaceCenteredCubic(
      'Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [7, 8, 7], 4.09)
    #
    # test core shell
    #atoms = CoreShellFCC(atoms, 'Pt', 'Ag', ratio=0.6, a_cell=4.09)  # ratio-based filling
    #atoms = sphericalFCC('Ag', 4.09, 8)
    #atoms = CoreShellFCC(atoms, 'Pt', 'Ag', ratio=0.0, a_cell=4.09, n_depth=1)
    #atoms = randomize_biatom(atoms, 'Pt', 'Ag', ratio=0.6)
    #atoms = randomize_biatom_13(atoms, 'Pt', 'Ag', ratio=0.6)
    atoms = hollowCore(atoms, 4.09, 5)

    from ase.visualize import view
    view(atoms)
