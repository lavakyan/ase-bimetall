# ase-bimetall

## Description
The collection of my scripts for Atomic Simulation Environment (ASE). Mostly concerned bimetallic nanoparticles.

## Files

*analyzecn.py -- procedure-driven calculation of coordination numbers and similar quantities. *Should be replaced by object oriented version in future.*

*coreshell.py -- procedures to build atomic clusters with particular architecture

*mc_search.py -- Monte-Carlo search of structures with predicted coordination numbers. Based on code from ASAP project (https://wiki.fysik.dtu.dk/asap/Monte%20Carlo%20simulations)

*qsar.py -- class for calculation of coordination numbers and similar quantities


## Examples

* Import libraries (overkill)

from ase.all import *
from coreshell import sphericalFCC, randomize_biatom

* Build a spherical nanoparticle and put it to object atoms:

atoms = sphericalFCC('Ag', 4.09, 8)

* Add some Pt atoms:

atoms = randomize_biatom(atoms, 'Pt', 'Ag', ratio=0.6)

* You can view the structure using ASE in a very simple way:

view(atoms)

* Find out coordination numbers:

from qsar import QSAR

qsar = QSAR(atoms)
qsar.biatomic('Pt', 'Ag')

print('N = {}'.format(qsar.N))

print('CN_PtPt = {}'.format(qsar.CN_AA))

print('CN_PtAg = {}'.format(qsar.CN_AB))

print('CN_AgAg = {}'.format(qsar.CN_BB))

print('CN_AgPt = {}'.format(qsar.CN_BA))

