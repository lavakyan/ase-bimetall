# ase-bimetall

Atomic Simulation Environment scripts for BIMETALLic nanoparticles

## Description

The collection of my scripts for Atomic Simulation Environment (ASE)
used for study of bimetallic nanoparticles.
The required ASE libraries are available at [official site](https://wiki.fysik.dtu.dk/ase/) under free license.

Implemented (Reverse) Monte-Carlo method for the search of two-component structures based on the coordination numbers information.
The MC class and Move nethods are heavily based on [ASAP3](https://wiki.fysik.dtu.dk/asap/) implementation.

## Scripts

* `analyzecn.py` -- procedure-driven calculation of coordination numbers and similar quantities. *Should be replaced by object oriented version QSAR*;

* `qsar.py` -- class for calculation of coordination numbers and similar quantities;

* `coreshell.py` -- procedures to build atomic clusters with particular architecture;

* `mc_search.py` -- Monte-Carlo search of structures with known coordination numbers. Based on MC code from [ASAP](https://wiki.fysik.dtu.dk/asap/Monte%20Carlo%20simulations) project.


## Examples

* Import libraries

    ``` python
    from ase import Atom, Atoms
    from coreshell import sphericalFCC, randomize_biatom
    ```
* Build a spherical nanoparticle and stores it in Atoms object:

    ``` python
    atoms = sphericalFCC('Ag', 4.09, 8)
    ```
* Add some Pt atoms:

    ``` python
    atoms = randomize_biatom(atoms, 'Pt', 'Ag', ratio=0.6)

    ```
* You can view the structure using ASE in a very simple way:

    ``` python
    from ase.visualize import view
    view(atoms)
    ```
    ![Screenshot of ASE GUI widndow](img/shot-PtAg.jpg?raw=true "Screenshot of ASE GUI window")
* Find out coordination numbers:

    ``` python
    from qsar import QSAR

    qsar = QSAR(atoms)

    qsar.biatomic('Pt', 'Ag')

    print(qsar.report_CNs())
    ```
    The output will like:

    ```
    Coordination numbers:
        Pt-Pt   6.364
        Pt-Ag   4.058
        Ag-Pt   6.097
        Ag-Ag   4.340

    ```

* The Monte-Carlo example can be found in `__main__` method in `mc_search.py`.
