"""Implement special border conditions for molecular dynamics."""

import weakref
# ase.parallel imported in __init__

class Freeze:
    """Class for freezing atoms close to computation cell
    in molecular dynamics simulations.

    Parameters:
    atoms:         The atoms.
    margin:        define freezine region at the cell bounds, Angstr.
    """

    #def __init__(self, dyn, atoms, margin=2):
    def __init__(self, atoms, margin=2):
        import ase.parallel
        if ase.parallel.rank > 0:
            logfile='/dev/null'  # Only log on master
        #~ if hasattr(dyn, "get_time"):
            #~ self.dyn = weakref.proxy(dyn)
        #~ else:
            #~ self.dyn = None
        self.atoms  = atoms
        self.margin = margin
        #self.natoms = atoms.get_number_of_atoms()

    def __call__(self):
        # TODO: extend for non-cubic cells
        cell = atoms.cell
        for atom in atoms:
            for i in range(3):
                if atom.position[i] < self.margin:
                    atom.momentum[i] = 0
                if atom.position[i] > cell[i,i]-self.margin:
                    atom.momentum[i] = 0

class Mirror(Freeze):
    """Class for reflection of atoms from computation cell
    in molecular dynamics simulations.

    Parameters:
    atoms:         The atoms.
    margin:        define freezine region at the cell bounds, Angstr.
    """
    def __call__(self):
        # TODO: extend for non-cubic cells
        cell = atoms.cell
        for atom in atoms:
            for i in range(3):
                if atom.position[i] < self.margin:
                    atom.momentum[i] = abs(atom.momentum[i])
                if atom.position[i] > cell[i,i]-self.margin:
                    atom.momentum[i] = -abs(atom.momentum[i])

if __name__ == '__main__':
    # test
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.calculators.emt import EMT
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.io.trajectory import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.units import fs, kB

    atoms = FaceCenteredCubic(
      'Pt', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [2, 3, 1], 4.09)
    atoms.center(vacuum=5)
    atoms.set_calculator( EMT() )
    T = 10000    # K -- try to vaporize
    MaxwellBoltzmannDistribution(atoms, kB*T)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = NVTBerendsen(atoms, 1*fs, T, 0.5, logfile='-')
    traj = Trajectory('freezer_test.traj', 'w', atoms)
    dyn.attach(traj.write, interval=10)
    #fr =  Freeze(atoms)
    fr =  Mirror(atoms)
    dyn.attach( fr, interval=20 )
    dyn.run(5000)
    traj.close()

