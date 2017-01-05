import numpy as np
from ase.constraints import FixConstraint

class ConstantForce(FixConstraint):
    """Applies a constant force to an atom."""

    def __init__(self, a, force):
        """ Force applied to ant atom a

        Parameters
        ----------
        a : int
           Index of atom
        force :  array three floats
           components of the force

        Example
        ----------
            c = ConstantForce( 10, [0,1,0] )  # y=dircted force
            atoms.set_constraint( c )
        """
        self.index = a
        self.force = np.array(force)

    def adjust_positions(self, oldpositions, newpositions):
        pass

    def adjust_forces(self, positions, forces):
            forces[self.index] += self.force

    #def adjust_potential_energy(self, positions, forces):
    #    pass

    def __repr__(self):
        return 'Constant force appliet to (%d) atom' % self.index

    def copy(self):
        return ConstantForce(a=self.index, force=self.force)

if __name__ == '__main__':
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.calculators.emt import EMT
    from ase.md.verlet import VelocityVerlet
    from ase.units import fs

    atoms = FaceCenteredCubic(
      'Ag', [(1, 0, 0)], [1], 4.09)
    atoms.center(10)

    atoms.set_calculator( EMT() )
    c = ConstantForce( 10, [0,1,0] )  # y=dircted force
    atoms.set_constraint( c )

    md = VelocityVerlet( atoms, 1*fs, trajectory = 'cf_test.traj', logfile='-' )
    md.run(100)

    #~ from ase.visualize import view
    #~ view(atoms)
