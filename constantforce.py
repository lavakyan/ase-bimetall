import numpy as np
from ase.constraints import FixConstraint


class ConstantForce(FixConstraint):
    '''Applies a constant force to an atom.'''

    def __init__(self, a, force):
        ''' Force applied to an atom

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
        '''
        self.index = a
        self.force = np.array(force)

    def adjust_positions(self, atoms, new):
        new = atoms.positions  # ???

    def adjust_forces(self, atoms, forces):
        forces[self.index] += self.force

    def __repr__(self):
        return 'Constant force applied to (%d) atom' % self.index

    def copy(self):
        return ConstantForce(a=self.index, force=self.force)


class CentralRepulsion(FixConstraint):
    '''
    Applies a central repulsion force to an atom.
    '''

    def __init__(self, R, A=4, alpha=1):
        ''' Apply repuslive force on some atoms

        Parameters
        ----------
        R : vector (array of three floats)
           center of potential
        A, alpha :  float
           parameters of repuslsive potential
           $ U(\vec{r}) = \frac{A} { \vline \vec{r}-\vec{R} \vline^\alpha }
           alpha is unitless
           A in units eV*A^alpha

        Example
        ----------
            c = CentralRepulsion([0,1,2,3,4,5], A=10, alpha=2)
            atoms.set_constraint( c )
        '''
        self.R = R
        self.A = A
        self.alpha = alpha

    def adjust_positions(self, atoms, new):
        new = atoms.positions  # ???

    def adjust_forces(self, atoms, forces):
        for i in range(len(atoms)):
            d = atoms[i].position - self.R
            forces[i] += -self.alpha * self.A * d / np.norm(d)**(self.alpha + 1)

    def adjust_potential_energy(self, atoms, energy):
        for i in range(len(atoms)):
            d = atoms[i].position - self.R
            np.norm(d)
            energy += self.A / np.norm(d)**self.alpha

    def __repr__(self):
        return 'Repulsion potential'

    def copy(self):
        return CentralRepulsion(self, R=self.R, A=self.A, alpha=self.alpha)


if __name__ == '__main__':
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.calculators.emt import EMT
    from ase.md.verlet import VelocityVerlet
    from ase.units import fs

    atoms = FaceCenteredCubic('Ag', [(1, 0, 0)], [1], 4.09)
    atoms.center(10)

    atoms.set_calculator(EMT())
    c = ConstantForce(10, [0, 1, 0])  # y=dircted force
    atoms.set_constraint(c)

    md = VelocityVerlet(atoms, 1*fs, trajectory='cf_test.traj',
                        logfile='-')
    md.run(100)

    # from ase.visualize import view
    # view(atoms)
