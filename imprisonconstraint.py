import numpy as np
from ase.constraints import FixConstraint

class ImprisonConstraint(FixConstraint):
    ''' Push atoms back inside the cell at each
    calculation step. '''

    def __init__(self, skin=0.01):
        '''
        skin : float, Angstrom
            depth of the skin layer near cell borders

        Example
        ----------
            c = ImprisonConstraint()
            atoms.set_constraint(c)
        '''
        self.skin = skin

    def adjust_positions(self, atoms, new):
        new_atoms = atoms.copy()
        new_atoms.set_positions(new, apply_constraint=False)
        scaled = new_atoms.get_scaled_positions(wrap=True)
        #~ if (scaled<0).any():
            #~ print('spotted negative')
        new_atoms.set_scaled_positions(scaled)
        new[:] = new_atoms.get_positions()

    def adjust_forces(self, atoms, forces):
        pass

    #def adjust_potential_energy(self, positions, forces):
    #    pass

    def __repr__(self):
        return 'Push atoms out of the cell back to the cell'

    def copy(self):
        return ConstantForce(a=self.index, force=self.force)

if __name__ == '__main__':
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.calculators.emt import EMT
    from ase.md.verlet import VelocityVerlet
    from ase.units import fs
    from constantforce import ConstantForce

    atoms = FaceCenteredCubic(
      'Ag', [(1, 0, 0)], [1], 4.09)
    atoms.center(10)
    atoms.pbc = True

    atoms.set_calculator( EMT() )
    cf = ConstantForce( 10, [0,1,0] )  # y=dircted force
    ic = ImprisonConstraint()
    atoms.set_constraint( [cf, ic] )

    md = VelocityVerlet( atoms, 1*fs, trajectory = 'cf_test.traj', logfile='-' )
    md.run(200)

    #~ from ase.visualize import view
    #~ view(atoms)
