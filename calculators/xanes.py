from __future__ import print_function

from ase import Atom, Atoms
from ase.calculators.calculator import Calculator, FileIOCalculator
from ase import units
import sys
import os
import shutil
import numpy as np
import scipy.optimize as sopt


class Fdmnes(FileIOCalculator):
    implemented_properties = ['energy', 'forces']  # forces should be listed here to use optimizer

    def __init__(self, exp_filename=None, energy_range=None, edge='K', nonexc=False,
        absorber='Cu', radius=6.0, energpho=True,
        eshift=None, scale=1.0, fermilevel=-2.5,
        relativism=False,
        green=False, eimag=0.01, rmt=None,                # MT
        iord=4, adimp=0.25,                                     # non-MT
        lmax=-1,
        ncores=1,
        fdmnes='fdmnes.bin', logfile='-'):
        """ Calculator for the fitting of XANES data.

        Instead of energy it calculates the misfit beetween experimental data and theory.
        Theory is calculated by external code, in particular,
        FDMNES available free of charge on <https://github.com/gudasergey/fdmnes>, <http://neel.cnrs.fr/spip.php?rubrique1007&lang=en >.

        Parameters:

        exp_filename:
           the exprimental XANES mu(E) to compare results with
           if set to None, no comparison performed
        energy_range:
           range of energies for calculation
           if None set to default [-5.0 0.5 -1.0 0.1 1.0 0.5 10. 1. 40.]
        edge:
           'K' or 'L3'
        nonexc: bool
            use non-excited state
        absorber: str or list
           symbol of absorbing element or list of indexes
        ncores: int
            number of cores/processes. Default is 1.
        fdmnes:
           command to run FDMNES
        logfile:
           file to write debug info.
           '-' value will write to stdout.

        Note:
          This calculator produced not energy, but misfit between computed and exerimental XANES!
          IMPORTANT: Don't rely on potential_energy() returned value in MD simulations!!!
        Based on:
          ASE caclulator interface <wiki.fysik.dtu.dk/ase/>
          and inspired by FitIt <www.nano.sfedu.ru/fitit/>.
        """
        import ase.parallel  # logging copied from ase.md.logger.MDLogger
        if ase.parallel.rank > 0:
            self.log = None  # Only log on master
        if logfile == '-':
            self.log = sys.stdout
        elif hasattr(logfile, 'write'):
            self.log = logfile
        elif not (logfile is None):
            self.log = open(logfile, 'w')
        else:
            self.log = None

        Calculator.__init__(self)

        self.scale = scale
        self.energpho = energpho
        if (eshift is None) and (not energpho):
            eshift = 8979  #TODO: do element-specific
            print('WARNING: eshift is not set! Using default value of %.1f eV corresponding to Cu K-edge.' % eshift)
        elif energpho:
            eshift = 0
        self.eshift = eshift

        if energy_range == None:
            energy_range = [-5.0, 0.5, -1.0, 0.1, 1.0, 0.5, 10., 1., 40.]
        if len(energy_range) % 2 != 1:
            raise Exception('Invalid energy_range.\nTuple of odd length expected.')
        self.energy_range = energy_range
        if energpho:
            self.energies = None  # will read from result
        else:  # can be precalculated
            self.energies = []
            for i in range(0, len(energy_range)-2, 2):
                self.energies.extend(
                    np.arange(energy_range[i], energy_range[i+2], energy_range[i+1])
                )
            self.energies.append(energy_range[-1])
            self.energies = np.array(self.energies)

        self.natoms = 0
        self.prefix = ''
        self.fdmnes = fdmnes
        self.directory = 'temp_fdmnes_calc'
        #self.command = os.path.join(self.directory, self.feff)
        #self.command = os.path.join('./', self.feff ) + ' > feff.out'
        self.command = self.fdmnes + ' > fdmnes.out'

        self.exp_filename = exp_filename
        self.exp_data = None
        #~ self._load_exp(exp_filename)

        #~ self.S02 = 1.0
        self.edge = edge
        self.absorber = absorber
        self.radius = radius
        self.green = green  # MT-calculation
        self.eimag = eimag  #

        if not(self.log is None):
            self.log.write(' Command:\t%s\n Directory:\t%s \n Chi file:\t%s\n'
               % (self.command, self.directory, exp_filename))

    def _load_exp(self):
        if (self.exp_filename is None) or (self.energies is None):
            self.exp_data = None
        else:
            data = np.genfromtxt(self.exp_filename)
            # rebin to self.energies
            self.exp_data = np.interp(self.energies, data[:,0], data[:,1])

    def _load_theor(self, scale=None, eshift=None):
        data = np.genfromtxt('temp_fdmnes_calc/fdmnes_output.txt', skip_header=2)
        if self.energpho:
            self.energies = data[:,0]
        if eshift is None:
            #~ print('self.eshift = %f' % self.eshift)
            self.energies += self.eshift
        else:
            #~ print('eshift = %f' % eshift)
            self.energies += eshift
        if scale is None:
            #~ print('self.scale = %f' % self.scale)
            self.theor_data = self.scale * data[:,1]
        else:
            #~ print('scale = %f' % scale)
            self.theor_data = scale * data[:,1]

    def residual(self, x=None):
        """
        If vary set to True than energy shift and intensity scale will be guessed
        """
        if x is not None:  # called from optimizer, have to reload to rebin
            self._load_theor(scale=x[0], eshift=x[1])
            self._load_exp()  # rebinned at load
        if self.exp_data is None:
            return -1
        else:
            rwin = np.ones(len(self.energies))
            res = np.sum(rwin * ((self.exp_data - self.theor_data)**2))
            norm = np.sum(rwin * ((self.exp_data)**2))
            # R-factor, %
            return res / norm * 100.0

    def calculate(self, atoms, properties=['energy'], system_changes=[]):
        """ overrides parent calculate() method for testing reason
            should not be in prod. version """
        try:
            FileIOCalculator.calculate(self, atoms, properties, system_changes)
            #~ self.read_results()           # to test
            #~ self.results['energy'] = self.residual()  # to test
        except RuntimeError:
            self.results['energy'] = 999 # big value

    def copy_fdmnes_bin(self):  # is it neccesary?
        src = self.fdmnes
        dst = self.directory
        shutil.copy(src, dst)

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        #~ self.copy_fdmnes_bin()
        # create fdmfile.txt
        filename = os.path.join( self.directory, 'fdmfile.txt' )
        try:
            fout = open(filename, 'w')
        except:
            if not(log is None):
                self.log.write("Cannot open file '%s'" % filename)
            return
        buff = []
        buff.append('1')
        buff.append('fdmnes_input.txt')
        try:
            fout.write('\n'.join(buff))
            fout.write('\n')
            fout.flush()
        except:
            if not(log is None):
                self.log.write("Cannot write to file '%s'" % filename)
            return
        fout.close()
        # create input file
        filename = os.path.join( self.directory, 'fdmnes_input.txt')
        try:
            fout = open(filename, 'w')
        except:
            if not(log is None):
                self.log.write("Cannot open file '%s'" % filename)
            return
        buff = []
        buff.append(' ! Fdmnes indata file')
        buff.append(' ! generated with ASE')
        buff.append(' Filout ')
        buff.append('   %s \n' % 'fdmnes_output.txt')
        buff.append(' Range   ! Energy range of calculation (eV)')
        buff.append(' '.join(map(str, self.energy_range)))
        #~ buff.append('  ! first energy, step, intermediary energy, step ..., last energy\n')
        buff.append('\n Radius ')
        buff.append('   %.2f \n' % self.radius)
        buff.append(' Edge ')
        buff.append('   %s \n' % self.edge)
        if self.energpho:
            buff.append(' Energpho \n')
        buff.append(' Absorber ')
        if isinstance(self.absorber, str):
            abs_indeces = [i+1 for i, atom in enumerate(atoms) if atom.symbol == self.absorber]
            if len(abs_indeces) > 1:
                print('WARNING: there is more than single absorber %s atom, but will use only first.' % self.absorber)
            elif len(abs_indeces) == 0:
                raise Exception('Absorber atoms of %s not found!' % self.absorber)
            buff.append('   %i \n' % abs_indeces[0])
        else:
            buff.append('   %i \n' % self.absorber[0])
        if self.green:  # MT-calculation
            buff.append(' Green \n')
            buff.append(' Eimag ')
            buff.append('   %.3f \n' % self.eimag)

        if atoms.get_pbc().any():
            buff.append(' Crystal   ! Periodic material description (unit cell)')
        else:
            buff.append(' Molecule  ! finite system')

        if atoms.cell[0,1]**2 + atoms.cell[0,2]**2 + atoms.cell[1,0]**2 + atoms.cell[1,2]**2 + \
           atoms.cell[2,0]**2 + atoms.cell[2,1]**2 > 1e-6:
               raise Exception('Currently only orthorombic cells are supported. Sorry!')
        buff.append(' %.3f %.3f %.3f 90. 90. 90.  ! a, b, c, (Angstr.) alpha, beta, gamma (degr.)' %
                   (atoms.cell[0,0], atoms.cell[1,1], atoms.cell[2,2]))

        for atom in atoms:
            num = atom.number
            x, y, z = atom.position
            a, b, c = atoms.cell[0,0], atoms.cell[1,1], atoms.cell[2,2]
            buff.append('  %i   %.6f   %.6f   %.6f '% (num, x/a, y/b, z/c))
        buff.append(' End')
        try:
            fout.write('\n'.join(buff))
            fout.write('\n')
        except:
            if not(log is None):
                self.log.write("Cannot write to file '%s'" % filename)
            return
        fout.close()
        #~ print( len(self.atoms) )
        #~ print( self.directory )

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
        self._load_theor()
        self._load_exp()
        # resid = self.residual()
        # vary scale and eshift for better agreement
        if self.exp_data is not None:
            values = [self.scale, self.eshift]
            sopt_result = sopt.minimize(fun=self.residual, x0=values, method='L-BFGS-B', tol=1e-3,
                                 bounds=[(0,None),(-10,10)],
                                 options={'maxiter':50, 'disp':False})
            #~ print(sopt_result)
            if not sopt_result.success:
                print('WARNINIG: scale-eshift optimization not converged.')
            self.scale = sopt_result.x[0]
            self.eshift = sopt_result.x[1]
            print('Choosen scale: %f and shift: %f eV. ' % (self.scale, self.eshift))
            # reread with new parameters to be sure
            self._load_theor()
            self._load_exp()
        #~ print('self.residual()')
        #~ print(self.residual())
        self.results['energy'] = self.residual()

    def _do_plot(self, rwin_divide=10):
        if (self.exp_filename is not None) and (self.exp_data is None):
            self._load_exp()
        if self.exp_data is not None:
            plt.plot(self.energies, self.exp_data, label='exp.')
        plt.plot(self.energies, self.theor_data, label='sim.')
        plt.legend()
        plt.xlabel('E, eV')
        plt.ylabel('mu(E)')

    def show_plot(self, rwin_divide=10):
        import matplotlib.pyplot as plt  # should import be here?
        self._do_plot()
        plt.show()

    def save_plot(self, filename):
        """ Save plot comparing experiment and model in file """
        import matplotlib.pyplot as plt  # should import be here?
        plt.ioff()
        self._do_plot()
        plt.savefig(filename)
        plt.close()
        plt.ion()

    #~ def set_atoms(self, atoms):
        #~ self.atoms = atoms        # .copy() ?


if __name__ == '__main__':
    #~ from ase.cluster.cubic import FaceCenteredCubic
    #~ atoms = FaceCenteredCubic(
      #~ 'Pt', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [2, 3, 1], 4.09)
    if False:
        a = 3.61
        atoms = Atoms('Cu4', positions=[[0,0,0],[a/2,a/2,0],[a/2,0,a/2],[0,a/2,a/2]], cell=[a,a,a])
        atoms.set_pbc(True)
    else:
        from ase.io import read
        atoms = read('optimized_T3_8r.cif')
    #~ from ase.visualize import view
    #~ view(atoms)

    calc = Fdmnes(
        fdmnes='/home/leon/bin/fdmnes_linux64_2018_11_30',
        exp_filename='5CMRTHe.txt.nor', scale=20.0, #eshift=8979.00,
        energpho=True,
        green=True, eimag=0.05,  # MT calc
        radius=4.0,  # small radius for test
        energy_range=[-10, 1.0, -5, 0.5, 5, 1.0, 10, 2.0, 40, 5.0, 60, 10.0, 100], edge='K', absorber='Cu'
    )

    atoms.set_calculator(calc)

    print ('   MISFIT: %.6f' % atoms.get_potential_energy() )

    import matplotlib.pyplot as plt
    calc.show_plot()
