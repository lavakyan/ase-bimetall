#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# based on the class from ASAP project
# https://wiki.fysik.dtu.dk/asap/Monte%20Carlo%20simulations
#
# MC to achieve structure with:
# i)   desired coordination numbers
# ii)  desired concentration of components
# iii) lowest total energy

from __future__ import print_function
import os, sys
import numpy as np
#from scipy import ndimage  # to calc neigbors w/o NeighborList
from scipy.signal import convolve
from ase import Atom, Atoms
#from qsar import QSAR
from ase.units import kB
from asap3 import EMT

class MC:
    basis_factor = 0.5
    int_basis = np.array([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0]])
    # последняя ось печатается слева направо,
    # предпоследняя — сверху вниз,
    # и оставшиеся — также сверху вниз, разделяя пустой строкй.

    # convolution with this array yields number of neigbors for FCC
    neib_matrix = np.array([
                            [[0, 0, 0],
                             [0, 1, 1],
                             [0, 1, 0]],
                            [[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]],
                            [[0, 1, 0],
                             [1, 1, 0],
                             [0, 0, 0]]
                            ])

    #def __init__(self, atoms, log='-'):
    #self.atoms = atoms # todo -- use this atoms to build initial configuaration
    #TODO: change default values of chems to [] - empty list
    def __init__(self, log='-', chems=[29, 78]):
        self.atoms = None
        self.moves = []   # available move types
        self.moves_weight = []  # some random moves are more random
        self.move = None  # current move
        self.nsteps = {}  # dict-counter
        self.naccept = {} # dict-counter
        #self.chems = [] # chemical symbols
        self.chems = chems # chemical symbols
        self.CNs = []   # coordination numbers
        #self.conc = []     # concentration of components
        self.E = 1e32   # potential energy per atom

        #self.A = 29     # Cu
        #self.chems.append(self.A)
        #self.chems.append( 29 ) # Cu
        #self.conc.append( 0.47 )
        #self.chems.append( 78 ) # Pt
        #self.conc.append( 0.53 )
        for a1 in self.chems:
            for a2 in self.chems:
                self.CNs.append(0)
        self.a = 3.610;  # lattice constant of Copper
        self.init_grid( 40 )
        #self.GRID[...] = 0
        n = int(self.L/2)
        #s = 2
        #self.GRID[(n-s):(n+s), (n-s):(n+s), (n-s):(n)] = self.chems[0]
        #self.GRID[(n-s):(n+s), (n-s):(n+s), (n):(n+s)] = self.chems[1]

        # hollow core-shell initial structure
        s = 4
        self.GRID[(n-s):(n+s), (n-s):(n+s), (n-s):(n+s)] = self.chems[1]
        s = 3
        self.GRID[(n-s):(n+s), (n-s):(n+s), (n-s):(n+s)] = self.chems[0]
        s = 2
        self.GRID[(n-s):(n+s), (n-s):(n+s), (n-s):(n+s)] = 0
        #self.GRID[(n+s), (n+s), (n+s)] = self.chems[1]
        #self.GRID[(n-s), (n+s), (n+s)] = self.chems[1]
        #self.GRID[(n+s), (n-s), (n+s)] = self.chems[1]
        #self.GRID[(n+s), (n+s), (n-s)] = self.chems[1]

        self.attach_move( MoveDestroy(), 0.5 )
        self.attach_move( MoveCreate(),  0.25 )
        self.attach_move( MoveSwap(),    0.25 )

        if isinstance(log, str):
            if log == '-':
                self.logfile = sys.stdout
            else:
                self.logfile = open(log, 'a')
        else:
            self.logfile = None

    def init_grid(self, GRID_SIZE):
        self.L = int(GRID_SIZE)
        self.GRID = np.zeros((self.L, self.L, self.L))
        self.NEIB = np.zeros((self.L, self.L, self.L))

    def set_targets(self, target_CNs, target_conc, temperature=1000):
        self.temp = temperature

        self.targetCNs = np.array(target_CNs)  # linked?
        #self.CNs = target_CNs[:]     # make a copy
        self.target_conc = target_conc

        if self.logfile is not None:
            self.logfile.write('='*20 + ' Targets ' + '='*20+'\n')
            self.logfile.write('Temperature %f\n' % self.temp)
            self.logfile.write('Coordination numbers:\n')
            i = 0
            for B in self.chems:
                for A in self.chems:
                    self.logfile.write('  CN[%i-%i] = %f\n' % (A, B, self.targetCNs[i]))
                    i += 1
            self.logfile.write('Concentrations:\n')
            i = 0
            for B in self.chems:
                self.logfile.write('  conc[%i] = %f\n' % (B, self.target_conc[i]))
                i += 1
            self.logfile.write('Energy: ASAP3.EMT per atom -> minimum\n')
            self.logfile.write('='*49+'\n')
            self.logfile.flush()

    def attach_move(self, move, weight=1.0):
        if not hasattr(move, '__call__'):
            raise ValueError("Attached move is not callable.")
        #if hasattr(move, 'set_atoms'):
        #    move.set_atoms(self.atoms)
        #if hasattr(move, 'set_optimizer'):
        #    move.set_optimizer(self)
        self.moves.append(move)
        self.moves_weight.append(weight)
        self.nsteps[move.get_name()] = 0
        self.naccept[move.get_name()] = 0

    def random_move(self):
        self.calc_neighbors()
        #return self.moves[(np.random.uniform() < self.moves_weight).argmax()]
        ## first, choose move, then - position ##
        self.move = self.moves[0] # instance of MoveSwap  #TODO choose by random
        if isinstance(self.move, MoveSwap):
            found = False
            while not found: # find non-emty position
                n1, n2, n3 = np.random.random_integers(0, self.L-1, 3)
                found = self.GRID[n1, n2, n3] > 0
            # find chemical element to swap
            B = filter(lambda A: A != self.GRID[n1, n2, n3], self.chems)[0]
            self.move.setup(self.GRID, n1, n2, n3, B)

        #~ while not found:
            #~ n1, n2, n3 = np.random.random_integers(0, self.L-1, 3)
            #~ found = self.NEIB[n1, n2, n3] > 0
        #~ if self.GRID[n1, n2, n3] > 0: # already filled, may be destroyed or swapped
            #~ try: # http://stackoverflow.com/questions/5655198/python-get-instance-from-list-based-on-instance-variable
                #~ self.move = filter(lambda mv: isinstance(mv, MoveDestroy), self.moves)[0]
            #~ except IndexError:
                #~ return self.random_move()
            #~ #if self.weightedChoice([True, False], self.moves_weight[[0,2])): # destroy
            #~ if (np.random.uniform() > 0.5):
                #~ self.move.setup(self.GRID, n1, n2, n3)
            #~ else: #swap
                #~ try:
                    #~ self.move = filter(lambda mv: isinstance(mv, MoveSwap), self.moves)[0]
                #~ except IndexError:
                    #~ return self.random_move()
                #~ A = self.GRID[n1, n2, n3]
                #~ B = self.GRID[n1, n2, n3]
                #~ while B == A:
                    #~ B = self.weightedChoice(self.chems, self.target_conc)
                #~ self.move.setup(self.GRID, n1, n2, n3, B)
        #~ else: # empty, may be filled
            #~ #np.random.choice(self.chems, self.target_conc)
            #~ A = self.weightedChoice(self.chems, self.target_conc)
            #~ #print('** Adding ',A)
            #~ try:
                #~ self.move = filter(lambda mv: isinstance(mv, MoveCreate), self.moves)[0]
            #~ except IndexError:
                #~ return self.random_move()
            #~ self.move.setup(self.GRID, n1, n2, n3, A)
        return self.move

        #print [np.random.uniform() < self.moves_weight].any()

    def get_atoms(self):
        if True:  #self.atoms == None:
            self.atoms = Atoms()
            self.atoms.set_cell( self.int_basis*self.basis_factor*self.a*self.L )
            for n1 in xrange(self.L):
                for n2 in xrange(self.L):
                    for n3 in xrange(self.L):
                        A = self.GRID[n1, n2, n3]
                        if (A > 0):
                            pos = np.empty(3)
                            #for i in range(3):
                            pos[0] = 0.5*self.a*(n2+n3)
                            pos[1] = 0.5*self.a*(n1+n3)
                            pos[2] = 0.5*self.a*(n1+n2)
                            atom = Atom(A, position=pos)
                            self.atoms.append(atom)
        return self.atoms

    def set_atoms(self, atoms, margin = 5):
        """ set atoms position as initial values of GRID
        This function will alter the size of the GRID
        atoms - ASE.Atoms object with atomic system
        margin - the extra space from the borders of GRID array

        Example:
            mc = MC()
            atoms = read('initial.cube')
            mc.set_lattice_constant(2.772*sqrt(2))
            mc.set_atoms(atoms, margin=1)
            from ase.visualize import view
            view(mc.get_atoms())"""
        x = atoms.positions[:,0]
        y = atoms.positions[:,1]
        z = atoms.positions[:,2]
        n1 = np.round(1/self.a*(-x + y + z ))
        n2 = np.round(1/self.a*( x - y + z ))
        n3 = np.round(1/self.a*( x + y - z ))
        #~ n1.astype(int)
        #~ n2.astype(int)
        #~ n3.astype(int)
        # change GRID array to fit all the data + margin space
        min1 = n1.min()
        min2 = n2.min()
        min3 = n3.min()
        max1 = n1.max()
        max2 = n2.max()
        max3 = n3.max()
        L = max(max1-min1, max2-min2, max3-min3) + 1  # +1 is required for correct treatment of margin=0 case
        L += 2*margin
        print('L = %i\n' % L)
        self.init_grid( L )
        for i_atom in xrange(len(atoms)):
            in1 = margin + n1[i_atom] - min1
            in2 = margin + n2[i_atom] - min2
            in3 = margin + n3[i_atom] - min3
            self.GRID[in1, in2, in3] = atoms[i_atom].number
            if not(atoms[i_atom].number in self.chems):
                self.chems.append(atoms[i_atom].number)
                print('WARNING: Added atom with Z=%i'%atoms[i_atom].number)
        return L

    def calc_neighbors(self):
        """ To fill array of neighbors numbers.
        3x3x3 array neib_matrix is specific for FCC structure"""
        #self.NEIB = ndimage.convolve((self.GRID>1), self.neib_matrix, mode='constant', cval=0.0)
        self.NEIB = convolve(1*(self.GRID>1), self.neib_matrix, mode='same')
        return self.NEIB

    def get_N(self, chem=-1):
        """ Number of atoms. If chem == -1 returns total number of atoms """
        if chem > 0:
            return (self.GRID == chem).sum()
        else:
            return (self.GRID > 0).sum()

    def calc_CNs(self):
        """ To fill coordination numbers.
        Should be called after calc_neighbors"""
        # total CN:
        #self.CNs[0] = (self.NEIB[(self.NEIB>0) & (self.GRID>0)]).sum()
        #self.CNs[0] = self.CNs[0] * 1.0 / self.get_N()
        # partial CN:
        i = 0
        for B in self.chems:
            NEIB_AB = convolve(1*(self.GRID==B), self.neib_matrix, mode='same')
            for A in self.chems:
                # calc number of B around A
                self.CNs[i] = (NEIB_AB[(self.NEIB>0) & (self.GRID==A)]).sum()
                self.CNs[i] = self.CNs[i] * 1.0 / self.get_N(A)
                i += 1
        #print nnn
        #print sum(nnn)*1.0 / len(atoms)
    def calc_conc(self):
        N = (self.GRID > 0).sum()
        N_A = (self.GRID == self.chems[0]).sum()
        N_B = (self.GRID == self.chems[1]).sum()
        return [1.0*N_A/N, 1.0*N_B/N]

    def calc_energy(self):
        atoms = self.get_atoms()
        atoms.set_calculator(EMT())
        self.E = atoms.get_potential_energy() #/ self.get_N() # energy per atom!
        return self.E

    def run(self, nsteps=10):
        """ Run Monte-Carlo simulation for nsteps moves """
        for step in xrange(nsteps):
            move = self.random_move()
            if self.logfile is not None:
                #self.logfile.write('* Move: %s \t' % move.get_name())
                #self.logfile.write(' Pos.: ['+str(move.n1)+','+str(move.n2)+','+str(move.n3)+'] \t')
                self.logfile.write('* '+str(move))
            # perform and evaluate move
            if self.evaluate_move():
                self.accept_move()
                if self.logfile is not None:
                    self.logfile.write(' Acc!\n')
            else:
                self.reject_move()
                if self.logfile is not None:
                    self.logfile.write(' Rej.\n')
        self.log_stats()
        # show stats
        #if self.logfile is not None:
        #    self.logfile.write(
        #      '* Move accepted: %7i / %7i \t Total accepted: %7i / %7i\n' %
        #      (self.get_naccept(move.get_name()), self.get_nsteps(move.get_name()),
        #      self.get_naccept(), self.get_nsteps() )
        #    )

    def log_stats(self):
        if self.logfile is not None:
            self.logfile.write('='*60)
            self.logfile.write('\n%-13s  %-15s  %-15s\n' %
                               ('Move', 'Steps', 'Accepts'))
            self.logfile.write('-' * 60 + '\n')
            for m in self.moves:
                name = m.get_name()
                ns = self.get_nsteps(name)
                fs = 1.0 * ns / self.get_nsteps()
                na = self.get_naccept(name)
                if ns != 0:
                    fa = 1.0 * na / ns
                else:
                    fa = -1
                self.logfile.write('%-13s  %-7i (%5.3f)  %-7i (%5.3f)\n' %
                                   (name, ns, fs, na, fa))
            self.logfile.write('-' * 60 + '\n')
            ns = self.get_nsteps()
            na = self.get_naccept()
            self.logfile.write('%-13s  %-7i (%5.3f)  %-7i (%5.3f)\n' %
                               ('Total', ns, 1.0, na, 1.0 * na / ns))
            self.logfile.write('=' * 60 + '\n')
            self.logfile.write('Target CN  : '+str(self.targetCNs)
                            +'\nAchieved CN: '+str(self.CNs)
                            +'\nTarget conc: '+str(self.target_conc)
                            +'\nAchieved conc: '+str(self.calc_conc())
                            +'\nError function: '+str(self.error_function())
                            +'\n')
            self.logfile.write('Natoms = %i\n' % self.get_N())
            self.logfile.write('='*60+'\n')
            # for comparison with list version
            #~ from qsar import QSAR
            #~ q = QSAR(self.get_atoms())
            #~ q.monoatomic()
            #~ self.logfile.write('qsar> N  = %i\n' % q.N )
            #~ self.logfile.write('qsar> CN = %f\n' % q.CN)
            self.logfile.flush()
        #else:
        #    raise "Called log stats without logfile setted"

    def clear_stats(self):
        for key in self.nsteps:
            self.naccept[key] = 0
            self.nsteps[key] = 0

    def set_lattice_constant(self, lattice_constant):
        self.a = lattice_constant

    def error_function(self):
        # a-la-Lagrange coefficients (weights)
        wCN = 1             # CN contrib.
        wE  = 0  # 0.001    # Energy contrib.
        wX  = 1             # Concentration  contrib.
        self.calc_neighbors()
        self.calc_CNs()
        #E = self.calc_energy()
        #result = wCN * sum((np.array(self.targetCNs)-np.array(self.CNs))**2)
        # result = wCN * sum((self.targetCNs-self.CNs)**2) # -- last working version. Changed to treat skipping
        result = 0
        if wCN > 0:
            for i in range(len(self.targetCNs)):
                if self.targetCNs[i] > 0:
                    result += (self.targetCNs[i]-self.CNs[i])**2
        result *= wCN
        if wE > 0:
            result += wE * self.calc_energy()
        if wX > 0:
            curr_conc = self.calc_conc();
            #print(N, N_A, N_B)
            result += wX*( (self.target_conc[0] - curr_conc[0])**2 + (self.target_conc[1] - curr_conc[1])**2 )
        return result

    def evaluate_move(self):
        #oldCNs = self.CNs[:]
        Eold = self.error_function()
        #if self.logfile is not None:
        #    self.logfile.write('\nOld: E %f \t CN %f \t Energy %f' % (Eold, self.CNs[0], self.E))
        self.move()
        newCNs = self.CNs
        Enew = self.error_function()
        #if self.logfile is not None:
        #    self.logfile.write('New: E %f \t CN %f \t Energy %f\n' % (Enew, self.CNs[0], self.E))
        if Enew < Eold:
            if self.logfile is not None:
                self.logfile.write(' P: 1+   \t')
            return True
        else:
            prob = np.exp( (Eold - Enew) / (self.temp * kB))
            if self.logfile is not None:
                self.logfile.write(' P: %.3f\t' % prob)
            return prob > np.random.uniform()

    def accept_move(self):
        self.move.accept()
        self.naccept[self.move.get_name()] += 1
        self.nsteps[self.move.get_name()] += 1

    def reject_move(self):
        self.move.reject()
        self.nsteps[self.move.get_name()] += 1

    def get_nsteps(self, name='total'):
        if name == 'total':
            return sum(self.nsteps.values())
        else:
            return self.nsteps[name]

    def get_naccept(self, name='total'):
        if name == 'total':
            return sum(self.naccept.values())
        else:
            return self.naccept[name]

    def weightedChoice(self, objects, weights):
        """Return a random item from objects, with the weighting defined by weights
        (which must sum to 1).
        http://stackoverflow.com/questions/10803135/weighted-choice-short-and-simple"""
        # should be replaced by np.random.choice()
        # awaliable in newer versions of numpy..
        cs = np.cumsum(weights)     # An array of the weights, cumulatively summed.
        idx = sum(cs < np.random.uniform()) # Find the index of the first weight over a random value.
        return objects[idx]

#######################################################################

class Move:
    str_template = '\t[%i,%i,%i]: %i -> %i\t'

    def __init__(self):
        self.GRID = None

    def __call__(self):
        #
        #self.backup()
        pass

    def setup(self, GRID=None):
        self.set_grid(GRID)

    def set_grid(self, GRID):
        if GRID is not None:
           self.GRID = GRID

    def accept(self):
        pass

    def reject(self):
        pass

    def log(self):
        return "Move.log() call shoud not happen!"

    def get_name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.get_name()

class MoveDestroy(Move):
    def __init___(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        Move.__init__(self)

    def __call__(self):
        Move.__call__(self)
        self.GRID[self.n1, self.n2, self.n3] = 0

    def setup(self, GRID, n1, n2, n3):
        Move.setup(self, GRID)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        assert GRID[n1,n2,n3] > 0, 'Zero value of GRID passed to destroy!'
        self.backup_value = self.GRID[self.n1,self.n2,self.n3]

    def reject(self):
        self.GRID[self.n1,self.n2,self.n3] = self.backup_value

    def __str__(self):
        return Move.__str__(self) + self.str_template % (self.n1, self.n2, self.n3, self.backup_value, 0)

class MoveCreate(Move):
    def __init___(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        Move.__init__(self)

    def __call__(self):
        #self.backup()
        self.GRID[self.n1, self.n2, self.n3] = self.Zatom

    def setup(self,  GRID, n1, n2, n3, Zatom):
        Move.setup(self, GRID)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.Zatom = Zatom
        assert GRID[n1,n2,n3] == 0, 'Cant create, already filled!'

    def reject(self):
        self.GRID[self.n1,self.n2,self.n3] = 0

    def __str__(self):
        #('* Move: %s \t' % move.get_name())
        #self.logfile.write(' Pos.: ['+str(move.n1)+','+str(move.n2)+','+str(move.n3)+'] \t')
        return Move.__str__(self) + self.str_template % (self.n1, self.n2, self.n3, 0, self.Zatom)

class MoveSwap(Move):
    def __init___(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        Move.__init__(self)

    def __call__(self):
        self.GRID[self.n1, self.n2, self.n3] = self.Zatom

    def setup(self,  GRID, n1, n2, n3, Zatom):
        Move.setup(self, GRID)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.Zatom = Zatom
        assert GRID[n1,n2,n3] > 0, 'Use MoveCreate instead Swap to fill empty place'
        self.backup_value = self.GRID[self.n1,self.n2,self.n3]

    def reject(self):
        self.GRID[self.n1, self.n2, self.n3] = self.backup_value

    def __str__(self):
        #('* Move: %s \t' % move.get_name())
        #self.logfile.write(' Pos.: ['+str(move.n1)+','+str(move.n2)+','+str(move.n3)+'] \t')
        return Move.__str__(self) + self.str_template % (self.n1, self.n2, self.n3, self.backup_value, self.Zatom)

#######################################################################

if __name__ == '__main__':
    from ase.io import write
    mc = MC(log="mc.log")
    #mc = MC(log="-")
    #mc.setup( [2], temperature=T)
    target_CN = np.zeros(4)
    target_CN[0] = 2.8  # Cu-Cu
    target_CN[1] = 4.0  # Pt-Cu
    target_CN[2] = 1.8  # Cu-Pt
    target_CN[3] = 6.2  # Pt-Pt
    #i = 0
    #for B in mc.chems:
    #    for A in mc.chems:
    #        print('CN [',A,'-',B,'] = ', target_CN[i])
    #        i += 1
    mc.set_targets( target_CN, [0.47, 0.53], temperature=1000)

    if False:
        atoms = mc.get_atoms()
        mc.calc_neighbors()
        mc.calc_CNs()
        print(' CN[0] = %f' % mc.CNs[0])
        print(' CN[1] = %f' % mc.CNs[1])
        print(' CN[2] = %f' % mc.CNs[2])
        print(' CN[3] = %f' % mc.CNs[3])
        from qsar import QSAR
        q = QSAR(atoms)
        #q.monoatomic()
        q.biatomic('Pt','Cu')
        #print('CN(list method) = %f' % q.CN)
        print('CNs(list method) \n CuCu %f\tPtCu %f\t CuPt %f\t PtPt %f\n'
          % (q.CN_BB, q.CN_BA, q.CN_AB, q.CN_AA) )
        print('Natoms = %f' % mc.get_N())
        from ase.visualize import view
        view(atoms)
        raw_input('Press enter')

    step = 0
    while step < 5:
        #for i in range(20):
        mc.clear_stats()
        mc.set_targets( target_CN, [0.47, 0.53], temperature=1000/(2**step) )
        mc.run(400)
        atoms = mc.get_atoms()
        write('xyz/dumb'+str(step)+'.cube', atoms)
        step += 1


    print('** See you! **')
