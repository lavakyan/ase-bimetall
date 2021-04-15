from __future__ import print_function

from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.data import atomic_numbers
import sys
import os
import numpy as np
from scipy.special import i0 as bessel_i0  # used for Kaiser FT widnow


class EXAFS(FileIOCalculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, chi_filename=None, kmin=3, kmax=11, kw=1, edge='K',
                 absorber='Ag', atom_index=None, cluster_radius=10.0,
                 feff='~/bin/feff85', logfile='-'):
        ''' Calculator for the fitting of EXAFS data.
        Instead of energy it provides the misfit beetween experimental
        data and theory. Theory is calculated by external code,
        in particular, Feff8.5L available free of charge on official
        website <http://feffproject.org>.

        Parameters:

        chi_filename:
            the exprimental EXAFS oscillation function chi(k) to compare
            results with if set to None, no comparison performed.
        k_min, k_max:
            k-range for FT
        kw:
            k-weight of chi(k) function to emphasise far oscillations.
            Usually is set to
                1 (light neighbor atoms),
                2 or 3 (heavy neighbor atoms).
        edge:
            'K' or 'L3'
        absorber:
            symbol of absorbing element.
        atom_index:
            index of X-ray absorbing atoms or None.
            If None then spectra will be averaged using
            CFAVERAGE flag (Feff9 or newer required).
        cluster_radius:
            cutoff radius of cluster.
            Specify None if no need to cut cluster.
        feff:
            command to run FEFF.
        logfile:
            file to write debug info.
            '-' value will write to stdout.

        Note:
            This calculator produced not energy, but misfit between
            computed and exerimental EXAFS in R-space!
            IMPORTANT: Don't rely on potential_energy() returned value
            in MD simulations!!!
            Caclulation performed using call of external code,
            Feff85 or later, freely available.
        Based on:
            ASE caclulator interface <wiki.fysik.dtu.dk/ase/>,
            X-ray Larch <http://xraypy.github.io/xraylarch>
            and inspired by EvAX <http://www.dragon.lv/evax/>
        '''
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

        self.natoms = 0
        self.exp_group = None
        self.theor_group = None
        self.prefix = ''
        self.feff = feff
        self.directory = 'temp_feff_calc'
        self.command = self.feff + ' > feff.out'

        self.kmin = kmin
        self.kmax = kmax
        self.kweight = kw
        self.dk = 0.5
        self.rmin = 1.5
        self.rmax = 3.1
        self._load_exp(chi_filename)  # kmin kmax kw must be set before

        self.S02 = 1.0
        self.edge = edge
        self.absorber = absorber
        self.atom_index = atom_index
        self.cluster_radius = cluster_radius
        # Feff parameters
        self.n_scf = 15   # number of self-consistency steps for FMS
        self.r_scf = 0.1  # radius of FMS cluster (small value if no FMS)
        self.rpath = 7.0  # radius of chain scattering cluster
        self.nlegs = 4    # maximum scattering order (2 = single scattering)

        if self.log is not None:
            self.log.write(' Command:\t%s\n Directory:\t%s \n Chi file:\t%s\n'
                           % (self.command, self.directory, self.chi_filename))

    def _load_exp(self, chi_filename):
        self.chi_filename = chi_filename
        if chi_filename is not None:
            data = np.genfromtxt(chi_filename)
            # chi.dat:  k - column 1, chi - column 2
            self.exp_k = data[:, 0]
            self.exp_chi = data[:, 1]
            # do FT. Result will be pushed to exp_group
            self.exp_R, self.exp_F = exafs_ft(self.exp_k, self.exp_chi,
                                              kmin=self.kmin, kmax=self.kmax,
                                              kweight=self.kweight, dk=self.dk)

    def _load_theor(self):
        data = np.genfromtxt(os.path.join(self.directory, 'xmu.dat'))
        # xmu.dat:  k - column 3, chi - column 6
        self.theor_k = data[:, 2]
        self.theor_chi = data[:, 5]
        # sometimes k arrays contains zero, and sometimes - not. Why?
        # This leads to problems in cumulation and averaging.
        # temporary solution is to remove k=0
        if np.abs(self.theor_k[0]) < 1e-5:
            self.theor_k = self.theor_k[1:]
            self.theor_chi = self.theor_chi[1:]
        # do FT
        self.theor_R, self.theor_F = exafs_ft(self.theor_k, self.theor_chi,
                                              kmin=self.kmin, kmax=self.kmax,
                                              kweight=self.kweight, dk=self.dk)

    def residual(self):
        if self.chi_filename is None:
            return -1
        else:
            rwin = ftwindow(self.exp_R, xmin=self.rmin, xmax=self.rmax,
                            dx=0.1, window='hanning')
            res = np.absolute(np.sum(rwin * (self.exp_F - self.theor_F)))**2
            norm = np.absolute(np.sum(rwin * self.exp_F**2))**2
            # R-factor, %
            return res / norm * 100.0

    def calculate(self, atoms, properties=['energy'], system_changes=['positions', 'numbers']):
        try:
            super().calculate(atoms, properties, system_changes)
        except RuntimeError:
            self.results['energy'] = 999  # big value

    def write_input(self, atoms, properties=None, system_changes=None):
        '''Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically.'''

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        filename = os.path.join(self.directory, 'feff.inp')
        try:
            fout = open(filename, 'w')
        except:
            if self.log is not None:
                self.log.write('Cannot open file "%s"' % filename)
            return

        cluster = atoms.copy()
        if self.atom_index is not None:  # center at abosrber
            cluster.translate(-cluster[self.atom_index].position)
            if self.cluster_radius is not None:  # remove far atoms
                dists2 = np.sum((cluster.get_positions())**2, axis=1)
                rem = np.nonzero(dists2 > self.cluster_radius**2)[0]
                if len(rem) > 0:
                    del cluster[rem]
        # used later:
        positions = cluster.get_positions()
        distances = np.sqrt(np.sum((positions)**2, axis=1))

        buff = []
        buff.append(' TITLE   EXAFS-ASE\n')
        buff.append(' EDGE      %s' % self.edge)
        buff.append(' S02       %f\n' % 1.0)  #self.S02)
        buff.append(' *         pot    xsph  fms   paths genfmt ff2chi')
        buff.append(' CONTROL   1      1     1     1     1      1')
        buff.append(' PRINT     1      0     0     0     0      0\n')
        buff.append(' *         r_scf   [ l_scf  n_scf  ca ]')
        buff.append(' SCF       %f   0      %i     0.1\n' %
                    (self.r_scf, self.n_scf))
        buff.append(' *         ixc  [ Vr  Vi ]')
        buff.append(' EXCHANGE  0      0   0\n')
        buff.append(' EXAFS  %f' % (self.kmax + self.dk))  # + dk window
        buff.append(' RPATH    %f' % self.rpath)
        buff.append(' NLEGS    %i' % self.nlegs)
        buff.append(' CRITERIA  4.0  2.5\n')  # path skipping creiteria

        buff.append(' POTENTIALS')
        buff.append(' *   ipot   z [ label   l_scmt  l_fms  stoichiometry ]')
        buff.append('       0   %i    %s     -1      -1       0' %
                    (atomic_numbers[self.absorber], self.absorber))

        species = cluster.get_chemical_symbols()
        dct_spec = {}  # dictionary: atom type -> number of atoms
        for A in species:
            if A in dct_spec:
                dct_spec[A] += 1
            else:
                dct_spec[A] = 1
        if self.atom_index is not None:
            dct_spec[self.absorber] -= 1  # remove the absorber atom from count

        dct_pot = {}   # dictionary: atom type -> ipot
        ipot = 1
        for A in dct_spec:
            if dct_spec[A] > 0:
                buff.append('       %i   %i    %s     -1      -1       %i' %
                            (ipot, atomic_numbers[A], A, dct_spec[A]))
                dct_pot[A] = ipot
                ipot += 1

        buff.append('\n ATOMS')
        buff.append(' *   x    y    z   ipot    distance   '
                    '# this list contains %i atoms' % len(cluster))
        if self.atom_index is not None:  # central (absorbing) atom
            buff.append('   0.000000   0.000000   0.000000   0   0.000000')
        else:  # average over all absorbers
            buff.append('   0.000000   0.000000   0.000000   %i   0.000000' %
                        dct_pot[self.absorber])

        for i in np.argsort(distances):
            dist = distances[i]
            if dist > 0.1:  # if not absorber
                pos = positions[i]
                buff.append('   %f   %f   %f   %i   %f' %
                            (pos[0], pos[1], pos[2],
                             dct_pot[species[i]], dist))

        if self.atom_index is None:  # average over all absorbers!
            buff.append('\nCFAVERAGE %i 0 %f' % (dct_pot[self.absorber],
                                                 self.rpath))
        buff.append(' END')
        try:
            fout.write('\n'.join(buff))
            fout.write('\n')
        except:
            if self.log is not None:
                self.log.write('Cannot write to file "%s"' % filename)
            return

    def read_results(self):
        """Read energy, forces, ... from output file(s)."""
        self._load_theor()
        resid = self.residual()
        self.results['energy'] = resid

    def _do_plot_r(self, plt):
        ymax = 0
        if self.chi_filename is not None:
            plt.plot(self.exp_R, np.absolute(self.exp_F), linestyle='None',
                     markerfacecolor='none', marker='o', label='exp.')
            ymax = np.max(np.absolute(self.exp_F))
        if len(self.theor_R) > 0:
            ymax = max(ymax, self.S02 * np.max(np.absolute(self.theor_F)))
            plt.plot(self.theor_R, self.S02 * np.absolute(self.theor_F),
                     label='model')
            # plt.plot(self.theor_R, np.imag(self.theor_F), '--')
        if self.chi_filename is not None:
            rwin = ftwindow(self.exp_R, xmin=self.rmin, xmax=self.rmax,
                            dx=0.1, window='hanning')
            rwin = rwin / np.max(rwin) * ymax
            plt.plot(self.exp_R, rwin, linewidth=0.5, color='#0F0F0F')
        plt.legend()
        plt.ylim([None, ymax])
        plt.xlabel('R, Angstr.')
        plt.ylabel('|F(R)|, arb.units')

    def _do_plot_k(self, plt, kweight=None):
        kmax = 11
        if kweight is None:
            kweight = self.kweight
        if self.chi_filename is not None:
            kmax = np.max(self.exp_k)
            plt.plot(self.exp_k, self.exp_chi * self.exp_k**kweight,
                     label='exp.')
        if len(self.theor_k) > 0:
            plt.plot(self.theor_k,
                     self.S02 * self.theor_chi * self.theor_k**kweight,
                     label='model')
        # if self.chi_filename is not None:
            # TODO: plot window
        plt.legend()
        plt.xlim([0, kmax])
        plt.xlabel('$k$, Angstr.$^{-1}$')
        plt.ylabel('$\chi(k) \cdot k^%.0f$' % kweight)

    def _do_plot_kw(self, plt, kweights=[1, 2, 3]):
        f, axarr = plt.subplots(len(kweights), 1, sharex=True)
        for i, kweight in enumerate(kweights):
            plt.sca(axarr[i])
            self._do_plot_k(plt, kweight)
            if i != len(kweights) - 1:  # not last
                plt.xlabel('')

    def plot(self, mode='k', rwin_divide=10, filename=None):
        ''' Do the plots and FTs, if specified.
            Parameters:
            mode : {'k'|'r'|'kw'|'all'}
        '''
        import matplotlib.pyplot as plt
        if mode == 'r':
            self._do_plot_r(plt)
        elif mode == 'k':
            self._do_plot_k(plt)
        elif mode == 'kw':
            self._do_plot_kw(plt)

        if filename is None:
            plt.show()
        else:
            # plt.ioff()
            plt.savefig(filename)
            plt.close()
            # plt.ion()

    def save(self, filename):
        np.savetxt(filename,
                   np.transpose(np.stack([self.theor_k, self.theor_chi])),
                   header='k\tchi')


def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None,
             window='hanning', **kws):
    '''
    this code is heavily based on X-ray Larch:
        (c) 2012 Matthew Newville, The University of Chicago
                         <http://xraypy.github.io/xraylarch>
                                <newville@cars.uchicago.edu>

    create a Fourier transform window array.

    Parameters:
    -------------
      x:        1-d array array to build window on.
      xmin:     starting x for FT Window
      xmax:     ending x for FT Window
      dx:       tapering parameter for FT Window
      dx2:      second tapering parameter for FT Window (=dx)
      window:   name of window type

    Returns:
    ----------
    1-d window array.

    Notes:
    -------
    Valid Window names:
        hanning              cosine-squared taper
        kaiser               Kaiser-Bessel function-derived window
    '''
    if window is None:
        window = 'han'
    nam = window.strip().lower()[:3]
    if nam not in ['han', 'kai']:
        raise RuntimeError('invalid window name %s' % window)

    dx1 = dx
    dx2 = dx1
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    xstep = (x[-1] - x[0]) / (len(x) - 1)
    xeps = 0.0001 * xstep
    x1 = max(np.min(x), xmin - dx1 / 2.0)
    x2 = xmin + dx1 / 2.0 + xeps
    x3 = xmax - dx2 / 2.0 - xeps
    x4 = min(np.max(x), xmax + dx2 / 2.0)

    def get_idx(val):
        return int((val + xeps) / xstep)

    i1, i2, i3, i4 = get_idx(x1), get_idx(x2), get_idx(x3), get_idx(x4)
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(x)-1, i3), min(len(x)-1, i4)
    if i2 == i1:
        i1 = max(0, i2-1)
    if i4 == i3:
        i3 = max(i2, i4-1)
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2:
        x2 = x2 + xeps
    if x3 == x4:
        x4 = x4 + xeps
    # initial window
    fwin = np.zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = np.ones(i3-i2)
    # now finish making window
    if nam == 'han':
        fwin[i1:i2+1] = np.sin(np.pi / 2 * (x[i1:i2+1] - x1) / (x2 - x1))**2
        fwin[i3:i4+1] = np.cos(np.pi / 2 * (x[i3:i4+1] - x3) / (x4 - x3))**2
    elif nam == 'kai':
        cen = (x4 + x1) / 2
        wid = (x4 - x1) / 2
        arg = 1 - ((x - cen) / wid)**2
        arg[arg<0] = 0
        scale = max(1.e-10, bessel_i0(dx)-1)
        fwin = (bessel_i0(dx * np.sqrt(arg)) - 1) / scale
    return fwin


def exafs_ft(k, chi, kmin=3, kmax=9, kweight=1, dk=0.1, rmax_out=10,
             kstep=0.05, nfft=2048):
    k_max = np.max(k)
    k_ = np.arange(kstep, k_max, kstep)
    npts = len(k_)
    print('npts: %i' % npts)

    # rebin
    chi_ = np.interp(k_, k, chi)

    # window
    win = ftwindow(x=k_, xmin=kmin, xmax=kmax, dx=dk,
                   window='kaiser')
    # plt.plot(k_, win)
    # plt.show()

    # pad with zeros
    cchi = np.zeros(nfft)
    cchi[:npts] = chi_ * k_**kweight * win

    # do FT
    out = np.fft.fft(cchi)
    out = kstep / np.sqrt(np.pi) * out

    # R scale
    rmax = np.pi / kstep
    r = np.linspace(0, rmax, nfft)

    # cut
    out = out[r < rmax_out]
    r = r[r < rmax_out]

    return r, out


def exafs_wt(k, chi, kweight=2, kmin=None, kmax=None, rmax_out=10, nfft=2048):
    """
    Cauchy Wavelet Transform for XAFS, following work [Munoz M., Argoul P.
    and Farges F. Continuous Cauchy wavelet transform analyses of EXAFS
    spectra: a qualitative approach, American Mineralogist 88,
    pp. 694-700 (2003).]
    Python adaptation by M. Newville for Larch project
    <https://xraypy.github.io/xraylarch/>
    pep8 style edits, etc - L. Avakyan

    Parameters:
    -----------
      k:        1-d array of photo-electron wavenumber in Ang^-1
      chi:      1-d array of chi
      kweight:  exponent for weighting spectra by k**kweight
      kmin:     lower k bound for transform
      kmax:     upper k bound for transform
      rmax_out: highest R for output data (10 Ang)
      nfft:     value to use for N_fft (2048).

      Returns:
    ---------
      (k, R, wcauchy), where
      k:          1-d array of photo-electron wavenumber
                  (could differ from input k)
      R:          1-d array of R, from 0 to rmax_out.
      wcauchy     2-d array of transform result
    """
    if kmin is not None:
        chi = chi[k>kmin]
        k = k[k>kmin]
    if kmax is not None:
        chi = chi[k<kmax]
        k = k[k<kmax]
    # ~ kstep = np.round(1000.*(k[1]-k[0]))/1000.0
    kstep = k[1] - k[0]
    rstep =  np.pi / nfft / kstep
    rmin = 1.e-7
    rmax = rmax_out
    nrpts = int(np.round((rmax - rmin) / rstep))
    nkout = len(k)
    if kweight != 0:
        chi = chi * k**kweight

    NFT = nfft // 2
    if len(k) < NFT:  # extend EXAFS to 1024 data points
        knew = np.arange(NFT) * kstep
        xnew = np.zeros(NFT) * kstep
        xnew[:len(k)] = chi
    else:  # limit EXAFS by 1024 data points
        knew = k[:NFT]
        xnew = chi[:NFT]

    # FT parameters
    freq = 1. / kstep * np.arange(nfft) / (2 * nfft)
    omega = 2 * np.pi * freq

    # simple FT calculation
    tff = np.fft.fft(xnew, n=2*nfft)

    # scale parameter
    r  = np.linspace(0, rmax, nrpts)
    r[0] = 1.e-19
    a  = nrpts / (2 * r)

    # Characteristic values for Cauchy wavelet:
    # ~ cauchy_sum = np.log(2*np.pi) - np.log(1.0 + np.arange(nrpts)).sum()
    cauchy_sum = np.log(2*np.pi) - np.sum(np.log(1.0 + np.arange(nrpts)))

    # Main calculation:
    wcauchy = np.zeros(nkout*nrpts, dtype='complex').reshape(nrpts, nkout)
    for i in range(nrpts):
        aom = a[i] * omega
        aom[np.where(aom==0)] = 1.e-19
        filt = cauchy_sum + nrpts * np.log(aom) - aom
        tmp  = np.conj(np.exp(filt)) * tff[:nfft]
        wcauchy[i, :] = np.fft.ifft(tmp, 2*nfft)[:nkout]

    return k, r, wcauchy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if False:  # test FT
        x = np.arange(1, 10, 0.05)
        y = 5 * np.sin(2 * 2.25 * x + 1.5) * np.exp(-2 * 0.05 * x**2)
        plt.plot(x, y)
        plt.show()
        plt.plot(xft, np.absolute(yft))
        plt.show()
    if True:  # test wavelet
        x = np.arange(1, 12, 0.05)
        y = 5 * np.sin(2 * 2.25 * x + 1.5) * np.exp(-2 * 0.05 * x**2)
        xwt, rwt, result = exafs_wt(x, y)
        xx, yy = np.meshgrid(xwt, rwt)
        plt.pcolormesh(xx, yy, np.absolute(result), cmap='hot_r')
        plt.xlabel('k, 1/Angstr.')
        plt.ylabel('R, Angstr.')
        plt.colorbar()
        plt.show()

    if True:  # test sim
        from ase.cluster.cubic import FaceCenteredCubic
        atoms = FaceCenteredCubic('Pt',
                                  [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                                  [2, 3, 1], 4.09)

        exafs = EXAFS(# chi_filename='nonleached_285PROX.chi',
                      kmin=3.4, kmax=10.5, kw=1, edge='L3',
                      absorber='Pt', atom_index=12,
                      feff='~/bin/feff85')
        exafs.nlegs = 5
        # ~ atoms.set_calculator(exafs)

        exafs.calculate(atoms)

        exafs.plot(mode='k')
        exafs.plot(mode='r')

        exafs.plot(mode='kw')
        exafs.save('chi.dat')
        if True:
            xwt, rwt, result = exafs_wt(exafs.theor_k, exafs.S02 * exafs.theor_chi, kweight=1, kmin=2.5, kmax=10.5)
            xx, yy = np.meshgrid(xwt, rwt)
            plt.pcolormesh(xx, yy, np.absolute(result), cmap='hot_r')
            plt.xlabel('k, 1/Angstr.')
            plt.ylabel('R, Angstr.')
            plt.colorbar()
            plt.show()
        # print('   MISFIT: %.6f' % atoms.get_potential_energy())

