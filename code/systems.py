#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 03:12:01 2020

@author: shirongbao
"""

from qutip import *
from util import comb
import numpy as np

class fermion:
    """
    The fermion system object.
    Attributes
    ----------
    N_site: integer
        Number of sites

    N_e: integer
        Number of electrons (of the same spin flavor)

    N_dim: integer
        Dimension of effective Hilbert space
    """
    def __init__(self, params):
        
        self.N_site = params["N_site"]
        self.N_e = params["N_e"] # Integer
        self.N_dim = None
        self.get_HilbertSpaceDim()

    def get_HilbertSpaceDim(self):
        self.N_dim = comb(self.N_site, self.N_e)
        return self.N_dim

class sites:
    """
    Site system object.
    Attributes
    ----------
    chi: float
        Drift Hamiltonian magnitude

    J: float
        Hopping Hamiltonian magnitude. Also expressed in t in my reports.

    siteMag: float
        The site Hamiltonian

    U: float
        Nearest-neighbor interaction magnitude
    """
    def __init__(self, params):
        self.chi = params["chi"]
        self.J = self.chi * params['J_ratio']
        self.site_energy, self.site_energy_method, self.site_energy_seed = params["site_energy_params"]
        self.U = self.chi * params['U_ratio']
        if params["problem"] == "variational":
            self.penalty = params['penalty']
        else:
            self.penalty = 0

        self.N_site = params["N_site"]
        if self.site_energy_method == 'random':
            if len(self.site_energy) != 1:
                raise Exception('Only one scaling')
            np.random.seed(self.site_energy_seed)
            self.siteMag = list((np.random.rand(params['N_site'], ) * 2 - 1) * self.site_energy)

        elif self.site_energy_method == 'fix':
            if len(self.site_energy) != params['N_site']:
                raise Exception('Wrong dimension')
            self.siteMag = [-1 * _site for _site in self.site_energy]
        self.siteMag = np.array(self.siteMag)

class singleCavity:
    """
    Single cavity object.
    Attributes
    ----------
    N_cav: integer
        Number of levels for the truncated Hilbert space

    N_q: integer
        Number of levels in the qubit
    """
    def __init__(self, params):

        self.N_cav = params["N_cav"]
        self.N_q = 2
        self.initiateOperators()

    def initiateOperators(self):
        self.I = tensor(qeye(self.N_cav), qeye(self.N_q))

        self.a = tensor(destroy(self.N_cav), qeye(self.N_q))
        self.ad = self.a.dag()
        self.sm = tensor(qeye(self.N_cav), destroy(self.N_q))
        self.smd = self.sm.dag()

        self.P_e = tensor(qeye(self.N_cav), basis(self.N_q, 1) * basis(self.N_q, 1).dag())
        self.P_g = tensor(qeye(self.N_cav), basis(self.N_q, 0) * basis(self.N_q, 0).dag())
    
    def state(self, cav, qubit=0):
        return tensor(basis(self.N_cav, cav), basis(self.N_q, qubit))
    