#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 03:13:45 2020

@author: shirongbao
"""
from qutip import *
import numpy as np
from systems import fermion, singleCavity, sites

def initialize_Hamiltonian(params):
    """
    Initialize either singleCavHam or doubleCavHam object according to the number of cavities specified
    """
    if params["cavity"] == 1:
        return singleCavHam(params)
    elif params["cavity"] == 2:
        return doubleCavHam(params)
    else:
        raise Exception('Not defined yet')

class Hamiltonian(object):
    """
    Base class for Hamiltonian objects. Should not be called.
    Attributes
    ----------
    initialize_params : function
        Initialize the parameters

    get_NNHam : function
        Get nearest-neighbor interaction Hamiltonian

    get_hopHam : function
        Get hopping Hamiltonian

    get_siteHam : function
        Get site Hamiltonian

    get_penaltyHam : function
        Get penalty Hamiltonian for the auxiliary levels of the cavity levels. Only needed for variational method

    get_fermionHam : function
        Get total Hamiltonian on the fermions in the Hubbard model

    get_driftHam : function
        Get the drift Hamiltonian of the cavity-qubit system

    get_controls : function
        Get the control Hamiltonian

    get_cavHam: function
        Get the sum of drift Hamiltonian and Hamiltonian on fermions

    get_targUnitary: function
        Get target unitary at certain time

    id_full: function
        Returns identity matrix over the whole Hilbert space

    id_cavity: function
        Returns matrix with zero entries exceot identity matrix over the effective Hilbert space

    zero: function
        Returns null matrix

    extract_hopping: function
        Only leave the effective matrix entries to be non-zero

    print_hop_unitary: function
        Print the effective matrix entries for comparison

    broadcast: function
        Broadcast a smaller matrix to the full Hilbert space

    focus: function
        Focus into a smaller number of entries (normally the effective levels) of a large matrix
    """
    def __init__(self, params, fermSys = None, cavSys = None, site = None):
        self.fermSys = fermSys
        self.cavSys = cavSys
        self.site = site

        # self.id = None
        self.initialize_params(params)

        self.ctrls = None
        self.ctrls_label = None
        self.N_ctrl = len(self.get_controls())
    def initialize_params(self, params):
        pass

    def get_NNHam(self):
        pass
    def get_hopHam(self):
        pass
    def get_siteHam(self):
        pass
    def get_penaltyHam(self):
        pass
    def get_fermionHam(self):
        # return self.get_hopHam() + self.get_NNHam() + self.get_siteHam()
        return self.get_hopHam() + self.get_NNHam() + self.get_siteHam() + self.get_penaltyHam()
    def get_driftHam(self):
        pass
    def get_controls(self):
        pass
    def get_cavHam(self):
        return self.get_driftHam() + self.get_fermionHam()
    def get_targUnitary(self, evo_time):
        return (-1j * self.get_cavHam() * evo_time).expm()
    def id_full(self):
        pass
    def id_cavity(self):
        pass
    def zero(self, N):
        return Qobj(np.zeros((N, N), dtype = complex))
    def extract_hopping(self):
        pass
    def print_hop_unitary(self):
        pass

    def broadcast(self, A, N, M = None):
        if M == None:
            M = N
        H = np.zeros((N, M), dtype=complex)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                H[i, j] = A[i, j]
        H = Qobj(H)
        return H
    def focus(self, A, N):
        H = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                H[i, j] = A[i, j]
        H = Qobj(H)
        return H

class singleCavHam(Hamiltonian):
    """
    Single cavity Hamiltonian object.
    """
    def initialize_params(self, params):
        if self.fermSys == None or self.cavSys == None or self.site == None:
            if params["N_e_list"][1] != 0 and params["N_cav_list"][1] != 0:
                raise Exception('Only one non zero')
            self.fermSys = fermion(dict(N_site = params["N_site"], N_e = params["N_e_list"][0]))
            self.cavSys = singleCavity(dict(N_cav=params["N_cav_list"][0]))
            self.site = sites(params)
        self.trans_matrix = self.get_trans()

    def get_trans(self):
        if self.fermSys.N_e == 0:
            A = Qobj(np.array([1]))

        elif self.fermSys.N_e == 1:
            A = Qobj(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                               [0, 0, 0, 1]]))

        elif self.fermSys.N_e == 2:
            A = Qobj(np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0],
                               [0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]]))
        elif self.fermSys.N_e == 3:
            A = Qobj(np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1],
                               [0, 1, 1, 1]]))
        else:
            raise Exception('Not prepared')
        # return A
        return self.broadcast(A, self.cavSys.N_cav, self.site.N_site)
    def get_NNHam(self):
        return tensor(self.zero(self.cavSys.N_cav), self.zero(self.cavSys.N_q))
    def get_penaltyHam(self):
        diag = self.site.penalty * np.ones((self.cavSys.N_cav,))
        diag[:self.fermSys.N_dim] = np.zeros((self.fermSys.N_dim, ))
        return tensor(Qobj(np.diag(diag)), qeye(self.cavSys.N_q))

    def get_hopHam(self):
        if self.fermSys.N_e == 0:
            H_hop = self.zero(self.fermSys.N_dim)

        elif self.fermSys.N_e == 1 or self.fermSys.N_e == 3:
            H_hop = tunneling(self.fermSys.N_site, 1)
            H_hop += basis(self.fermSys.N_site, 0) * basis(self.fermSys.N_site, self.fermSys.N_site - 1).dag() + basis(
                self.fermSys.N_site, self.fermSys.N_site - 1) * basis(self.fermSys.N_site, 0).dag()
            H_hop = -self.site.J * H_hop

        elif self.fermSys.N_e == 2:
            N_dim = self.fermSys.N_dim
            H_add = self.zero(N_dim) + basis(N_dim, 0) * basis(N_dim, 2).dag() - basis(N_dim, 0) * basis(N_dim,3).dag() \
                    + basis(N_dim, 1) * basis(N_dim, 2).dag() + basis(N_dim, 1) * basis(N_dim, 3).dag() \
                    + basis(N_dim, 2) * basis(N_dim, 4).dag() - basis(N_dim, 2) * basis(N_dim, 5).dag() \
                    + basis(N_dim, 3) * basis(N_dim, 4).dag() + basis(N_dim, 3) * basis(N_dim, 5).dag()

            H_hop = (-self.site.J) * (H_add + H_add.dag())
        else:
            raise Exception('Not prepared')

        H_cav = self.broadcast(H_hop, self.cavSys.N_cav)
        return tensor(H_cav, qeye(self.cavSys.N_q)) * self.cavSys.P_g

    def get_siteHam(self):
        H_site = self.trans_matrix * np.diag(self.site.siteMag) * self.trans_matrix.trans()
        H_site = np.diag(np.diag(H_site))
        return tensor(Qobj(H_site), qeye(self.cavSys.N_q)) * self.cavSys.P_g

    def get_driftHam(self):
        return self.site.chi * self.cavSys.ad * self.cavSys.a * self.cavSys.P_e
    
    def get_controls(self):
        if self.ctrls == None:
            self.ctrls_label = ["a+a^dag ", "1j * (a-a^dag) ", "sigma_x_a ", "sigma_y_a "]
            self.ctrls = [self.cavSys.a+self.cavSys.ad, (self.cavSys.a-self.cavSys.ad) * 1j,self.cavSys.sm+self.cavSys.smd, (self.cavSys.sm-self.cavSys.smd) * 1j]
        else:
            pass
        return self.ctrls

    def id_cavity(self, size='full'):
        if size == 'full':
            return qeye(self.cavSys.N_cav)
        elif size == 'hop':
            _I = qeye(self.fermSys.N_dim)
            return self.broadcast(_I, self.cavSys.N_cav)

    def id_full(self):
        return tensor(self.id_cavity(), qeye(self.cavSys.N_q))
    
    def extract_hopping(self, U):
        U_hop = tensor(self.id_cavity(size = 'hop'), basis(self.cavSys.N_q,0).dag()) * U * tensor(self.id_cavity(size = 'hop'), basis(self.cavSys.N_q,0))
        hopMat = U_hop.full()
        return self.focus(hopMat, self.fermSys.N_dim)
    
    def print_hop_unitary(self, U, method = 'original'):
        U_hop = self.extract_hopping(U)
        
        np.set_printoptions(precision=3, suppress = True)
        
        if method == 'abs':
            print(np.abs(U_hop))
        elif method == 'abs_sq':
            print(np.power(np.abs(U_hop),2))
        elif method == 'real_imag':
            print(U_hop.full().real)
            print(U_hop.full().imag)
        elif method == 'original':
            print(U_hop)
            
        np.set_printoptions(edgeitems=3, infstr='inf',
                            linewidth=75, nanstr='nan', precision=8,
                            suppress=False, threshold=1000, formatter=None)

class doubleCavHam(Hamiltonian):
    """
    Double cavity Hamiltonian object. The individual cavities are each initialized to be a single cavity Hamiltonain object.
    """
    def initialize_params(self, params):
        self.site = sites(params)
        ferm_1_param = dict(N_site = params["N_site"], N_e = params["N_e_list"][0])
        ferm_2_param = dict(N_site = params["N_site"], N_e = params["N_e_list"][1])
        cav_1_param = dict(N_cav=params["N_cav_list"][0])
        cav_2_param = dict(N_cav=params["N_cav_list"][1])

        self.cav_1 = singleCavHam(params, fermion(ferm_1_param), singleCavity(cav_1_param), self.site)
        self.cav_2 = singleCavHam(params, fermion(ferm_2_param), singleCavity(cav_2_param), self.site)

        # self.id = 'double cavity'
        self.cavCtrlType = params["cavCtrlType"]
        self.intCtrlType = params["intCtrlType"]

    def get_NNHam(self):
        H_NN = tensor(self.zero(self.cav_1.cavSys.N_cav), self.zero(self.cav_1.cavSys.N_q),
                      self.zero(self.cav_2.cavSys.N_cav), self.zero(self.cav_2.cavSys.N_q))
        for i in range(self.cav_1.fermSys.N_dim):
            for j in range(self.cav_2.fermSys.N_dim):
                s_1 = basis(self.cav_1.cavSys.N_cav, i)
                s_2 = basis(self.cav_2.cavSys.N_cav, j)
                # print((self.cav_2.trans_matrix.trans() * s_2).dag() * self.cav_1.trans_matrix.trans() * s_1)
                H_NN += self.site.U * ((self.cav_2.trans_matrix.trans() * s_2).dag() * self.cav_1.trans_matrix.trans() * s_1) * tensor(
                    s_1 * s_1.dag(), qeye(self.cav_1.cavSys.N_q), s_2 * s_2.dag(), qeye(self.cav_2.cavSys.N_q))

        return H_NN * tensor(self.cav_1.cavSys.P_g, self.cav_2.cavSys.P_g)

    def get_hopHam(self):
        return (tensor(self.cav_1.get_hopHam(), self.cav_2.cavSys.I) + tensor(self.cav_1.cavSys.I, self.cav_2.get_hopHam()))\
               * tensor(self.cav_1.cavSys.P_g, self.cav_2.cavSys.P_g)
    def get_penaltyHam(self):

        return tensor(self.cav_1.get_penaltyHam(), self.cav_2.cavSys.I) + tensor(self.cav_1.cavSys.I,
                                                                              self.cav_2.get_penaltyHam())

    def get_siteHam(self):
        return (tensor(self.cav_1.get_siteHam(), self.cav_2.cavSys.I) + tensor(self.cav_1.cavSys.I, self.cav_2.get_siteHam()))\
               * tensor(self.cav_1.cavSys.P_g, self.cav_2.cavSys.P_g)

    def get_driftHam(self):
        return tensor(self.cav_1.get_driftHam(), self.cav_2.cavSys.I) + tensor(self.cav_1.cavSys.I, self.cav_2.get_driftHam())
    
    def get_controls(self):
        if self.ctrls == None:
            self.cav_1_ctrl = self.cav_1.get_controls()
            self.cav_2_ctrl = self.cav_2.get_controls()
            cav_1 = []
            cav_2 = []
            if len(self.cav_1_ctrl) == len(self.cav_2_ctrl):
                for i in range(len(self.cav_1_ctrl)):
                    cav_1.append(tensor(self.cav_1_ctrl[i], self.cav_2.cavSys.I))
                    cav_2.append(tensor(self.cav_1.cavSys.I, self.cav_2_ctrl[i]))
            else:
                raise Exception('Not equal length controls')

            bs = [tensor(self.cav_1.cavSys.ad, self.cav_2.cavSys.a) + tensor(self.cav_1.cavSys.a, self.cav_2.cavSys.ad),
                  1j * (tensor(self.cav_1.cavSys.ad, self.cav_2.cavSys.a) - tensor(self.cav_1.cavSys.a, self.cav_2.cavSys.ad))]
            sq = [tensor(self.cav_1.cavSys.ad, self.cav_2.cavSys.ad) + tensor(self.cav_1.cavSys.a, self.cav_2.cavSys.a),
                  1j * (tensor(self.cav_1.cavSys.ad, self.cav_2.cavSys.ad) - tensor(self.cav_1.cavSys.a,self.cav_2.cavSys.a))]

            cav_1_labels = ["a+a^dag ", "1j * (a-a^dag) ", "sigma_x_a ", "sigma_y_a "]
            cav_2_labels = ["b+b^dag ", "1j * (b-b^dag) ", "sigma_x_b ", "sigma_y_b "]
            bs_labels = ["a^dag * b + a * b^dag", "1j * (a^dag * b - a * b^dag)"]
            sq_labels = ["a^dag * b^dag + a * b", "1j * (a^dag * b^dag - a * b)"]

            self.ctrls = []
            self.ctrls_label = []

            if self.cavCtrlType == 'independent':
                self.ctrls += cav_1
                self.ctrls += cav_2
                self.ctrls_label += cav_1_labels
                self.ctrls_label += cav_2_labels
            elif self.cavCtrlType == 'identical':
                id = []
                for i in range(len(cav_1)):
                    id.append(cav_1[i] + cav_2[i])
                self.ctrls += id
                self.ctrls_label += np.char.add(np.array(cav_1_labels), np.array(cav_2_labels)).tolist()
            elif self.cavCtrlType == 'no_cav':
                pass
            else:
                raise Exception('not defined')

            if self.intCtrlType == 'bs':
                self.ctrls += bs
                self.ctrls_label += bs_labels
            elif self.intCtrlType == 'sq':
                self.ctrls += sq
                self.ctrls_label += sq_labels
            elif self.intCtrlType == 'both':
                self.ctrls += bs
                self.ctrls_label += bs_labels

                self.ctrls += sq
                self.ctrls_label += sq_labels
            elif self.intCtrlType == 'sym_bs':
                self.ctrls += bs[:1]
                self.ctrls_label += bs_labels[:1]
            elif self.intCtrlType == 'sym_both':
                self.ctrls += bs[:1]
                self.ctrls_label += bs_labels[:1]

                self.ctrls += sq
                self.ctrls_label += sq_labels
            elif self.intCtrlType == 'no_int':
                pass
            else:
                raise Exception('not defined')
        else:
            pass
        return self.ctrls
    
    def id_full(self):
        return tensor(self.cav_1.cavSys.I, self.cav_2.cavSys.I)

    def id_cavity(self, size='full'):
        return tensor(self.cav_1.id_cavity(size = size), qeye(self.cav_1.cavSys.N_q), self.cav_2.id_cavity(size = size), qeye(self.cav_2.cavSys.N_q))