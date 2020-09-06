#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 03:12:14 2020

@author: shirongbao
"""
from qutip import *
import numpy as np
from numpy import linalg as LA
from Hamiltonian import initialize_Hamiltonian
from pulseDriven import evol_operators

def fid_transform_matrix(Ham, method):
    """
    Transformation matrix used in the modified fidelity expression
    """
    if method == 'singleCav':
        return tensor(Ham.id_cavity(size='hop'), basis(Ham.cavSys.N_q, 0) * basis(Ham.cavSys.N_q, 0).dag())
    elif method == 'doubleCav':
        return tensor(Ham.cav_1.id_cavity(size='hop'), basis(Ham.cav_1.cavSys.N_q, 0) * basis(Ham.cav_1.cavSys.N_q, 0).dag(),
                      Ham.cav_2.id_cavity(size='hop'), basis(Ham.cav_2.cavSys.N_q, 0) * basis(Ham.cav_2.cavSys.N_q, 0).dag())

def fid_tranform(U, Ham, method = 'def'):
    """
    methods are
    'def' - default. Returning the original matrix
    'singleCav' or 'doubleCav' - Transform the matrix product to only leave the concerned entries non-zero
    """
    if method == 'def':
        return U
    elif method == 'singleCav' or method == 'doubleCav':
        return fid_transform_matrix(Ham = Ham, method = method) * U

def fid(U_targ, U_evol, Ham, method = 'def'):
    """
    methods are
    'def' - default. Fidelity expression of the full matrix
    'singleCav' or 'doubleCav' - Modified expression that only concerns with the effective entries
    """
    if method == 'def':
        return np.abs((U_targ.inv() * U_evol).tr())/(Ham.cavSys.N_cav * Ham.cavSys.N_q)
    elif method == 'singleCav':
        return np.abs((fid_tranform(U_targ.inv() * U_evol, Ham=Ham, method=method)).tr()) / Ham.fermSys.N_dim
    elif method == 'doubleCav':
        return np.abs((fid_tranform(U_targ.inv() * U_evol, Ham = Ham, method = method)).tr())/(Ham.cav_1.fermSys.N_dim * Ham.cav_2.fermSys.N_dim)
    elif method == 'other':
        return np.abs((U_targ.inv() * U_evol).tr())/U_evol.shape[0]

def energy(state, Hamiltonian):
    """
    Returns state's expected energy of a given Hamilonain
    """
    return (state.dag() * Hamiltonian * state).full()[0,0].real

def orderedEig(Hamiltonian):
    """
    Return ordered eigenstates and eigen-energies of the Hamiltonian
    """
    w, v = LA.eig(Hamiltonian)
    p = w.argsort()
    return w[p], v.T[p]

def groundEnergy(Hamiltonian, index=0):
    """
    Return the ground state energy.
    """
    w, v = orderedEig(Hamiltonian)
    return w[index].real

def HilbertSpaceProjection(state, Hamiltonian):
    """
    Return inner product of given state with a Hamiltonian's eigenstates.
    """
    w, v = orderedEig(Hamiltonian)
    innerProd = np.abs(v.dot(state)).T.tolist()[0]
    return w, innerProd

def checkHilbertSapceTruncation(targ_param, optimResult, evo_time = None):
    """
    Propogate the system under target parameters (normally a big enough cavity level) and
    return the evolved unitary/expected energy for the pulse-driven system.
    """
    if evo_time == None:
        evo_time = targ_param["evo_time"]

    if targ_param["cavity"] == 1:
        mode = 'singleCav'
    elif targ_param["cavity"] == 2:
        mode = 'doubleCav'
    else:
        raise Exception('Not defined yet')

    target_Ham = initialize_Hamiltonian(targ_param)

    if targ_param["problem"] == 'dynamics':
        target_Unitary = target_Ham.get_targUnitary(evo_time)
        operatorList = evol_operators(target_Ham, optimResult, initial=None)
        return fid(target_Unitary, operatorList[-1], target_Ham, method=mode)
    elif targ_param["problem"] == 'variational':
        if targ_param["cavity"] == 1:
            initial = target_Ham.cavSys.state(0)
        elif targ_param["cavity"] == 2:
            # initial = tensor(target_Ham.cav_1.cavSys.state(0), target_Ham.cav_1.cavSys.state(0))
            initial_state = targ_param["initial_state"]
            if initial_state == 'sym':
                initial = tensor(target_Ham.cav_1.cavSys.state(0), target_Ham.cav_2.cavSys.state(0))
            elif initial_state == 'antisym':
                initial = (tensor(target_Ham.cav_1.cavSys.state(0), target_Ham.cav_2.cavSys.state(1)) - tensor(target_Ham.cav_1.cavSys.state(1), target_Ham.cav_2.cavSys.state(0))).unit()
        operatorList = evol_operators(target_Ham, optimResult, initial=initial)
        return expect(target_Ham.get_fermionHam(), operatorList[-1])