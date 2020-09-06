#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 01:46:58 2020

@author: shirongbao
"""

import numpy as np
from qutip import *
from Hamiltonian import initialize_Hamiltonian
from plot import scatterPlot
from pulseDriven import readPulse, pulse_Ham, evol_operators
from fermionMapping import electronDenMapping

def targEvol(Ham, t_list, init_state, observables = []):
    targHam = Ham.get_cavHam()

    output = mesolve(targHam, init_state, t_list, [], observables)

    return output.states, output.expect

def pulseEvol(Ham, init_state, optimResult, method = 'mesolve', observables = []):
    t_list = optimResult.time
    if method == 'mesolve':
        drivenHam = pulse_Ham(Ham, optimResult)
        output = mesolve(drivenHam, init_state, t_list, [], observables)
        return output.states, output.expect
    elif method == 'operators':
        statesList = evol_operators(Ham, optimResult, initial = init_state)

        expectationList = []
        if len(observables) > 0:
            expectationList = expect(observables, statesList)

        return statesList, expectationList

def multiple_pulse_sequence(Ham, init_state, optimResult, n_period, method = 'mesolve', observables = []):
    start_state = init_state

    n_obs = len(observables)

    statesList = []
    expectationList = [[]] * n_obs
    for i in range(n_period):
        _states, _expect = pulseEvol(Ham, start_state, optimResult, method = method, observables = observables)
        
        if len(statesList) == 0:
            statesList += _states
        else:
            statesList += _states[1:]

        for j in range(n_obs):
            if len(expectationList[j]) == 0:
                expectationList[j] = _expect[j]
            else:
                expectationList[j] = np.concatenate((expectationList[j], _expect[j][1:]), axis=None)
        start_state = _states[-1]
    return statesList, expectationList

def multi_timeList(times, n_period):
    _period = times.copy()
    timeList = []
    for i in range(n_period):
        if len(timeList) == 0:
            timeList = _period
        else:
            timeList = np.concatenate((timeList, _period[1:] + timeList[-1]), axis=None)
    return timeList

def optimResultToTargetDens(optimResult, Ham, n_period = 5):

    if Ham.id == 'single cavity':
        psi0 = Ham.cavSys.state(0)
        observables = [Ham.cavSys.state(i) * Ham.cavSys.state(i).dag() for i in range(Ham.fermSys.N_dim)]
        stateLabels = ['Level ' + str(i) for i in range(Ham.fermSys.N_dim)]
    elif Ham.id == 'double cavity':
        psi0 = (Ham.cavSys.state(0, 0) + Ham.cavSys.state(0, 1)).unit()
        observables = []
        stateLabels = []
        for i in range(Ham.fermSys.N_dim_up):
            for j in range(Ham.fermSys.N_dim_down):
                observables.append(Ham.cavSys.state(i, j) * Ham.cavSys.state(i, j).dag())
                stateLabels.append(str(i) +  ', ' + str(j))

    t_list = multi_timeList(optimResult.time, n_period = n_period)

    period_indices = [i * (len(optimResult.time)-1) for i in range(n_period + 1)]
    targStates, targExpectations = targEvol(Ham, t_list = t_list, init_state = psi0, observables = observables)
    drivenStates, drivenExpectations = multiple_pulse_sequence(Ham, init_state = psi0, optimResult = optimResult, n_period = n_period, method = 'operators', observables = observables)
    return targExpectations, drivenExpectations, t_list, stateLabels, period_indices
if __name__ == "__main__":
    data_dir = '.'
    fileName = 'optimResult'
    optimResult = readPulse(data_dir = data_dir, fileName = fileName)
    Ham = initialize_Hamiltonian(optimResult.params)

    n_period = 50
    targExpectations, drivenExpectations, t_list, stateLabels, period_indices = optimResultToTargetDens(optimResult, Ham, n_period = n_period)

    siteLabels = ['Site ' + str(i) for i in range(Ham.fermSys.N_site)]

    targ_eDens = electronDenMapping(Ham, targExpectations)
    driven_eDens = electronDenMapping(Ham, drivenExpectations)

    scatterPlot(t_list, targExpectations, labels = stateLabels, 
                plot_dir = data_dir, fileName = 'stateDens_targ.png', 
                title = 'Evolution of state probabilities', xlabel = 'time/microseconds', ylabel = 'probability', ifAnnotate = False, plotType = 'norm', ifLegend = False)

    point_indices = list(range(0, n_period + 1, 2))
    period_indices = [i * (len(optimResult.time) - 1) for i in point_indices]
    n_points = len(point_indices)
    sum_abs_diff = []
    for j in range(2):
        for i in range(Ham.fermSys.N_site):
            scatterPlot(t_list, [targ_eDens[j][i], driven_eDens[j][i]], labels = ['targ', 'pulse'],
                    plot_dir = data_dir, fileName = 'site' + str(i) + '_spin_' +str(j) + '.png',
                    title = siteLabels[i], xlabel = 'time/microseconds', ylabel = 'electron density', ylimit = (-0.05,1.05), ifAnnotate = False, plotType = 'norm',
                    other = [t_list[period_indices], driven_eDens[j][i][period_indices], 'pulse at ends of pulse sequences'])

        print(np.sum(np.array(targ_eDens[j])[:, -1]))
        print(np.sum(np.array(driven_eDens[j])[:, -1]))

        sum_abs_diff.append(np.sum(np.array(targ_eDens[j])[:, period_indices] - np.array(driven_eDens[j])[:, period_indices], axis=0))

    scatterPlot(point_indices, sum_abs_diff, labels=['Spin up', 'Spin down'],
                plot_dir=data_dir, fileName='eDen_Leakage.png',
                title='Electron leakage vs number of pulse sequences applied', xlabel='number of pulse sequences', ylabel='electron leakage density',
                ifAnnotate=True, plotType='large', ifScatter=True)

    # for i in range(len(targExpectations)):
    #     print(targExpectations[i][-1])
    # scatterPlot(t_list, drivenExpectations, labels = stateLabels, 
    #         plot_dir = data_dir, fileName = 'stateDens_pulse.png', 
    #         title = 'Pulse driven state probabilities', xlabel = 'time/microseconds', ylabel = 'probability', ifAnnotate = False, plotType = 'norm')




    #
    # point_indices = list(range(0, n_period+1, 4))
    # period_indices = [i * (len(optimResult.time)-1) for i in point_indices]
    # n_points = len(point_indices)
    #
    # print(np.sum(np.array(targ_eDens)[:, -1]))
    # print(np.sum(np.array(driven_eDens)[:, -1]))
    #
    # sum_abs_diff = np.sum(np.array(targ_eDens)[:, period_indices] - np.array(driven_eDens)[:, period_indices], axis = 0)
    #
    #
    # scatterPlot(point_indices, [sum_abs_diff], labels = ['Electron leakage'],
    #                     plot_dir = data_dir, fileName = 'eDen_Leakage.png',
    #                     title = 'Electron leakage', xlabel = 'number of pulse sequences', ylabel = 'electron density', ifAnnotate = True, plotType = 'large', ifScatter = True)



