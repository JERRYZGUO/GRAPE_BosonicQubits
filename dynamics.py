#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:55:56 2020

@author: shirongbao
"""

import numpy as np
import qutip.control.pulseoptim as cpo
from qutip import *
import datetime, json

from fidelity import fid, fid_transform_matrix, groundEnergy
from Hamiltonian import initialize_Hamiltonian
from plot import plotCtrlPulses
from util import createDataDir
from pulseDriven import readPulse

def dynamics(params, data_dir = None, ifReport = False, ifPlotPulse = False, ifQsave = False, saveFormat = 'mod',
             fileName = 'optimResult', ifCreateDir = False):
    """
    Main function for running GRAPE algorithm.
    Input:
        params - parameters for GRAPE algorithm and the physical system
        data_dir - save directory path
        ifReport - True for printing the details of GRAPE algorithm run, False for not printing
        ifPlotPulse - True for plotting/saving the pulse of GRAPE algorithm
        ifQsave - True for saving the details of the GRAPE algorithm run in data_dir
        saveFormat - "mod" for only saving certain features of optimization result; "def" for using the default saving format, which saves all features of optimization result
        fileName - name of optimization result file saved
        ifCreateDir - True for creating a new directory to save the results
    Output:
        output - selected optimized parameters from GRAPE
    """
    if params["cavity"] == 1:
        mode = 'singleCav'
    elif params["cavity"] == 2:
        mode = 'doubleCav'
    else:
        raise Exception('Not defined yet')

    # List. First element specify the field type to be "normal" or "adiabatic." Second element is integer, specifying the number of adiabatic cycles.
    fieldType = params["fieldType"]
    # Specify the problem to be "dynamics" or "variational"
    problem = params["problem"]

    # Specify parameters for the GRAPE algorithm
    evo_time = params["evo_time"]
    n_ts = params["n_ts"]
    max_iter = params["max_iter"]
    max_wall_time = params["max_wall_time"]
    pulse_scaling = params["pulse_scaling"]
    initial_state = params["initial_state"]
    min_grad = 1e-20
    # Fidelity gradient calculation mode. "forLoop" is the default from QuTip, "serial_optimized" is a modified code I believe can be faster and easier to implement parallelization.
    fid_mode = params['fid_mode']
    # message level parameter in the L-BFGS-G optimization.
    msg_level = params['msg_level']

    # Initialize the Hamiltonian object
    Ham = initialize_Hamiltonian(params)

    # Define the drift Hamiltonian and the control Hamiltonian
    H_drift = Ham.get_driftHam()
    H_control = Ham.get_controls()

    if problem == 'dynamics':
        fid_params = dict(mode=mode, fid_mod=fid_transform_matrix(Ham, method=mode), fid_mode=fid_mode)

        # Use the modified expression for dynamics fidelity
        fid_type = "MODIFIED"

        # Initial and target unitary
        U_0 = Ham.id_full()
        U_targ = Ham.get_targUnitary(evo_time)

        # Target fidelity error
        fid_err_targ = 1e-10

        # Ground state energy (target fidelity error)
        gs = 0
    elif problem == 'variational':
        fid_params = dict(fid_mode=fid_mode)

        # Use a modified fidelity (cost function) expression defined for variational method
        fid_type = "VARIATIONAL"

        if mode == 'singleCav':
            U_0 = Ham.cavSys.state(0)
        elif mode == 'doubleCav':
            # Symmetric or antisymmetric initial state
            if initial_state == 'sym':
                U_0 = tensor(Ham.cav_1.cavSys.state(0), Ham.cav_2.cavSys.state(0))
            elif initial_state == 'antisym':
                U_0 = (tensor(Ham.cav_1.cavSys.state(0), Ham.cav_2.cavSys.state(1)) - tensor(Ham.cav_1.cavSys.state(1), Ham.cav_2.cavSys.state(0))).unit()

        # Hamiltonian expectation that is minimized
        U_targ = Ham.get_fermionHam()

        # Target energy (need to be lower than the actual ground state energy)
        fid_err_targ = -100

        # Ground state energy (target fidelity error)
        gs = groundEnergy(Ham.get_fermionHam())
    else:
        raise Exception('Need to specify which problem')

    # "random" - the default initial pulse shapes (normally random pulse shape)
    # "set" - input predefined values for the control pulses
    if params["input"] == "random":
        initial_amps = None
    elif params["input"] == "set":
        pulseData_dir = 'optimizedPulse'
        pulseFileName = problem + '_' + str(params["N_e_list"][0]) + 'e'
        optimResult = readPulse(data_dir = pulseData_dir, fileName = pulseFileName)
        optimizedPulse = optimResult.final_amps
        if n_ts != optimizedPulse.shape[0]:
            raise Exception('pulse dimension not match')
        initial_amps = np.zeros((n_ts, Ham.N_ctrl))
        initial_amps[:, :4] = optimizedPulse
        # initial_amps[:, 4:8] = optimizedPulse

    # Initial pulse shape. 'RND' for random initial pulse
    p_type = 'RND'
    np.random.seed(0)

    # 'normal' - directly invoking the field
    # 'adiabatic' - adiabatically invoking the field. fieldType[1] gives the number of steps to reach the target Hamiltonian magnitude.
    if fieldType[0] == 'normal':
        result = cpo.optimize_pulse_unitary(H_drift, H_control, U_0, U_targ, n_ts, evo_time,
                                            fid_err_targ=fid_err_targ, min_grad=min_grad,
                                            max_iter=max_iter, max_wall_time=max_wall_time,
                                            init_pulse_type=p_type, pulse_scaling=pulse_scaling,
                                            gen_stats=True, fid_type=fid_type, fid_params=fid_params,
                                            optim_params=dict(msg_level=msg_level),
                                            init_pulse_params=dict(set_initial_amps=initial_amps))
    elif fieldType[0] == 'adiabatic':
        n_divide = fieldType[1]
        target_U = Ham.site.U
        iter_left = max_iter
        time_left = max_wall_time
        for i in range(n_divide):
            Ham.site.U = target_U * (i + 1) / n_divide
            max_iter = iter_left / (n_divide - i)
            max_wall_time = time_left / (n_divide - i)
            if problem == 'dynamics':
                U_targ = Ham.get_targUnitary(evo_time)
            elif problem == 'variational':
                U_targ = Ham.get_fermionHam()
            result = cpo.optimize_pulse_unitary(H_drift, H_control, U_0, U_targ, n_ts, evo_time,
                                                fid_err_targ=fid_err_targ, min_grad=min_grad,
                                                max_iter=max_iter, max_wall_time=max_wall_time,
                                                init_pulse_type=p_type, pulse_scaling=pulse_scaling,
                                                gen_stats=True, fid_type=fid_type, fid_params=fid_params,
                                                optim_params=dict(msg_level=msg_level),
                                                init_pulse_params=dict(set_initial_amps=initial_amps))
            initial_amps = result.final_amps
            iter_left = iter_left - result.num_iter
            time_left = time_left - result.wall_time
            print('\n')
            print('Time left: ' + str(time_left))
            print('Iterations left: ' + str(iter_left))

    # Final optimized parameters
    U_f = result.evo_full_final
    infid_initial = result.initial_fid_err
    infid_final = result.fid_err

    # Creates saving directory
    if ifCreateDir:
        data_dir = createDataDir('dynamics')
        CONFIG_FILENAME = 'config.json'
        with open(data_dir + '/' + CONFIG_FILENAME, "w") as config_f:
            json.dump(params, config_f, indent=2)

    # Reports algorithm run details
    if ifReport:
        result.stats.report()
        print("Final evolution\n{}\n".format(result.evo_full_final))
        print("********* Summary *****************")
        print("Initial fidelity error {}".format(result.initial_fid_err))
        print("Final fidelity error {}".format(result.fid_err))
        print("Final gradient normal {}".format(result.grad_norm_final))
        print("Terminated due to {}".format(result.termination_reason))
        print("Number of iterations {}".format(result.num_iter))
        print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time))) 
        print('\n')

    # Save/plot the control pulse amplitudes
    if ifPlotPulse:
        plotCtrlPulses(result.time, result.initial_amps, plot_dir = data_dir,
                              titles = Ham.ctrls_label,
                          fileName = 'initial_pulses.png', xlabel = "time", ylabel = "Ctrl amp")
        plotCtrlPulses(result.time, result.final_amps, plot_dir=data_dir,
                       titles=Ham.ctrls_label,
                       fileName='final_pulses.png', xlabel="time", ylabel="Ctrl amp")

    # Save the algorithm run details
    if ifQsave:
        if data_dir == None:
            save_dir = '.'
        else:
            save_dir = data_dir
        if saveFormat == 'def':
            qsave(result, save_dir + '/' + fileName)
        elif saveFormat == 'mod':
            qsave([result.time, result.final_amps, result.evo_full_final, result.fid_err, params], save_dir + '/' + fileName)
        else:
            raise Exception('Not defined')


    output = dict(result = result, infidelity_initial = infid_initial, infidelity_final = infid_final,
                  wall_time = result.wall_time, grad_norm = result.grad_norm_final, term_reason = result.termination_reason,
                  n_iter = result.num_iter, groundEnergy = gs, U_0 = U_0, U_f = U_f, Ham = Ham, problem = problem, mode = mode)

    return output

if __name__ == "__main__":

    # Define the file that contains the parameters of the GRAPE algorithm and physical system
    CONFIG_FILENAME = 'config.json'
    with open('./' + CONFIG_FILENAME, 'r') as config_f:
        defaultParams = json.load(config_f)

    output = dynamics(defaultParams, ifReport=True, ifPlotPulse=False,
                      ifQsave=False, ifCreateDir=False, saveFormat='mod',
                      data_dir='.')

    if output["problem"] == 'dynamics':
        U_targ = output["Ham"].get_targUnitary(defaultParams['evo_time'])
        U_evol = output["U_f"]

        print(output["Ham"].site.siteMag)
        # Print the target and optimized unitary for comparison for single cavity case.
        # Only print the entries that are concerned.
        if output["mode"] == 'singleCav':
            output["Ham"].print_hop_unitary(U_targ, method = 'abs')
            print('\n')
            output["Ham"].print_hop_unitary(U_evol, method = 'abs')
        print("Fidelity: " + str(1 - output["infidelity_final"]))
    elif output["problem"] == 'variational':
        print("ground state energy:" + str(output["groundEnergy"]))
        print("Final energy:" + str(output["infidelity_final"]))

    # Verify if the Hilbert space truncation is large enough
    if defaultParams["verify"] == 1:
        # effective_states = expect(output["Ham"].id_cavity(size = 'hop'), output["U_f"])
        # print('Effective states: ' + str(effective_states))

        from fidelity import checkHilbertSapceTruncation
        N_cav_target = 20
        targ_param = defaultParams.copy()
        if output["mode"] == 'singleCav':
            targ_param['N_cav_list'] = [N_cav_target, 0]
        elif output["mode"] == 'doubleCav':
            targ_param['N_cav_list'] = [N_cav_target, N_cav_target]
        _fid_actual = checkHilbertSapceTruncation(targ_param=targ_param, optimResult=output["result"],
                                                  evo_time=targ_param["evo_time"])
        print("Actual: " + str(_fid_actual))
