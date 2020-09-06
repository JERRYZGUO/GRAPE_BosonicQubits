import numpy as np

def oneCavElectronMapping(prob, N_dim_spin, n_e, N_site = 4):
	if n_e == 0:
		return np.zeros((N_site, prob.shape[1]))
	elif n_e == 1:
		return prob.copy()
	elif n_e == 2:
		e_dens = [[]] * N_site
		e_dens[0] = prob[0] + prob[1] + prob[2]
		e_dens[1] = prob[0] + prob[3] + prob[4]
		e_dens[2] = prob[2] + prob[4] + prob[5]
		e_dens[3] = prob[1] + prob[3] + prob[5]
		return e_dens
	else:
		raise Exception('Not ready')

def electronDenMapping(Ham, prob):
    probDens = np.array(prob)
    if len(probDens) != Ham.fermSys.N_dim:
        raise Exception('Mismatch dimension')

    prob_up = np.zeros((Ham.fermSys.N_dim_up, probDens.shape[1]))
    prob_down = np.zeros((Ham.fermSys.N_dim_down,probDens.shape[1]))
    for i in range(Ham.fermSys.N_dim_up):
        prob_up[i] = np.sum(probDens[i * Ham.fermSys.N_dim_down : (i+1) * Ham.fermSys.N_dim_down, :], axis = 0)
    for i in range(Ham.fermSys.N_dim_down):
        prob_down[i] = np.sum(probDens[[j * Ham.fermSys.N_dim_down + i for j in list(range(Ham.fermSys.N_dim_up))], :], axis = 0)

    e_dens_up = oneCavElectronMapping(prob_up, Ham.fermSys.N_dim_up, Ham.fermSys.N_e[0])
    e_dens_down = oneCavElectronMapping(prob_down, Ham.fermSys.N_dim_down, Ham.fermSys.N_e[1])

    # return e_dens_up
    return [e_dens_up, e_dens_down]




def cavity_to_site_mapping(cav_index, N_site): # only for 2e
	return [(cav_index) % N_site, int((cav_index)/N_site)]






