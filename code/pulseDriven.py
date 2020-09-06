from qutip import *
from scipy.interpolate import interp1d


class pulseObject():
    def __init__(self, properties):
        self.time = properties[0]
        self.final_amps = properties[1]
        self.evo_full_final = properties[2]
        self.fid_err = properties[3]
        self.params = properties[4]

def readPulse(data_dir = None, fileName = 'optimResult'):
	if data_dir == None:
		save_dir = '.'
	else:
		save_dir = data_dir
	result = pulseObject(qload(save_dir + '/' + fileName))
	return result

# Note here times has one more indice than amps
def reproduce_constant_func(times, amps):
	funcs = []
	for i in range(amps.shape[1]):
		funcs.append(lambda t, args: interp1d(times[:-1], amps[:, i], kind='zero',
	                             bounds_error=False, fill_value = "extrapolate")(t))
	return funcs
	                         
def pulse_Ham(Ham, result):
	times = result.time
	diff = (times[-1] - times[0])/(len(times)-1)

	H_control = Ham.get_controls()
	if len(H_control) != result.final_amps.shape[1]:
	    raise Exception("Mismatch dimensions")

	H = [Ham.get_driftHam()]
	# args = {}
	# if len(H_control) != 4:4
	# 	raise Exception("wrong function number")

	funcs = reproduce_constant_func(times, result.final_amps)
  
	for i in range(len(H_control)):
		# args["func" + str(i)] = interp1d((time + diff/2)[:-1], pulseResult.final_amps[:, i], kind='zero',
	 #                             bounds_error=False, fill_value = "extrapolate")
	    H.append([H_control[i], funcs[i]])
	return H


def evol_operators(Ham, result, initial = None):

    if initial == None:
        initial = Ham.id_full()

    H_drift = Ham.get_driftHam()
    H_ctrl = Ham.get_controls()

    times = result.time
    n_ts = len(times)-1

    diff = (times[-1] - times[0])/n_ts

    H_list = []
    for i in range(n_ts):
        H_list.append(H_drift + sum([result.final_amps[i, j] * H_ctrl[j] for j in range(Ham.N_ctrl)]))

    U_list = [(-1j * _H * diff).expm() for _H in H_list]

    ops = [initial]

    for i in range(n_ts):
        ops.append(U_list[i] * ops[i])

    return ops