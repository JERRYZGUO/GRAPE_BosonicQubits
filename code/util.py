#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:48:47 2020

@author: shirongbao
"""

import os, datetime
import configparser

from math import log10, floor
import math
import numpy as np

def loadConfig(config_file, dir=''):
	root_dir = os.path.abspath('')
	cp = configparser.ConfigParser(allow_no_value=True)
	cp.read(root_dir + '/' + config_file)
	return cp

def createDataDir(data_root, sub_dir=''):
	root_dir = os.path.abspath('')
	if len(sub_dir) == 0:
		sub_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	new_dir = root_dir + '/' + data_root + '/' + sub_dir
	os.makedirs(new_dir)
	return new_dir

def rows2list(rows, attr):
	my_list = []
	for row in rows:
		if attr in row: my_list.append(row[attr])
	return my_list

def round_to(x, n):
    if x == 0:
    	return x
    elif x<0:
    	return -round_to(-1 * x, n)
    else:
    	return round(x, -int(floor(log10(x))) + (n - 1))

def comb(n, r):
	f = math.factorial
	return int(f(n)/(f(r) * f(n-r)))


def readOptimInfo(filename):
	"""
	Read from scipy L-BFGS optimization outputs.
	"""
	with open(filename, 'r') as r:
		a = r.readlines()
		fid = []
		iter = []
		_fid = []
		_iter = []
		gs = None
		for s in a:
			ind = s.find('* * *')
			ind_gs = s.find('ground state energy:')
			if ind != -1:
				if len(_fid) == 0:
					pass
				else:
					fid.append(_fid)
					iter.append(_iter)
				_fid = []
				_iter = []
			elif ind_gs != -1:
				if gs != None:
					raise Exception('More than one ground state energy')
				else:
					gs = float(s.replace(' ', '')[ind_gs+18:-1])
			else:
				ind0 = s.find('iterate')
				ind1 = s.find('f=')
				ind2 = s.find('|')
				if ind0 == -1 or ind1 == -1 or ind2 == -1:
					pass
				else:
					_iter.append(int(s[ind0 + 7:ind1].replace(' ', '')))
					_fid.append(float(s[ind1 + 2:ind2].replace(' ', '').replace('D', 'e')))
	if len(_fid) == 0:
		pass
	else:
		fid.append(_fid)
		iter.append(_iter)
	if gs == None:
		gs = 0
	if len(fid) == 1:
		iter_mod = iter[0]
		fid_mod = [_f - gs for _f in fid[0]]
	else:
		fid_mod = []
		iter_mod = []
		iter_add = 0
		for i in range(len(fid)):
			for j in range(len(fid[i])):
				fid_mod.append(fid[i][j] - gs)
				iter_mod.append(iter[i][j] + iter_add)
			iter_add = iter_mod[-1]
	return iter_mod, fid_mod

# Implementation of parallelization from qiskit
# def _task_wrapper(param):
#     (task, value, task_args, task_kwargs) = param
#     return task(value, *task_args, **task_kwargs)

# def parallel_map_qiskit(  # pylint: disable=dangerous-default-value
#         task, values, task_args=tuple(), task_kwargs={}, num_processes=CPU_COUNT):
#     if len(values) == 1:
#         return [task(values[0], *task_args, **task_kwargs)]
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         param = map(lambda value: (task, value, task_args, task_kwargs), values)
#         future = executor.map(_task_wrapper, param)
#
#     results = list(future)
#     return results




