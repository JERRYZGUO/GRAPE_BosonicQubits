#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:49:52 2020

@author: shirongbao
"""

import matplotlib.pyplot as plt
import numpy as np
from util import round_to
    
def scatterPlot(x, y_list, labels,
                plot_dir, fileName = 'plot.png',
                title = 'title', xlabel = 'xlabel', ylabel = 'ylabel', xlimit = None, ylimit = None, ifAnnotate = True,
                plotType = 'norm', ifScatter = False, other = None, ifSemiLog = False, ifLegend = True): # other = [x, y, label]
    """
    Customized plotting with some options. Used to plot data.
    """
    if len(y_list) != len(labels):
        raise Exception('Mismatch in label and y_list size')
    else:
        n = len(labels)
    defaultType = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    if ifScatter:
        pointType = [t + '^' for t in defaultType] + [t + 'o' for t in defaultType]
    else:
        pointType = defaultType.copy() + [t + '--' for t in defaultType]
    

    
    fig = plt.figure()
    if plotType == 'large':
        fig.set_size_inches(18.5, 10.5)
    elif plotType == 'norm':
        pass
    ax = fig.add_subplot(111)

    if ifSemiLog:
        plot = plt.semilogy
        plt.grid(True, which="minor")
    else:
        plot = plt.plot

    for i in range(n):
        plot(x, y_list[i], pointType[i], label = labels[i])
        if ifAnnotate:
            for i,j in zip(x, y_list[i]):
                ax.annotate(str(round_to(j,2)),xy=(i,j))
    if other != None:
        pointType.reverse()
        for i in range(len(other)):
            plot(other[i][0], other[i][1], pointType[i], label=other[i][2])
        # if len(other) == 3:
        #     plot(other[0], other[1], 'k^', label = other[2])
        # elif len(other) == 2:
        #     plot(other[0][0], other[0][1], 'b', label=other[0][2])
        #     plot(other[1][0], other[1][1], 'g', label=other[1][2])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylimit != None:
        plt.ylim(ylimit)
    if xlimit != None:
        plt.xlim(xlimit)
    plt.title(title)
    if ifLegend:
        plt.legend()
    if plot_dir == None or fileName == None:
        plt.show()
    else:
        plt.savefig(plot_dir + '/' + fileName)
        plt.close()
    
def plotCtrlPulses(time, pulses, titles = ["a+a^dag", "1j * (a-a^dag)", "sigma_x", "sigma_y"], plot_dir = None,
                      fileName = 'pulses.png', xlabel = "time", ylabel = "Ctrl amp", index = 0):
    """
    Plotting the control pulses. Only plot 4 pulse amplitudes at most.
    """
    if pulses.shape[1] != len(titles):
        raise Exception('Mismatch in label and y_list size')
    else:
        n = len(titles)
    
    if n > 4:
        plotCtrlPulses(time, pulses=pulses[:, :4], titles=titles[:4], plot_dir=plot_dir,
                       fileName=fileName, xlabel=xlabel, ylabel=ylabel, index=index)
        plotCtrlPulses(time, pulses=pulses[:, 4:], titles=titles[4:], plot_dir=plot_dir,
                       fileName=fileName, xlabel=xlabel, ylabel=ylabel, index=index+1)
    else:
        fig = plt.figure()
        for i in range(n):
            ax = fig.add_subplot(n, 1, i+1)
            ax.set_title(titles[i])
            #ax1.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.step(time,
                 np.hstack((pulses[:, i], pulses[-1, i])),
                 where='post')
        plt.tight_layout()
        if plot_dir == None:
            plt.show()
        else:
            plt.savefig(plot_dir + '/' + str(index) + '_'+fileName)
            plt.close()
    
def plotHSD_3D(defaultParams):
    """
    Plot the Hilbert space dimensions in 3D. Not used.
    """
    params = defaultParams.copy()

    params['N_site'] = 6
    from systems import fermion
    f = fermion(params)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(params['N_site'] + 1)
    _y = np.arange(params['N_site'] + 1)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = []
    for i in range(len(x)):
        f.N_e = [x[i], y[i]]
        top.append(f.get_HilbertSpaceDim())

    top = np.array(top)

    bottom = np.zeros_like(top)
    width = depth = 0.5

    ax.bar3d(x, y, bottom, width, depth, top, color='c', shade=True)
    ax.set_xlabel('spin up electrons')
    ax.set_ylabel('spin down electrons')
    ax.set_zlabel('Hilbert space Dimension')
    ax.set_title('Hilbert space Dimension vs number of electrons')

    ax.view_init(45, 35)
    plt.savefig('barPlot')
    plt.close()
    
    
    
    
    
    