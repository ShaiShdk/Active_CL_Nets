"""
Created in Nov. 2021
@author: Shahriar Shadkhoo -- CALTECH
"""

import numpy as np , scipy as sp , random
from scipy.spatial import Voronoi , voronoi_plot_2d
from scipy import sparse
import matplotlib.pyplot as plt
from copy import deepcopy
from math import atan2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from itertools import product

def Lattice_Points(XY_lens , dens  , disorder , unit_cell='square' , spdim=2):

    nx = int(round(XY_lens[0]*np.sqrt(dens)))
    ny = int(round(XY_lens[1]*np.sqrt(dens)))

    if unit_cell == 'square':
        R_lst  = [[i , j] for i in range(nx) for j in range(ny)]
    else:
        R_lst_0  = [[(1+(-1)**(j+1))/4 , j*np.sqrt(3)/2] for j in range(1,ny,2)]
        R_lst_1  = [[i+(1+(-1)**(j+1))/4 , j*np.sqrt(3)/2] for i in range(1,nx) for j in range(ny)]
        R_lst    = R_lst_0 + R_lst_1

    R_cnts  = np.asarray(R_lst).astype(float)
    R_cnts -= np.mean(R_cnts,axis=0)
    R_cnts *= np.asarray([np.sqrt(1/dens),np.sqrt(1/dens)])

    Ntot    = len(R_cnts)
    R_cnts += np.random.uniform(low=-disorder , high=+disorder , size=(Ntot,spdim))

    return R_cnts , Ntot
