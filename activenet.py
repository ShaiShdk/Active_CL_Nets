#%%

"""
Created in Nov. 2021
@author: Shahriar Shadkhoo -- CALTECH
"""
"""
    Notation:
        suffixes:
        1. "el" suffix stands for non-active "elastic" linkers.
        2. "mm" suffix stands for active symmetric "minus-minus" proteins.
        3. "pp" suffix stands for active symmetric "plus-plus" proteins.
        4. "mp" suffix stands for active asymmetric "minus-plus" proteins.

"""

import numpy as np
from numpy import pi
from scipy.spatial.distance import pdist , squareform
from scipy.sparse import csr_matrix, triu
from scipy.spatial import distance_matrix as dm
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import networkx as nx
from math import atan2
from activenet_funcs import pattern_density_convert , lattice
from copy import deepcopy
from time import time

seeding = 1
seednum = 20
if seeding:
    np.random.seed(seednum)

readimg , plotimg , saveimg = False , True , False
if not readimg:
    unitcell = 'square'

actv_on , elas_on , drag_on , nois_on = 1 , 1 , 1 , 0

cel = .0 ; cmm = .1 ; cpp = 0
ctot = cel + cmm + cpp
cel /= ctot ; cmm /= ctot ; cpp /= ctot
ccs = [cel,cmm,cpp]

emm , epp , emp = 1 , 1  , 1            # binding energy of different pairs of proteins
ael , amm , app = 0 , 1  , 1            # units of activity for mm , pp , mp motors
Vel , Vmm , Vpp = 0 , -1 , 1            # walking velocities for mm , pp , mp motors

A_type = np.asarray([ael,amm,app])
V_type = np.asarray([Vel,Vmm,Vpp])

kmm,kpp = 10 , 10 ; l_rest  = .1

selfint = -.5
fil_int = 1

T_tot , dt = 100 , 0.01
N_frame = np.min((20,int(T_tot/dt)))
tseries = range(round(T_tot/dt))
Nt      = len(tseries)

dens0   = 1
xi      = 1.5                             # this is len_avg * sqrt(dens)
XY_lens = [30,30]

rearr   = 1 ; Trearr  = 1
mft     = int( np.max((1,np.ceil(Trearr/dt))) )

Temp_t , Temp_a = 10 , 100              # effective temperatures for translational and angular velocities

mu      = 10
mpf     = 20                            # max num. motors per filament
diam    = 1
M0      = 1
I0      = 10
tau0    = 0                             # magnitude of unit of torque
Rc_0    = 1/2                           # cutoff distance in units of Avg_Lens

tplot = []
if plotimg:
    tplot = list(np.unique(np.around(np.linspace(0 , Nt , N_frame+1)).astype(int)))

t_rearr = []
if rearr:
    t_rearr = np.around(np.arange(mft , Nt-1 , mft)).astype(int)

#################################################################### NETWORK SETUP ###################################################################
if readimg:
    img     = plt.imread('IntensityImages/L_shape.tiff')
    [R_cnts , Ntot , img_converted , dens] = pattern_density_convert.Convert_Pattern_to_Points(img , dens0 , XY_lens)
else:
    dens = dens0
    if unitcell!='square':
        XY_lens[1] *= np.sqrt(3/2)
        XY_lens[1] -= (1+int(round(XY_lens[1]*np.sqrt(dens))))%2
    disorder= 1
    [R_cnts , Ntot] = lattice.Lattice_Points(XY_lens, dens, disorder=disorder, unit_cell=unitcell)

phi_range = [-np.pi/1,np.pi/1]
phi_avg   = 0
phi       = phi_avg + np.random.uniform(phi_range[0] , phi_range[1] , (Ntot,1))
px,py     = np.cos(phi) , np.sin(phi)

len_avg = xi*(dens)**(-1/2)
len_std = 0
lens    = len_avg

SumLens = 2*lens
Lrest   = l_rest * len_avg
MM      = M0
II      = I0

if len_std:
    lens *= np.abs(np.random.normal(len_avg , len_std , (Ntot,1)))
    MM   *= lens
    II   *= lens

Vx = np.zeros((Ntot,1))
Vy = np.zeros((Ntot,1))
av = np.zeros((Ntot,1))

xCtr  , yCtr  = R_cnts[:,0].reshape((Ntot,1)) , R_cnts[:,1].reshape((Ntot,1))
xMend , yMend = xCtr - (lens/2) * np.cos(phi) , yCtr - (lens/2) * np.sin(phi)
xPend , yPend = xCtr + (lens/2) * np.cos(phi) , yCtr + (lens/2) * np.sin(phi)

###################################################################### FUNCTIONS #####################################################################
def diff_operators(NN,R_cnts,SumLens,ccs,mpf,Rcutoff=Rc_0):

    ds     = pdist(R_cnts, 'euclidean')
    ds_rad = np.heaviside( - ds + SumLens*Rcutoff , 0 ).astype(int)

    if mpf==[]:
        ds_sq       = squareform(ds_rad) * np.triu(np.ones((NN,NN)),1)
    else:
        ds_vals     = ds * ds_rad
        ds_vals_mat = squareform(ds_vals) * np.triu(np.ones((NN,NN)),1)
        ds_vals_sym = ds_vals_mat + ds_vals_mat.T
        inds        = np.argsort(ds_vals_sym , axis=1)[:,::-1][:,:mpf]
        ds_sq       = np.zeros((NN,NN))
        for vv in range(NN):
            nz_inds = inds[vv][ds_vals_sym[vv][inds[vv]] != 0]
            ds_sq[vv][nz_inds] = 1
        ds_sq += ds_sq.T
        ds_sq[ds_sq!=0] = 1
        ds_sq = np.triu(ds_sq,1)

    v1  = np.where(ds_sq!=0)[0]
    v2  = np.where(ds_sq!=0)[1]
    N_bonds = len(v1)
    B12 = np.zeros((2*N_bonds , NN))
    B12[np.arange(N_bonds) , v2] = 1
    B12[np.arange(N_bonds,2*N_bonds) , v1] = 1

    B21 = np.vstack((B12[N_bonds:2*N_bonds] , B12[0:N_bonds]))
    B12 = csr_matrix(B12)
    B21 = csr_matrix(B21)
    Bas = B21 - B12

    edges     = list(zip(R_cnts[v1],R_cnts[v2]))
    MotorType = np.where(np.random.multinomial(1,ccs,2*N_bonds))[1]
    V_mot     = V_type[MotorType].reshape((2*N_bonds,1))
    A_mot     = A_type[MotorType].reshape((2*N_bonds,1))

    G = nx.Graph()
    ed_list = list(zip(v1,v2))
    G.add_edges_from(ed_list)

    return B12,B21,Bas,V_mot,A_mot,edges,G


[B12,B21,Bas,V_mot,A_mot,edges,G] = diff_operators(Ntot,R_cnts,SumLens,ccs,mpf)
N_bonds = int(len(A_mot)/2)
incid_el = Bas[0:N_bonds]

################################################################### PLOT SETTINGS ####################################################################
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]
    })

fsize = (10,10)
marg  = 1.2
color_rods , color_Pend , color_Mend = [.8,.8,.8] , [.6,.0,.4] , [.0,.6,.4]
color_rand = .3 + np.random.uniform(0,.7,(Ntot,3))

cmap = plt.get_cmap('RdBu')
color_ang = cmap((np.pi + np.arctan2(py,px))/(2*np.pi))

quiv_set = {'color':color_rods , 'scale_units':'xy' , 'scale':1 , 'headwidth':1, 'pivot':'tail'}
s0    = fsize[0]/2
txtsize = 7*fsize[0]/5

def box_corners(xcnt0,ycnt0,gap):
    x_corners = [np.min(xcnt0-gap) , np.max(xcnt0+gap) , np.max(xcnt0+gap) , np.min(xcnt0-gap) , np.min(xcnt0-gap)]
    y_corners = [np.min(ycnt0-gap) , np.min(ycnt0-gap) , np.max(ycnt0+gap) , np.max(ycnt0+gap) , np.min(ycnt0-gap)]
    return x_corners , y_corners

[x_corners , y_corners] = box_corners(xCtr,yCtr,gap=1)
xmin , xmax = min(x_corners) , max(x_corners)
ymin , ymax = min(y_corners) , max(y_corners)
XLIM , YLIM = np.asarray([marg*xmin , marg*xmax]) ,  np.asarray([marg*ymin , marg*ymax])
xrange,yrange= np.asarray([xmin , xmax]) ,  np.asarray([ymin , ymax])

################################################################### PLOT FUNCTIONS ###################################################################

def plot_rods(tt=0,plot_radius=True):
    R_Mend = np.concatenate((xMend,yMend),axis=1)
    R_Pend = np.concatenate((xPend,yPend),axis=1)
    rods   = list(zip(R_Mend,R_Pend))
    fig    = plt.figure(facecolor='w',figsize=fsize,dpi=100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('k')
    lc= mc.LineCollection(rods,color=[.7,.7,.7],alpha=.7,linewidths=2)
    plt.gca().add_collection(lc)
    plt.scatter(xMend,yMend,color=color_Mend,s=s0,zorder=2)
    plt.scatter(xPend,yPend,color=color_Pend,s=s0,zorder=2)
    plt.text(.9*xmin,1.07*ymax,'time = '+str(np.around(tt*dt,str(dt)[::-1].find('.'))),c='w',alpha=1,fontsize=txtsize)
    plt.text(.9*xmin,1.11*ymin,'seed = '+str(seednum),c='w',alpha=1-0*tt/Nt,fontsize=txtsize)
    plt.text(.65*xmax,1.11*ymin,'mpf = '+str(mpf),c='w',alpha=1,fontsize=txtsize)
    plt.text(.92*xmax,ymax+1.05,'$\langle \ell \\rangle$',c='w',alpha=1,fontsize=txtsize)
    plt.plot(x_corners, y_corners, 'gray' ,linewidth = 1)
    plt.plot([x_corners[2]-len_avg,x_corners[2]] , [y_corners[2],y_corners[2]],'-',color=[.6,0,.2],lw=5,alpha=1,zorder=3)
    if plot_radius:
        plt.gca().add_patch(plt.Circle((x_corners[2], y_corners[2]), SumLens*Rcutoff, color=[.0,.4,.6],alpha=.7,clip_on=False))
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    if saveimg:
        fname = 'Simulation_Movie/actnet_t%07d.tif'%tt
        plt.savefig(fname)
    plt.show()

def plot_PtsEds(rs,edges,tt=0,plot_eds=True,plot_radius=True,color_scatter=color_rand):
    xs, ys = rs[:,0] , rs[:,1]
    fig    = plt.figure(facecolor='w',figsize=fsize,dpi=100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('k')
    if plot_eds:
        lc= mc.LineCollection(edges,color=[.3,.4,.5],alpha=.7,linewidths=.9)
        plt.gca().add_collection(lc)
    nodes = plt.scatter(xs,ys,color=color_scatter,s=s0,zorder=2)
    plt.text(.9*xmin,1.07*ymax,'time = '+str(np.around(tt*dt,str(dt)[::-1].find('.'))),c='w',alpha=1,fontsize=txtsize)
    plt.text(.9*xmin,1.11*ymin,'seed = '+str(seednum),c='w',alpha=1-0*tt/Nt,fontsize=txtsize)
    plt.text(.65*xmax,1.11*ymin,'mpf = '+str(mpf),c='w',alpha=1,fontsize=txtsize)
    plt.text(.92*xmax,ymax+1.05,'$\langle \ell \\rangle$',c='w',alpha=1,fontsize=txtsize)
    plt.plot(x_corners, y_corners, 'gray' ,linewidth = 1)
    plt.plot([x_corners[2]-len_avg,x_corners[2]] , [y_corners[2],y_corners[2]],'-',color=[.6,0,.2],lw=5,alpha=1,zorder=3)
    if plot_radius:
        plt.gca().add_patch(plt.Circle((x_corners[2], y_corners[2]), SumLens*Rcutoff, color=[0,.4,.6],alpha=.7, clip_on=False))
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    if saveimg:
        fname = 'Simulation_Movie/actnet_t%07d.tif'%tt
        plt.savefig(fname)
    plt.show()

def plot_timefcts(Xt,timeseries,VarName=[]):
    ts  = dt * timeseries
    fig = plt.figure(facecolor='w',dpi=100,figsize=fsize)
    plt.plot(ts, Xt)
    plt.xlabel('\it{time}',fontsize=20)
    plt.ylabel(VarName,fontsize=20)
    if saveimg:
        fname = f'R2_C_{dens}.png'
        plt.savefig(fname)
    plt.show()

################################################################### INITIALIZATION ###################################################################
DxC , DyC = incid_el.dot(xCtr) , incid_el.dot(yCtr) ; DrC = np.sqrt(DxC**2 + DyC**2)
conv_fact = Rc_0/np.std(DrC)
Rcutoff   = Rc_0

plot_PtsEds(R_cnts,edges,0,plot_eds=True,plot_radius=False,color_scatter=color_ang)
plot_rods(0,plot_radius=False)

#%%

ti = time()

for tt in tseries:

    if tt in t_rearr:
        [B12,B21,Bas,V_mot,A_mot,edges,G] = diff_operators(Ntot,R_cnts,SumLens,ccs,mpf,Rcutoff)
        N_bonds = int(len(A_mot)/2)
        incid_el = Bas[0:N_bonds]

    xMend , yMend = xCtr - (lens/2) * np.cos(phi) , yCtr - (lens/2) * np.sin(phi)
    xPend , yPend = xCtr + (lens/2) * np.cos(phi) , yCtr + (lens/2) * np.sin(phi)

    DxM , DyM = incid_el.dot(xMend) , incid_el.dot(yMend)
    DxP , DyP = incid_el.dot(xPend) , incid_el.dot(yPend)
    DrM , DrP = np.sqrt(DxM**2 + DyM**2) , np.sqrt(DxP**2 + DyP**2)

    DxC , DyC = incid_el.dot(xCtr) , incid_el.dot(yCtr) ; DrC = np.sqrt(DxC**2 + DyC**2)

    lev_r = len_avg/2
    Vr    = np.hstack((Vx - lev_r * av * np.sin(phi) , Vy + lev_r * av * np.cos(phi)))
    Pr    = np.hstack((px,py))

    ############################################################ Force Calculation ############################################################

    Fax = Fay = 0
    if actv_on:
        Fact = fil_int * B21.T.dot(( A_mot * (V_mot - np.sum((B12.dot(Pr))*(Bas.dot(Vr)) , axis=1).reshape(B12.shape[0],1)) ) * B12.dot(Pr))
        Fact+= selfint * B21.T.dot(( A_mot * (V_mot + np.sum((B21.dot(Pr))*(B21.dot(Vr)) , axis=1).reshape(B21.shape[0],1)) ) * B21.dot(Pr))
        Fax  = Fact[:,0].reshape(Ntot,1)
        Fay  = Fact[:,1].reshape(Ntot,1)

    Fpx = Fpy = 0
    if elas_on:
        FpxA = -incid_el.T.dot(kmm*np.exp(-(DrM-2*Lrest)**2/(2*Lrest**2))*(DxM/DrM) + kpp*np.exp(-(DrP-2*Lrest)**2/(2*Lrest**2))*(DxP/DrP))
        FpyA = -incid_el.T.dot(kmm*np.exp(-(DrM-2*Lrest)**2/(2*Lrest**2))*(DyM/DrM) + kpp*np.exp(-(DrP-2*Lrest)**2/(2*Lrest**2))*(DyP/DrP))
        FpxR = +incid_el.T.dot(10*kmm*np.exp(-(DrM-1*Lrest)**2/(1*Lrest**2))*(DxM/DrM) + 10*kpp*np.exp(-(DrP-1*Lrest)**2/(1*Lrest**2))*(DxP/DrP))
        FpyR = +incid_el.T.dot(10*kmm*np.exp(-(DrM-1*Lrest)**2/(1*Lrest**2))*(DyM/DrM) + 10*kpp*np.exp(-(DrP-1*Lrest)**2/(1*Lrest**2))*(DyP/DrP))
        Fpx  = FpxA + FpxR
        Fpy  = FpyA + FpyR

    Fdx = Fdy = 0
    if drag_on:
        FparX = - mu * np.pi*(diam**2) * (+px*Vx + py*Vy) * px
        FparY = - mu * np.pi*(diam**2) * (+px*Vx + py*Vy) * py
        FprpX = - mu * diam**1 * len_avg * (+px*Vy - py*Vx) * (-py)
        FprpY = - mu * diam**1 * len_avg * (+px*Vy - py*Vx) * (+px)
        Fdx   = FparX + FprpX
        Fdy   = FparY + FprpY

    ax    = (Fpx + Fax + Fdx)/MM + nois_on * Temp_t**.5 * np.random.normal(0 , 1 , (Ntot,1))
    ay    = (Fpy + Fay + Fdy)/MM + nois_on * Temp_t**.5 * np.random.normal(0 , 1 , (Ntot,1))
    xCtr += Vx*dt + (ax * dt**2)/2 ;     yCtr += Vy*dt + (ay * dt**2)/2
    Vx   += ax*dt ;                      Vy   += ay*dt

    ############################################################ Torque Calculation ############################################################
    if tau0:
        Tau    = - tau0 * (np.sin(incid_el.T.dot(incid_el).dot(phi))) - (1)*av * mu * len_avg**2/2
        ang_ax = tau0 * Tau/II + nois_on * (Temp_a**.5/II) * np.random.normal(0 , 1 , (Ntot,1))
        phi   += av * dt + (ang_ax * dt**2)/2
        av    += ( ang_ax ) * dt
        px,py  = np.cos(phi) , np.sin(phi)

    if tt in tplot:
        color_ang = cmap((np.pi + np.arctan2(py,px))/(2*np.pi))
        plot_PtsEds(R_cnts,edges,tt,plot_eds=True,plot_radius=False,color_scatter=color_ang)
        plot_rods(tt,plot_radius=False)

color_ang = cmap((np.pi + np.arctan2(py,px))/(2*np.pi))
plot_PtsEds(R_cnts,edges,tt+1,plot_eds=True,plot_radius=False,color_scatter=color_ang)
plot_rods(tt+1,plot_radius=False)

