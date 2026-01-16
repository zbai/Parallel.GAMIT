
# Author:   Chong-You (Kevin) Wang, 2025-09, OSU
""" 
    Code to compute green function and build design matrix for dislocation. This code is the python version of
    ConstructDesignMatrix.m provided by Zhou. et al. (2019b).

usage: 
    design_matrix = build_design_matrix(source_data,field_data,fault_para,core_file,mantle_file,ISO)

Zhou, J., Pan, E., & Bevis, M. (2019). A point dislocation in a layered, transversely isotropic and self-gravitating
Earth—Part II: accurate Green's functions. Geophysical Journal International, 219(3), 1717-1728.
"""
import numpy as np
import os
from numba import njit

from ..elasticity import DLN_func as DLN

CONST_R_km = 6371 # in km
CONST_R = 6371e3
CONST_gR = 9.82


def build_design_matrix(source_data,field_data,fault_para,core_file=None,mantle_file=None,ISO=1):
    """ function to build design matrix
    
    Parameters:
        source_data : 2D np.ndarray in the size of (num_source, 3), source_data = [co_lat,lon,depth]. coordinate is in degree
        field_data  : 2D np.ndarray in the size of (num_field, 2), field_data = [co_lat,lon]. coordinate is in degree
        fault_para  : 2D np.ndarray in the size of (num_fualt, 3), fault_para = [strike,dip,area]. angles are in degree
        core_file   : string, file  of core model
        mantle_file : string, file  of mantle model
        ISO         : index to show whether mantile is isotropic or not. ISO==1: isotropic
    
    return:
        design_matrix: 2D np.ndarray in the size of (3 * num_field, 2 * num_source), design matrix based on green function
    """
    if not mantle_file:
        from importlib.resources import files
        data_path = files('geode.elasticity.data').joinpath(
            'EarthMantle56.txt'
        )
        mantle_file = str(data_path)

    if not core_file:
        from importlib.resources import files
        data_path = files('geode.elasticity.data').joinpath(
            'EarthCore.txt'
        )
        core_file = str(data_path)

    source_data[:,:2] = source_data[:,:2] * np.pi / 180
    field_data = field_data * np.pi / 180
    fault_para[:,:2] = fault_para[:,:2] * np.pi / 180
    half_pi = np.pi / 2
    
    nodal_point = np.zeros(2789,)
    nodal_point[:99] = np.arange(0.001,0.1,0.001)
    nodal_point[99:99+990] = np.arange(0.1,10,0.01)
    nodal_point[99+990:] = np.arange(10,180,0.1)
    nodal_point = nodal_point * np.pi / 180
    CM_models = DLN.Read_models(core_file=core_file,mantle_file=mantle_file)
    DLN_model = DLN.DLN(CM_models)
    design_matrix = np.zeros((3 * field_data.shape[0],2 * source_data.shape[0]))
    for i in range(source_data.shape[0]):
        strike = fault_para[i,0]
        dip = fault_para[i,1]
        area = fault_para[i,2]
        lat_s = source_data[i,0]
        lon_s = source_data[i,1]
        depth_s = source_data[i,2].round()
        Ur, Vs, Vt, tVs, tVt = get_green_func(DLN_model,depth_s,nodal_point,ISO)
        # spherical trigonometry
        cos_psi = np.cos(lat_s) * np.cos(field_data[:,0]) + np.sin(lat_s) * np.sin(field_data[:,0]) * np.cos(lon_s - field_data[:,1])
        sin_psi = np.sqrt(1 - cos_psi**2)
        psi = np.arctan2(sin_psi,cos_psi)
        sin_A = np.sin(field_data[:,0]) * np.sin(field_data[:,1] - lon_s) / sin_psi
        cos_A = (np.cos(field_data[:,0]) - np.cos(lat_s) * cos_psi) / np.sin(lat_s) / sin_psi 
        angle_B = strike - np.arctan2(sin_A,cos_A)
        sin_alpha = np.sin(lat_s) * np.sin(field_data[:,1] - lon_s) / sin_psi
        cos_alpha = -(np.cos(lat_s) - np.cos(field_data[:,0]) * cos_psi) / np.sin(field_data[:,0]) / sin_psi
        Ur_field = linear_interp(nodal_point,Ur,psi)
        Vs_field = linear_interp(nodal_point,Vs,psi)
        Vt_field = linear_interp(nodal_point,Vt,psi)
        tVs_field = linear_interp(nodal_point,tVs,psi)
        tVt_field = linear_interp(nodal_point,tVt,psi)
        # Ur_field = [Ur12,Ur32,Ur22,Ur33]
        # Vs_field = [Vs12, Vs32, Vs22, Vs33]
        # Vt_field = [Vt12, Vt32]
        # D11
        design_matrix[::3,2*i] = area * (-Ur_field[:,0] * np.sin(2 * angle_B) * np.sin(dip) + Ur_field[:,1] * np.sin(angle_B + half_pi) * np.cos(dip))
        #D12
        design_matrix[::3,2*i+1] = area * (0.5 * (Ur_field[:,3] - Ur_field[:,2] - Ur_field[:,0] * np.sin(2 * angle_B - half_pi)) * np.sin(2 * dip) + Ur_field[:,1] * np.sin(angle_B) * np.cos(2 * dip)) 
        #D21
        design_matrix[1::3,2*i] = area * (cos_alpha * (\
                                        - Vs_field[:,0] * np.sin(2 * angle_B) * np.sin(dip) \
                                        + Vs_field[:,1] * np.sin(angle_B + half_pi) * np.cos(dip)\
                                        - tVs_field[:,0] * np.sin(2 * angle_B) * np.sin(dip) \
                                        + tVs_field[:,1] * np.sin(angle_B + half_pi) * np.cos(dip)) + \
                                        sin_alpha * (-Vt_field[:,0] * np.cos(2 * angle_B) * np.sin(dip) \
                                        + Vt_field[:,1] * np.cos(angle_B + half_pi) * np.cos(dip) \
                                        - tVt_field[:,0] * np.cos(2 * angle_B) * np.sin(dip) \
                                        + tVt_field[:,1] * np.cos(angle_B + half_pi) * np.cos(dip)))
        #D22
        design_matrix[1::3,2*i+1] = area * (cos_alpha * (0.5 * (Vs_field[:,3] - Vs_field[:,2] - Vs_field[:,0] * np.sin(2 * angle_B -      half_pi)) * np.sin(2 * dip) \
                                        + Vs_field[:,1] * np.sin(angle_B) * np.cos(2 * dip) \
                                        - 0.5 * tVs_field[:,0] * np.sin(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + tVs_field[:,1] * np.sin(angle_B) * np.cos(2 * dip)) \
                                        + sin_alpha * (\
                                        - 0.5 * Vt_field[:,0] * np.cos(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + Vt_field[:,1] * np.cos(angle_B) * np.cos(2 * dip) \
                                        - 0.5 * tVt_field[:,0] * np.cos(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + tVt_field[:,1] * np.cos(angle_B) * np.cos(2 * dip)))
        
        # D31
        design_matrix[2::3,2*i] = area * (sin_alpha * (-Vs_field[:,0] * np.sin(2 * angle_B) * np.sin(dip) + \
                                        Vs_field[:,1] * np.sin(angle_B + half_pi) * np.cos(dip)\
                                        - tVs_field[:,0] * np.sin(2 * angle_B) * np.sin(dip) \
                                        + tVs_field[:,1] * np.sin(angle_B + half_pi) * np.cos(dip)) \
                                        - cos_alpha * (-Vt_field[:,0] * np.cos(2 * angle_B) * np.sin(dip) \
                                        + Vt_field[:,1] * np.cos(angle_B + half_pi) * np.cos(dip) \
                                        - tVt_field[:,0] * np.cos(2 * angle_B) * np.sin(dip) \
                                        + tVt_field[:,1] * np.cos(angle_B + half_pi) * np.cos(dip)))
        
        # D32
        design_matrix[2::3,2*i+1] =  area * (sin_alpha * (0.5 * (Vs_field[:,3] - Vs_field[:,2] - Vs_field[:,0] * np.sin(2 * angle_B -      half_pi)) * np.sin(2 * dip) \
                                        + Vs_field[:,1] * np.sin(angle_B) * np.cos(2 * dip) \
                                        - 0.5 * tVs_field[:,0] * np.sin(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + tVs_field[:,1] * np.sin(angle_B) * np.cos(2 * dip)) \
                                        - cos_alpha * (-0.5 * Vt_field[:,0] * np.cos(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + Vt_field[:,1] * np.cos(angle_B) * np.cos(2 * dip) \
                                        - 0.5 * tVt_field[:,0] * np.cos(2 * angle_B - half_pi) * np.sin(2 * dip) \
                                        + tVt_field[:,1] * np.cos(angle_B) * np.cos(2 * dip)))
    return design_matrix
    
@njit
def linear_interp(x_p,f_p,interp_x):
    n_dim = f_p.shape[1]
    interp_f = np.zeros((interp_x.shape[0],n_dim))
    for i in range(n_dim):
        interp_f[:,i] = np.interp(interp_x,x_p,f_p[:,i])
    return interp_f

# %%

def get_green_func(DLN_obj,depth_s,nodal_point,ISO=1):
    love_degree_e = int(np.round(CONST_R_km*10 / depth_s))
    dir_name = os.path.join(os.getcwd(),f"GF_{os.path.splitext(DLN_obj.core_file)[0]}_{os.path.splitext(DLN_obj.mantle_file)[0]}")
    os.makedirs(dir_name,exist_ok=True)
    GF_name = f"S_{int(depth_s)}_F{0}_ISO_{ISO}_highD_{love_degree_e}.npz"
    save_name = os.path.join(dir_name,GF_name)
    if not os.path.isfile(save_name):
        if ISO == 1:
            rx = (CONST_R_km - depth_s) / CONST_R_km 
            if abs(depth_s) > 1:
                degree_b = ((-100 * np.log(10) / np.log(rx)).round()).astype(np.int64)
                degree_e = ((-150 * np.log(10) / np.log(rx)).round()).astype(np.int64)
                degree_i = (((degree_e - degree_b) / 10).round()).astype(np.int64)
            else:
                degree_b = 1000000
                degree_e = 2000000
                degree_i = 100000
            love = DLN_obj.loves_dislocation(depth_s, 0, 0, np.round(CONST_R_km*10 / depth_s), 1)
            love_H = DLN_obj.loves_dislocation(depth_s, 0, degree_b, degree_e, degree_i)
            love_para = fitDLN(rx,love_H)
        else:
            love = DLN_obj.loves_dislocation(depth_s, 0, 0, np.round(CONST_R_km*10 / depth_s), 1)
            love_para = None
        Ur, Vs, Vt, tVs, tVt = cal_green_func(depth_s,love,love_para,nodal_point)
        np.savez_compressed(save_name,Ur=Ur,Vs=Vs,Vt=Vt,tVs=tVs,tVt=tVt)
    else:
        tmp = np.load(save_name)
        Ur, Vs, Vt, tVs, tVt = tmp['Ur'], tmp['Vs'], tmp['Vt'], tmp['tVs'], tmp['tVt']
    return Ur, Vs, Vt, tVs, tVt

def fitDLN(rx,love):
    ind_normalized = [0,2,3]
    love_para = {'Hs':None,'Ls':None,'Ks':None,'Lt':None}
    scaling = rx**(love['degree_list'])
    love['Hs'][:,ind_normalized] = love['Hs'][:,ind_normalized] * love['degree_list'][:,None]
    love['Ls'][:,ind_normalized] = love['Ls'][:,ind_normalized] * love['degree_list'][:,None]
    love['Ks'][:,ind_normalized] = love['Ks'][:,ind_normalized] * love['degree_list'][:,None]
    love['Lt'][:,0] =  love['Lt'][:,0] * love['degree_list'][:]**2
    love['Lt'][:,1] =  love['Lt'][:,1] * love['degree_list'][:]
    love_para['Hs'] = np.polyfit(x=love['degree_list'][:],y=love['Hs'][:]/scaling[:,None],deg=2).T # matlab polyfit is slightly different from numpy. so the results are slightly different
    love_para['Ls'] = np.polyfit(x=love['degree_list'][:],y=love['Ls'][:]/scaling[:,None],deg=2).T
    love_para['Ks'] = np.polyfit(x=love['degree_list'][:],y=love['Ks'][:]/scaling[:,None],deg=2).T
    love_para['Lt'] = np.polyfit(x=love['degree_list'][:],y=love['Lt'][:]/scaling[:,None],deg=2).T
    return love_para

def cal_green_func(depth_s,love,love_para,nodal_point):
    rs = CONST_R - depth_s * 1000
    rR = rs / CONST_R  # in matlab rR = rx = rs/rf = rf/CONST_R. I just remove unnecessary variables.
    n_max = love['degree_list'][-1]
    rRn = rR**love['degree_list'][:]
    Ur = np.zeros((nodal_point.shape[0],4))
    Vs = np.zeros((nodal_point.shape[0],4))
    Vt = np.zeros((nodal_point.shape[0],2))
    tVs = np.zeros((nodal_point.shape[0],2))
    tVt = np.zeros((nodal_point.shape[0],2))
    # degree == 0
    Ur2_0 = np.sqrt(1/4/np.pi) / rs**2 * love['H22'] * np.sqrt(1/4/np.pi)
    Ur3_0 = np.sqrt(1/4/np.pi) / rs**2 * love['H33'] * np.sqrt(1/4/np.pi) # c33 == c22
    
    love['Ls'][:] = love['Ls'][:] / love['degree_list'][:,np.newaxis]
    love['Lt'][:] = love['Lt'][:] / love['degree_list'][:,np.newaxis]
    #c33 = c22 
    if love_para is not None:
        for i in range(nodal_point.shape[0]):
            pnm = asso_legendre(nodal_point[i],n_max)
            sums = infinite_sum(nodal_point[i],rR)
            
            c22 = np.sqrt((2 * love['degree_list'][:] + 1) / np.pi / 4) / rs**2 * pnm[:,6]
            c32 = c22 / np.sqrt((love['degree_list'][:] + 1) * (love['degree_list'][:]))
            c12 = c32 * np.sqrt((love['degree_list'][:] + 2) * (love['degree_list'][:] -1))
        # degree == 1~n_max
            tmp =  love['Hs'][:,0] - (love['degree_list'][:] * love_para['Hs'][0,0] + love_para['Hs'][0,1]) * rRn
            Ur[i,0] = (c12 * tmp * pnm[:,2]).sum()
        
            tmp =  love['Hs'][:,1] - (love['degree_list'][:]**2 * love_para['Hs'][1,0] + love['degree_list'][:] * love_para['Hs'][1,1] + love_para['Hs'][1,2]) * rRn
            Ur[i,1] = (c32 * tmp * pnm[:,1]).sum() 
        
            tmp =  love['Hs'][:,2] - (love['degree_list'][:] * love_para['Hs'][2,0] + love_para['Hs'][2,1]) * rRn
            Ur[i,2] = (c22 * tmp * pnm[:,0]).sum() + Ur2_0
        
            tmp =  love['Hs'][:,3] - (love['degree_list'][:] * love_para['Hs'][3,0] + love_para['Hs'][3,1]) * rRn
            Ur[i,3] = (c22 * tmp * pnm[:,0]).sum() + Ur3_0# c33 == c22
        
        
            tmp =  love['Ls'][:,0] - (love_para['Ls'][0,0] + love_para['Ls'][0,1] / love['degree_list'][:]) * rRn
            Vs[i,0] = (c12 * tmp * pnm[:,5]).sum() # c33 == c22
            Vt[i,0] = (c12 * tmp * pnm[:,2]).sum() / np.sin(nodal_point[i])
        
            tmp =  love['Ls'][:,1] - (love['degree_list'][:] * love_para['Ls'][1,0] + love_para['Ls'][1,1]) * rRn
            Vs[i,1] = (c32 * tmp * pnm[:,4]).sum() # c33 == c22
            Vt[i,1] = (c32 * tmp * pnm[:,1]).sum() / np.sin(nodal_point[i]) 
        
            tmp =  love['Ls'][:,2] - (love_para['Ls'][2,0] + love_para['Ls'][2,1] / love['degree_list'][:]) * rRn
            Vs[i,2] = (c22 * tmp * pnm[:,3]).sum() # c33 == c22
        
            tmp =  love['Ls'][:,3] - (love_para['Ls'][3,0] + love_para['Ls'][3,1] / love['degree_list'][:]) * rRn
            Vs[i,3] = (c22 * tmp * pnm[:,3]).sum() # c33 == c22
        
        # toroidal
            tmp = love['Lt'][:,0] - (love_para['Lt'][0,0] / love['degree_list'][:]) * rRn
            tVs[i,0] = (c12 * tmp * pnm[:,2]).sum() / np.sin(nodal_point[i])
            tVt[i,0] = (c12 * tmp * pnm[:,5]).sum()
        
            tmp = love['Lt'][:,1] - (love_para['Lt'][1,0]) * rRn
            tVs[i,1] = (c32 * tmp * pnm[:,1]).sum() / np.sin(nodal_point[i])
            tVt[i,1] = (c32 * tmp * pnm[:,4]).sum()
        
        
        # add infinite sums
            cc = 1/4/np.pi/rs**2
            Ur[i,0] = -2 * (Ur[i,0] + cc * (love_para['Hs'][0,0] * (2 * sums[8] - sums[10]) + love_para['Hs'][0,1] * (sums[9] + sums[10])))
            Ur[i,1] = -2 * (Ur[i,1] + cc * (love_para['Hs'][1,0] * (2 * sums[4] - sums[5] + sums[7]) + love_para['Hs'][1,1] * (2 * sums[5] - sums[7]) + love_para['Hs'][1,2] * (sums[6] + sums[7])))
            Ur[i,2] = Ur[i,2] + cc * (love_para['Hs'][2,0] * (2 * sums[0] + sums[1]) + love_para['Hs'][2,1] * (2 * sums[1] + sums[2]))
            Ur[i,3] = Ur[i,3] + cc * (love_para['Hs'][3,0] * (2 * sums[0] + sums[1]) + love_para['Hs'][3,1] * (2 * sums[1] + sums[2]))
        
            Vs[i,0] = -2 * (Vs[i,0] + cc * (love_para['Ls'][0,0] * (sums[19] + sums[20]) + love_para['Ls'][0,1] * (sums[21] + sums[22])))
            Vs[i,1] = -2 * (Vs[i,1] + cc * (love_para['Ls'][1,0] * (2 * sums[16] - sums[18]) + love_para['Ls'][1,1] * (sums[17] + sums[18])))
            Vs[i,2] = Vs[i,2] + cc * (love_para['Ls'][2,0] * (2 * sums[13] + sums[14]) + love_para['Ls'][2,1] * (2 * sums[14] + sums[15]))
            Vs[i,3] = Vs[i,3] + cc * (love_para['Ls'][3,0] * (2 * sums[13] + sums[14]) + love_para['Ls'][3,1] * (2 * sums[14] + sums[15]))
        
            Vt[i,0] = -4 * (Vt[i,0] + cc * (love_para['Ls'][0,0] * (sums[9] + sums[10]) + love_para['Ls'][0,1] * (sums[11] + sums[12])) / np.sin(nodal_point[i]))
            Vt[i,1] = -2 * (Vt[i,1] + cc * (love_para['Ls'][1,0] * (2 * sums[5] - sums[7]) + love_para['Ls'][1,1] * (sums[6] + sums[7])) / np.sin(nodal_point[i]))
        
            tVs[i,0] = -4 * (tVs[i,0] + cc * (love_para['Lt'][0,0] * (sums[11] + sums[12])) / np.sin(nodal_point[i]))
            tVs[i,1] = 2 * (tVs[i,1] + cc * (love_para['Lt'][1,0] * (sums[6] + sums[7])) / np.sin(nodal_point[i]))
            tVt[i,0] = -2 * (tVt[i,0] + cc * (love_para['Lt'][0,0] * (sums[21] + sums[22])))
            tVt[i,1] = 2 * (tVt[i,1] + cc * (love_para['Lt'][1,0] * (sums[17] + sums[18])))
        return Ur, Vs, Vt, tVs, tVt
    else:
        for i in range(nodal_point.shape[0]):
            pnm = asso_legendre(nodal_point[i],n_max)
            sums = infinite_sum(nodal_point[i],rR)
            c22 = np.sqrt((2 * love['degree_list'][:] + 1) / np.pi / 4) / rs**2 * pnm[:,6]
            c32 = c22 / np.sqrt((love['degree_list'][:] + 1) * (love['degree_list'][:]))
            c12 = c32 * np.sqrt((love['degree_list'][:] + 2) * (love['degree_list'][:] -1))
        # degree == 1~n_max
            Ur[i,0] = (c12 * love['Hs'][:,0] * pnm[:,2]).sum()
            Ur[i,1] = (c32 * love['Hs'][:,1] * pnm[:,1]).sum() 
            Ur[i,2] = (c22 * love['Hs'][:,2] * pnm[:,0]).sum() + Ur2_0 
            Ur[i,3] = (c22 * love['Hs'][:,3] * pnm[:,0]).sum() + Ur3_0# c33 == c22
        
            Vs[i,0] = (c12 * love['Ls'][:,0] * pnm[:,5]).sum() # c33 == c22
            Vt[i,0] = (c12 * love['Ls'][:,0] * pnm[:,2]).sum() / np.sin(nodal_point[i])
        
            Vs[i,1] = (c32 * love['Ls'][:,1] * pnm[:,4]).sum() # c33 == c22
            Vt[i,1] = (c32 * love['Ls'][:,1] * pnm[:,1]).sum() / np.sin(nodal_point[i]) 
        
            Vs[i,2] = (c22 * love['Ls'][:,2] * pnm[:,3]).sum() # c33 == c22
        
            Vs[i,3] = (c22 * love['Ls'][:,3] * pnm[:,3]).sum() # c33 == c22
        
        # toroidal
            tVs[i,0] = (c12 * love['Lt'][:,0] * pnm[:,2]).sum() / np.sin(nodal_point[i])
            tVt[i,0] = (c12 * love['Lt'][:,0] * pnm[:,5]).sum()
        
            tVs[i,1] = (c32 * love['Lt'][:,1] * pnm[:,1]).sum() / np.sin(nodal_point[i])
            tVt[i,1] = (c32 * love['Lt'][:,1] * pnm[:,4]).sum()
        
        
        # add infinite sums
            Ur[i,0] = -2 * Ur[i,0]
            Ur[i,1] = -2 * Ur[i,1]
        
            Vs[i,0] = -2 * Vs[i,0]
            Vs[i,1] = -2 * Vs[i,1]
        
            Vt[i,0] = -4 * Vt[i,0]
            Vt[i,1] = -2 * Vt[i,1]
        
            tVs[i,0] = -4 * tVs[i,0]
            tVs[i,1] = 2 * tVs[i,1]
            tVt[i,0] = -2 * tVt[i,0]
            tVt[i,1] = 2 * tVt[i,1]
        return Ur, Vs, Vt, tVs, tVt
        
@njit
def recurrent_legendre(n,k,pn_1,pn_2,x):
    return ((2 * n - 1) * x * pn_1 - (n - 1 + k) * pn_2) / (n - k)
@njit
def asso_legendre(theta,n_max): 
    """ Compute normalizated associated ledendre function up to n_max degrees, with k = 0~2 only.
    The normalization method is Orthonormalized. Also compute disc factor
    
    Parameters:
        theta   : float, radian
        n_max   : int, highest degree
                    
    Returns:
        p  : 2D np.ndarray in the size of (n_max+1,7), output: [pn0,pn1,pn2,dpn0,dpn1,dpn2,disc_fact]
    """
    p = np.zeros((n_max+1,7))
    x = np.cos(theta)
    y = np.sin(theta)
    p[0,0], p[1,0], p[2,0], p[3,0] = 1, x, 1.5 * x**2 - 0.5, 2.5 * x**3 - 1.5 * x # pn0
    p[1,1], p[2,1] = y, 3 * x * y
    p[3,1] = recurrent_legendre(3,1,p[2,1],p[1,1],x) # pn1
    
    p[2,2], p[3,2] = 3 * y**2, 15 * x * y**2 # pn2
    # let iteration index is degree instead of slicing index.
    degree_list = np.arange(1,n_max+1,dtype=np.int64)
    order_list = np.arange(3,dtype=np.int64)
    if theta != 0: # for north
        tx = np.cos(theta / 15)
        ty = np.sin(theta/15)
        p[1,6], p[2,6] = ty, 3 * tx * ty
        p[3,6] = recurrent_legendre(3,1,p[2,6],p[1,6],tx) 
        
        for i in range(4,n_max+1): # compute n = 4,5...
            p[i,0] = recurrent_legendre(i,0,p[i-1,0],p[i-2,0],x)
            p[i,1] = recurrent_legendre(i,1,p[i-1,1],p[i-2,1],x)
            p[i,2] = recurrent_legendre(i,2,p[i-1,2],p[i-2,2],x)
            p[i,6] = recurrent_legendre(i,1,p[i-1,6],p[i-2,6],tx)
        p[1:,6] = (1 + tx) * p[1:,6] / (ty * degree_list * (degree_list + 1))
    else:
        for i in range(4,n_max+1): # compute n = 4,5...
            p[i,0] = recurrent_legendre(i,0,p[i-1,0],p[i-2,0],x)
            p[i,1] = recurrent_legendre(i,1,p[i-1,1],p[i-2,1],x)
            p[i,2] = recurrent_legendre(i,2,p[i-1,2],p[i-2,2],x)
        p[1:,6] = 1
        
    if theta == 0:
        p[1:,3] = degree_list * (degree_list + 1) / 2
    elif theta == np.pi:
        tmp = np.ones(degree_list.shape[0])
        tmp[1::2] = -1
        p[1:,3] = degree_list * (degree_list + 1) / 2 * tmp
    else:
        p[1:,3:-1] = (-p[:-1,:3] * (degree_list[:,np.newaxis] + order_list) + p[1:,:3] * x * degree_list[:,np.newaxis]) / y
    nor_1 = np.sqrt((2 * degree_list + 1 ) / 4 / np.pi) 
    nor_2 = nor_1 / np.sqrt(degree_list * (degree_list+1))
    nor_3 = nor_2[1:] / np.sqrt((degree_list[1:] + 2) * (degree_list[1:] - 1))
    for k in [0,3]:
        p[1:,k] = p[1:,k] * nor_1
        p[1:,k+1] = p[1:,k+1] * nor_2
    p[2:,2] = p[2:,2] * nor_3
    p[2:,5] = p[2:,5] * nor_3
    p[1,2] = 0
    p[1,5] = 0
    return p[1:,:]

@njit
def infinite_sum(theta,t):
    sums = np.zeros(23,)
    x = np.cos(theta)
    y = np.sin(theta)
    sq1 = np.sqrt(1 - 2 * t * x + t**2)
    sq3 = sq1**3
    sq5 = sq1**5
    tQ = 1 - t * x + sq1
    ss2 = np.sin(theta/2)
    cc2 = np.cos(theta/2)
    sums[0] = t * (-2 * t + t**3 + x - t**2 * x + t * x**2) / sq5
    sums[1] = (t * x - t**2) / sq3
    sums[2] =  1 / sq1 - 1
    sums[3] = -np.log(tQ)+np.log(2)
    sums[4] = t * y * (1 + t * x - 2 * t**2) / sq5
    sums[5] =  t * y / sq3
    sums[6] = t * y * ( 1 + 1 / sq1) / tQ
    sums[7] =  t * y / tQ / sq1
    sums[8] =  3 * t**2 * y**2 / sq5
    sums[9] = 2 * t**2 / tQ / sq1 + 2 / sq1 + (t * x - 1) / sq3 -1
    sums[10] = 2 * t * x / tQ / sq1 - t * (x - t) / sq3
    if theta > 0.002:
        
        sums[11] = 1 - 1 / sq1 + ss2**2 /cc2**2 * np.log(tQ / 2) + 2 * x * \
            (np.log(1 + 2 * (1 + t) / (x - t + sq1) * ss2**2) + np.log(cc2**2)) / y**2
            
        sums[21] = 2 * t / y / sq1 + ss2 / cc2**3 * np.log((x - t + sq1) / 2 / cc2**2) - t * y * (1 + 1 / sq1) / tQ \
                + 4 * x * (np.log(1 - 2 * (1 + t) / tQ * ss2**2) - np.log(cc2**2)) / y**3 + t * y / sq3
    else:
        sums[11] = 1 - 1 / sq1 + ss2**2 /cc2**2 * np.log(tQ / 2) + x / 2 /cc2**2 * \
           (2 * (1 + t) / (x - t + sq1) - 2 * (1 + t)**2 / (x - t + sq1) / (x - t + sq1)*ss2**2 + (-1 - ss2**2/2))
        sums[21] = 2 * t / sq1 / y + ss2/cc2**3 * np.log((x - t + sq1) / 2 / cc2**2) - t * y * (1 + 1 / sq1) / tQ + x / cc2**2 * (-2 * (1 + t) / tQ - 2 * (1 + t)**2 / tQ**2 *ss2**2 - (- 1 - ss2**2 / 2)) / y + t * y / sq3
    sums[12] = 1 - 1 / sq1 + 2 * t * x / tQ
    sums[13] = -sums[4]
    sums[14] = -sums[5]
    sums[15] = -sums[6]
    sums[16] = t * x / sq3 - 3 * t**2 *y**2 / sq5
    sums[17] = -t**2 / sq1 / tQ - t * (t - x) / sq3
    sums[18] = -1 / t / (x - t + sq1) / sq1 + 1 / 2 / t / cc2**2 - (t * x - 1) / sq3
    sums[19] = (-4 / (x - t + sq1) / sq1 + 2 * t / sq1 + 2 / cc2**2 - (2 * t**2 * x - 2 * t * x**2) / sq3) / y \
            - t * y / sq3 - 3 * t * y * (t * x - 1) / sq5
    sums[20] = 2 * t * y / tQ / sq3 * (t**2 / tQ - sq1 + t / (t - x - sq1) - t * x + t * sq1 /2 /cc2**2 ) \
           + t * y / sq3 - 3 * t**2 * y * (t - x) / sq5
           
    sums[22] = t * y * (1 / sq3 + 2 * x / (t - x - sq1) / tQ / sq1 - 1 / tQ / cc2**2)
    return sums


