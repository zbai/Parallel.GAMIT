# Author:   Chong-You (Kevin) Wang, 2025-09, OSU
""" 
    Code to compute love numbers for dislocation. This code is the python version of DLN.m provided by
    Zhou. et al. (2019a). This code also compute love numbers for tide, load and shear, but the correctness
    is yet validated.

usage: 
    CM_models = Read_models(core_file=core_file,mantle_file=mantle_file)
    DLN_model = DLN(CM_models)
    love = DLN_obj.loves_dislocation(depth_s, depth_f, degree_b, degree_e, degree_i)
    love contains Hs, Ls, Ks and Lt

Zhou, J., Pan, E., & Bevis, M. (2019). A point dislocation in a layered, transversely isotropic and self-gravitating
Earth. Part I: analytical dislocation Love numbers. Geophysical Journal International, 217(3), 1681-1705.

"""

import numpy as np
import os
from numba import njit
from cmath import sqrt as csqrt # for complex value

CONST_G = 6.6732e-11
CONST_R = 6.371e6

# %%
@njit
def find_layer(mantle_layer,depth):
    d = 1 - depth / CONST_R
    return np.argmax(d <= mantle_layer) - 1

# %%
@njit
def spherical_bessel2(x,n):
    n_max = n + 1 if n >=1000 else 1000
    jn = np.zeros(n,dtype=type(x))
    Zn = np.zeros(n_max,dtype=type(x))
    Zn[-1] = x**2 / (2 * n_max + 1)
    for i in range(n_max-1,0,-1):
        Zn[i-1] = x**2 / (2 * i + 1 - Zn[i])
    Zn1 = 1 - x * np.cos(x) / np.sin(x)
    Zn[:n] = Zn[:n] / Zn[0] * Zn1
    jn[0] = np.sin(x) / x
    for i in range(2,n+1):
        jn[i-1] = Zn[i-2] * jn[i-2] / x
    return jn

# %%
@njit
def cal_eig(matrix):
    # calculate eignvalue and vector and sort them.
    eig_val, eig_vec = np.linalg.eig(matrix.astype(np.complex128)) # I convert matrix to complex values so that numba can manage this function
    ind = np.argsort(-(eig_val.real)) # numpy.sort does not have descending mode (only ascending),so i make input negative and sort it
    eig_val = eig_val[ind]
    eig_vec = eig_vec[:,ind]
    return eig_val, eig_vec

@njit
def inner_core_sol_0(core_data):
    n = 0
    r = core_data[0,0] * CONST_R
    rho = core_data[0,1]
    gamma = core_data[0,4]
    #gravity = gamma * r
    sigma = core_data[0,2] + 2 * core_data[0,3]
    vp2 = sigma / rho
    miu = core_data[0,3]
    vs2 = miu / rho
    x = 4 * gamma / vp2
    y = 4 * n * (n + 1) * gamma**2 / vp2 / vs2
    kk1 = (x + np.sqrt(x**2 + y)) / 2
    kk2 = (x - np.sqrt(x**2 + y)) / 2
    f1 = vs2 / gamma * kk1
    h1 = f1 - n - 1
    #f2 = vs2 / gamma * kk2
    #h2 = f2 - n - 1
    k1 = np.sqrt(kk1) # kk1 is always positive. no need to consider issue of complex value
    x = k1 * r
    jn = spherical_bessel2(x,n+2)
    output = np.zeros(3,)
    output[0] = (n * h1 *jn[n] - f1 * k1 * r * jn[n+1]) / r
    output[1] = r * (-sigma * f1 * kk1 * r**2 * jn[n] + \
                2 * miu * (n * (n - 1) * h1 * jn[n] + \
                (2 * f1 + n * (n + 1)) * k1 * r * jn[n+1])) / r**2
    output[2] = -3 * gamma * f1 * jn[n]
    return output

@njit
def inner_core_sol_complex(core_data,n):
    r = core_data[0,0] * CONST_R
    rho = core_data[0,1]
    gamma = core_data[0,4]
    gravity = gamma * r
    sigma = core_data[0,2] + 2 * core_data[0,3]
    vp2 = sigma / rho
    miu = core_data[0,3]
    vs2 = miu / rho
    x = 4 * gamma / vp2
    y = 4 * n * (n + 1) * gamma**2 / vp2 / vs2
    kk1 = (x + np.sqrt(x**2 + y)) / 2
    kk2 = (x - np.sqrt(x**2 + y)) / 2
    f1 = vs2 / gamma * kk1
    h1 = f1 - n - 1
    f2 = vs2 / gamma * kk2
    h2 = f2 - n - 1
    k1 = np.sqrt(kk1) # kk1 is always positive. no need to consider issue of complex value
    x = k1 * r
    jn = spherical_bessel2(x,n+2)
    
    y = np.zeros((6,3),dtype=np.complex128)
    y[0,0] = (n * h1 *jn[n] - f1 * k1 * r * jn[n+1]) / r
    y[1,0] = (-sigma * f1 * kk1 * r**2 * jn[n] + \
    2 * miu * (n * (n - 1) * h1 * jn[n] + \
    (2 * f1 + n * (n + 1)) * k1 * r * jn[n+1])) / r**2
    y[3,0] = (miu *(kk1 * r**2 * jn[n] + 2 * (n - 1) * h1 * jn[n] - 2 * (f1 + 1) * k1 * r * jn[n+1]) ) / r**2
    y[4,0] = -3 * gamma * f1 * jn[n]
    y[5,0] = (2 * n + 1) * y[4,0] + 3 * n * gamma * h1 * jn[n]
            
    k2 = csqrt(kk2) # handle negetive number (kk2). k2 is always complex value.
    x = k2 * r
    jn = spherical_bessel2(x,n+2)
    y[0,1] = (n * h2 * jn[n] - f2 * k2 * r * jn[n+1]) / r

    y[1,1]=(-sigma * f2 * kk2 * r**2 * jn[n] \
        + 2 * miu * (n * (n - 1) * h2 * jn[n] \
        +(2 * f2 + n * (n + 1)) * k2 * r * jn[n+1])) / r**2
    y[3,1] = (miu * (kk2 * r**2 * jn[n] + 2 * (n - 1) * h2 * jn[n] - 2 * (f2 + 1) * k2 * r * jn[n+1])) / r**2
    y[4,1] = -3 * gamma * f2 * jn[n]
    y[5,1] = (2 * n + 1) * y[4,1] + 3 * n * gamma * h2 * jn[n]

    y[0,2] = n / r
    y[1,2] = 2 * miu * n * (n - 1) / r**2
    y[3,2] = 2 * miu * (n - 1) / r**2
    y[4,2] = -n * gamma
    y[5,2] = (2 * n + 1) * y[4,2]+ 3 * n * gamma

    y_array = y[5,:] - 4 * np.pi * CONST_G / gravity * r * y[1,:]
    A = np.zeros((2,2),dtype=np.complex128)
    b = np.zeros((2,1),dtype=np.complex128)
    A[0,0] = y[3,0]
    A[0,1] = y[3,1]
    A[1,0] = rho * gravity * y[0,0] - y[1,0] + rho * y[4,0]
    A[1,1] = rho * gravity * y[0,1] - y[1,1] + rho * y[4,1]
    b[0,0] = -y[3,2]
    b[1,0] = -(rho * gravity * y[0,2] - y[1,2] + rho * y[4,2])
    
    c = np.linalg.solve(A,b).flatten()
    output = np.zeros((2,),np.complex128)
    output[0] = y[4,0] * c[0] + y[4,1] * c[1] + y[4,2]  # phi = y[4,0] * c[0] + y[4,1] * c[1] + y[4,2]
    output[1] = y_array[0] * c[0] + y_array[1] * c[1] + y_array[2] # Y = y1 * c[0] + y2 * c[1] + y3
    return output

# %%
@njit
def layer_propagator(eig_val,eig_vec,et):
    m = eig_val.shape[0] //2 
    C1 = np.eye(m) * np.exp(-eig_val[:m] * et)
    C2 = np.eye(m) * np.exp(eig_val[m:] * et)
    D1 = np.zeros((2 * m, 2 * m),dtype=eig_val.dtype)
    D2 = np.zeros((2 * m, 2 * m),dtype=eig_val.dtype)
    #D1 = np.zeros((2 * m, 2 * m),dtype=type(eig_val[0]))
    #D2 = np.zeros((2 * m, 2 * m),dtype=type(eig_val[0]))
    D1[:m,:m] = eig_vec[:m,:m] @ C1
    D1[:m,m:] = eig_vec[:m,m:]
    D1[m:,:m] = eig_vec[m:,:m]
    D1[m:,m:] = eig_vec[m:,m:] @ C2
        
    D2[:m,:m] = eig_vec[:m,:m]
    D2[:m,m:] = eig_vec[:m,m:] @ C2
    D2[m:,:m] = eig_vec[m:,:m] @ C1
    D2[m:,m:] = eig_vec[m:,m:]
    return np.linalg.solve(D1.T,D2.T).T

@njit
def dvp_propagator(matrix_s2,matrix_s1):
    m = matrix_s2.shape[0] //2
    tmp = (np.eye(m) - matrix_s1[m:,:m] @ matrix_s2[:m,m:])
    tmp1 = np.linalg.solve(tmp.T,matrix_s2[:m,m:].T).T
    tmp2 = np.linalg.solve(tmp.T,matrix_s2[m:,m:].T).T
    matrix_prop = np.zeros((2 * m, 2 * m),dtype=tmp.dtype)
    matrix_prop[:m,:m] = matrix_s1[:m,:m] @ (matrix_s2[:m,:m] + tmp1 @ matrix_s1[m:,:m] @ matrix_s2[:m,:m])
    matrix_prop[:m,m:] = matrix_s1[:m,:m] @ tmp1 @ matrix_s1[m:,m:] + matrix_s1[:m,m:]
    matrix_prop[m:,:m] = matrix_s2[m:,:m] + tmp2 @ matrix_s1[m:,:m] @ matrix_s2[:m,:m]
    matrix_prop[m:,m:] = tmp2 @ matrix_s1[m:,m:]
    return matrix_prop
@njit
def matrix_cd(matrix_1,matrix_2,tM,delta):
    n = matrix_2.shape[0]
    m = n // 2
    matrix_eye = np.eye(m,dtype=np.complex128)
    c_matrix = np.zeros((2 * n, 2 * n),dtype=np.complex128)
    c_matrix[:m,:m] = -matrix_eye
    c_matrix[:m,m:n] = matrix_2[:m,:m]
        
    c_matrix[m:n,m:n] = matrix_2[m:,:m]
    c_matrix[m:n,n:m+n] = -matrix_eye
        
    c_matrix[n:m+n,m:n] = -matrix_eye
    c_matrix[n:m+n,n:m+n] = matrix_1[:m,m:]
    c_matrix[n:m+n,n+m:] = matrix_1[:m,:m] @ tM[:m,:]
        
    c_matrix[n+m:,n:m+n] = matrix_1[m:,m:]
    c_matrix[n+m:,n+m:] = matrix_1[m:,:m] @ tM[:m,:] - tM[m:,:]
        
    d_vector = np.zeros((2 * n, delta.shape[1]),dtype=np.complex128)
    d_vector[:m,:] = -matrix_2[:m,m:] @ delta[:m,:]
    d_vector[m:n,:] = -matrix_2[m:,:m] @ delta[:m,:] + delta[m:,:]
    return c_matrix, d_vector

@njit
def matrix_cd_internal1(matrix_1,matrix_2,tM,T_top):
    n = matrix_1.shape[0]
    m = n // 2
    matrix_eye = np.eye(m,dtype=np.complex128)
    c_matrix = np.zeros((2 * n, 2 * n),dtype=np.complex128)
    c_matrix[:m,:m] = -matrix_eye
    c_matrix[:m,m:n] = matrix_2[:m,:m]
    
    c_matrix[m:n,m:n] = matrix_2[m:,:m]
    c_matrix[m:n,n:m+n] = -matrix_eye
        
    c_matrix[n:m+n,m:n] = -matrix_eye
    c_matrix[n:m+n,n:m+n] = matrix_1[:m,m:]
    c_matrix[n:m+n,n+m:] = matrix_1[:m,:m] @ tM[:m,:]
        
    c_matrix[n+m:,n:m+n] = matrix_1[m:,m:]
    c_matrix[n+m:,n+m:] = matrix_1[m:,:m] @ tM[:m,:] - tM[m:,:]
        
    d_vector = np.zeros((2 * n, T_top.shape[1]),dtype=np.complex128)
    d_vector[:m,:] = -matrix_2[:m,m:] @ T_top
    d_vector[m:n,:] = -matrix_2[m:,m:] @ T_top
    return c_matrix, d_vector
@njit
def matrix_cd_internal0(matrix_1,matrix_2,T_bottom):
    n = matrix_1.shape[0]
    m = n // 2
    matrix_eye = np.eye(m,dtype=np.complex128)
    c_matrix = np.zeros((2 * n, 2 * n),dtype=np.complex128)
    c_matrix[:m,:m] = -matrix_eye
    c_matrix[:m,m:n] = matrix_2[:m,:m]
    
    c_matrix[m:n,m:n] = matrix_2[m:,:m]
    c_matrix[m:n,n:m+n] = -matrix_eye
        
    c_matrix[n:m+n,m:n] = -matrix_eye
    c_matrix[n:m+n,n:m+n] = matrix_1[:m,m:]
    c_matrix[n:m+n,n+m:] = matrix_1[:m,:m]
        
    c_matrix[n+m:,n:m+n] = matrix_1[m:,m:]
    c_matrix[n+m:,n+m:] = matrix_1[m:,:m]
    d_vector = np.zeros((2*n,T_bottom.shape[1]),dtype=np.complex128)
    d_vector[n+m:,:] = T_bottom
    return c_matrix, d_vector

# %%
@njit
def spherical_bessel(x):
    j1 = (np.sin(x) - x * np.cos(x)) / x**2
    y1 = -(np.cos(x) + x * np.sin(x)) / x**2
    j2 = (x**2 * np.sin(x) - 2 * (np.sin(x) - x * np.cos(x))) / x**2
    y2 = -(x**2 * np.cos(x) - 2 * (np.cos(x) + x * np.sin(x))) / x**2
    return j1, y1, j2, y2

@njit
def diff_eq2_core(core_data,layer_id,m_nH):
    # spheroidal part in core
    # id == 1
    # s0 and s1 are not involved in
    eq = np.zeros((2,2))
    rho_r = core_data[layer_id+1,1]
    k = core_data[layer_id+1,4]
    eq[0,0] = (4 * np.pi * CONST_G * rho_r / k - (m_nH + 1)) / m_nH
    eq[0,1] = 1
    eq[1,0] = 8 * np.pi * CONST_G * rho_r * (m_nH - 1) / k / m_nH**2
    eq[1,1] = ((m_nH - 1) - 4 * np.pi * CONST_G * rho_r / k + 1) / m_nH
    return eq

@njit
def diff_eq2_T(mantle_data,layer_id,m_nH):
    # toroidal part in mantle
    # id == 3
    # s0 and s1 are not involved in
    eq = np.zeros((2,2))
    c44 = mantle_data[layer_id+1,4]
    c66 = mantle_data[layer_id+1,5]
    eq[0,0] = 1 / m_nH
    eq[0,1] = 1 /c44
    eq[1,0] = c66 * (m_nH * (m_nH + 1) - 2) / m_nH**2
    eq[1,1] = -2 / m_nH
    return eq
@njit
def diff_eq2_S(mantle_data,layer_id,s1,s0,CONST_C):
    # spheroidal part in mantle
    # For id == 2 , only for degree == 0 
    eq = np.zeros((2,2))
    if s1 == s0:
        sbar = s0
    else:
        sbar = 2 / 3 * (s1**3 - s0**3) / (s1**2 - s0**2)
    rho_s = mantle_data[layer_id + 1,1]
    c11 = mantle_data[layer_id + 1,2]
    c33 = mantle_data[layer_id + 1,3]
    c13 = mantle_data[layer_id + 1,6]
    c44 = mantle_data[layer_id+1,4]
    c66 = mantle_data[layer_id+1,5]
    g_s = mantle_data[layer_id + 1,7]
    c12 = c11 - 2 * c66
    eq[0,0] = -2 * c13 / c33
    eq[0,1] = 1 / c33
    eq[1,0] = -4 * CONST_C * rho_s * sbar * g_s + 2 * (c33 * (c11 + c12) - 2 * c13**2) / c33
    eq[1,1] = 2 * (c13 / c33 - 1) + 1
    return eq
@njit
def diff_eq4(mantle_data,layer_id,s1,s0,CONST_B,CONST_C,m_nH=0):
    # only accept for id==2 and m_nH==0
    # adding m_nH in variables for conveniently use 
    eq = np.zeros((4,4))
    if s1 == s0:
        sbar = s0
    else:
        sbar = 2 / 3 * (s1**3 - s0**3) / (s1**2 - s0**2)
    rho_s = mantle_data[layer_id + 1,1]
    c11 = mantle_data[layer_id + 1,2]
    c33 = mantle_data[layer_id + 1,3]
    #c44 = self.mantle_data[layer_id + 1,4]
    c66 = mantle_data[layer_id + 1,5]
    c13 = mantle_data[layer_id + 1,6]
    g_s = mantle_data[layer_id + 1,7]
    c12 = c11 - 2 * c66
    eq[0,0] = -2 * c13 / c33
    eq[0,1] = 1 / c33
    eq[1,0] = -4 * CONST_C * rho_s * sbar * g_s + 2 * (c33 * (c11 + c12) - 2 * c13**2) / c33
    eq[1,1] = 2 * (c13 / c33 - 1) + 1
    eq[2,0] = -CONST_B * rho_s * sbar
    eq[3,0] = -CONST_B * rho_s * sbar
    return eq
@njit
def diff_eq6(mantle_data,layer_id,s1,s0,CONST_B,CONST_C,m_nH):
        # only accepts for id == 2 and m_nH ~=0
    eq = np.zeros((6,6))
    if s1 == s0:
        sbar = s0
    else:
        sbar = 2 / 3 * (s1**3 - s0**3) / (s1**2 - s0**2)
    rho_s = mantle_data[layer_id + 1,1]
    c11 = mantle_data[layer_id + 1,2]
    c33 = mantle_data[layer_id + 1,3]
    c44 = mantle_data[layer_id + 1,4]
    c66 = mantle_data[layer_id + 1,5]
    c13 = mantle_data[layer_id + 1,6]
    g_s = mantle_data[layer_id + 1,7]
    c12 = c11 - 2 * c66
    NN = m_nH * (m_nH + 1)
            
    eq[0,0] = -2 * c13 / c33 / m_nH
    eq[0,1] = NN * c13 / c33 / m_nH**2
    eq[0,3] = 1 / c33
            
    eq[1,0] = -1
    eq[1,1] = 1 / m_nH
    eq[1,4] = 1 / c44
            
    eq[2,0] = -CONST_B * rho_s * sbar
    eq[2,2] = -(m_nH + 1) / m_nH
    eq[2,5] = 1
            
    eq[3,0] = (-4 * CONST_C * rho_s * sbar * g_s + 2 * (c33 * (c11 + c12) - 2 * c13**2) / c33) / m_nH**2
    eq[3,1] = (CONST_C * NN * rho_s * sbar * g_s - NN * (c33 * (c11 + c12) - 2 * c13**2) / c33) / m_nH**3
    eq[3,2] = -CONST_C * (m_nH + 1) * rho_s * sbar / m_nH**3
    eq[3,3] = (2 * (c13 / c33 - 1) + 1) / m_nH
    eq[3,4] = NN / m_nH**2
    eq[3,5] = CONST_C * rho_s * sbar / m_nH**2
            
    eq[4,0] = (CONST_C * rho_s * sbar * g_s + (2 * c13**2 - c33 * (c11 + c12)) / c33 ) /m_nH
    eq[4,1] = (-(c11 - c12) + NN * (c11 * c33 - c13**2) / c33) / m_nH**2
    eq[4,2] = CONST_C * rho_s * sbar / m_nH**2
    eq[4,3] = -c13 / c33
    eq[4,4] = -2 / m_nH
            
    eq[5,0] = -(m_nH + 1) * CONST_B * rho_s * sbar / m_nH
    eq[5,1] = NN * CONST_B * rho_s * sbar / m_nH**2
    eq[5,5] = 1
    return eq

# %%
@njit
def DVP_S_cmd2surface(mantle_data,diff_eq,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,m_nH,matrix_dim=4):
    rs = (CONST_R - depth_s) / CONST_R
    rf = (CONST_R - depth_f) / CONST_R
    if rs <= rf:
        ID_SF = 0
        m_layer_1 = m_source_layer
        m_layer_2 = m_field_layer
    else:
        ID_SF = 1
        m_layer_1 = m_field_layer
        m_layer_2 = m_source_layer
    #matrix_dim = int(diff_eq.__name__[-1]). # is not supported by numba
    matrix_dvp_1 = np.eye(matrix_dim,dtype=np.complex128)
    for i in range(m_layer_1+1):
        s0 = mantle_data[i,0]
        if i == m_layer_1:
            if ID_SF == 0:
                s1 = rs
            else:
                s1 = rf
        else:
            s1 = mantle_data[i+1,0]
        eq = diff_eq(mantle_data,i,s1,s0,CONST_B,CONST_C,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_1 = dvp_propagator(matrix_dvp_1,matrix_s2)
        
    matrix_dvp_2 = np.eye(matrix_dim,dtype=np.complex128)
        
    if m_layer_1 == m_layer_2:
        if ID_SF == 0:
            s0 = rs
            s1 = rf
        else:
            s0 = rf
            s1 = rs
        eq = diff_eq(mantle_data,i,s1,s0,CONST_B,CONST_C,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_2 = dvp_propagator(matrix_dvp_2,matrix_s2)
            
    else:
        for i in range(m_layer_1,m_layer_2+1):
            if i == m_layer_1:
                if ID_SF == 0:
                    s0 = rs
                else:
                    s0 = rf
                s1 = mantle_data[i+1,0]
            elif i == m_layer_2:
                if ID_SF == 0:
                    s1 = rf
                else:
                    s1 = rs
                s0 = mantle_data[i,0]
            else:
                s0 = mantle_data[i,0]
                s1 = mantle_data[i+1,0]
                
            eq = diff_eq(mantle_data,i,s1,s0,CONST_B,CONST_C,m_nH)
            eig_val, eig_vec = cal_eig(eq)
            t = np.log(s1/s0)
            if (t == 0):
                matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
            else:
                matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
                    
            matrix_dvp_2 = dvp_propagator(matrix_dvp_2,matrix_s2)
            
    matrix_dvp_3 = np.eye(matrix_dim,dtype=np.complex128)
    for i in range(m_layer_2,mantle_data.shape[0]-1):
        s1 = mantle_data[i+1,0]
        if i == m_layer_2:
            if ID_SF == 0:
                    s0 = rf
            else:
                    s0 = rs
        else:
            s0 = mantle_data[i,0]
            
        eq = diff_eq(mantle_data,i,s1,s0,CONST_B,CONST_C,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_3 = dvp_propagator(matrix_dvp_3,matrix_s2)
    return matrix_dvp_1,matrix_dvp_2,matrix_dvp_3, ID_SF
@njit
def DVP_T_cmd2surface(mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,m_nH):
    rs = (CONST_R - depth_s) / CONST_R
    rf = (CONST_R - depth_f) / CONST_R
    if rs <= rf:
        ID_SF = 0
        m_layer_1 = m_source_layer
        m_layer_2 = m_field_layer
    else:
        ID_SF = 1
        m_layer_1 = m_field_layer
        m_layer_2 = m_source_layer
        
    matrix_dvp_1 = np.eye(2,dtype=np.complex128)
    for i in range(m_layer_1+1):
        s0 = mantle_data[i,0]
        if i == m_layer_1:
            if ID_SF == 0:
                s1 = rs
            else:
                s1 = rf
        else:
            s1 = mantle_data[i+1,0]
        eq = diff_eq2_T(mantle_data,i,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_1 = dvp_propagator(matrix_dvp_1,matrix_s2)
        
    matrix_dvp_2 = np.eye(2,dtype=np.complex128)
    if m_layer_1 == m_layer_2:
        if ID_SF == 0:
            s0 = rs
            s1 = rf
        else:
            s0 = rf
            s1 = rs
        eq = diff_eq2_T(mantle_data,i,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_2 = dvp_propagator(matrix_dvp_2,matrix_s2)
            
    else:
        for i in range(m_layer_1,m_layer_2+1):
            if i == m_layer_1:
                if ID_SF == 0:
                    s0 = rs
                else:
                    s0 = rf
                s1 = mantle_data[i+1,0]
            elif i == m_layer_2:
                if ID_SF == 0:
                    s1 = rf
                else:
                    s1 = rs
                s0 = mantle_data[i,0]
            else:
                s0 = mantle_data[i,0]
                s1 = mantle_data[i+1,0]
            eq = diff_eq2_T(mantle_data,i,m_nH)
            eig_val, eig_vec = cal_eig(eq)
            t = np.log(s1/s0)
            if (t == 0):
                matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
            else:
                matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
                    
            matrix_dvp_2 = dvp_propagator(matrix_dvp_2,matrix_s2)
            
    matrix_dvp_3 = np.eye(2,dtype=np.complex128)
    for i in range(m_layer_2,mantle_data.shape[0]-1):
        s1 = mantle_data[i+1,0]
        if i == m_layer_2:
            if ID_SF == 0:
                s0 = rf
            else:
                s0 = rs
        else:
            s0 = mantle_data[i,0]
            
        eq = diff_eq2_T(mantle_data,i,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(eq.shape[0],dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_3 = dvp_propagator(matrix_dvp_3,matrix_s2)
    return matrix_dvp_1,matrix_dvp_2,matrix_dvp_3, ID_SF

# %%
@njit
def DVP_S_core(core_data,m_nH):
    matrix_dvp_init = np.eye(2,dtype=np.complex128)
    for i in range(core_data.shape[0]-1):
        s1 = core_data[i+1,0]
        s0 = core_data[i,0]
        t = np.log(s1/s0)
        eq = diff_eq2_core(core_data,i,m_nH)
        eig_val, eig_vec = cal_eig(eq)
        t = np.log(s1/s0)
        if (t == 0):
            matrix_s2 = np.eye(2,dtype=np.complex128)
        else:
            matrix_s2 = layer_propagator(eig_val,eig_vec,m_nH * t)
        matrix_dvp_init = dvp_propagator(matrix_dvp_init,matrix_s2)

    return matrix_dvp_init
@njit
def DVP_S_core_mantle_boundary(core_data,ppgC,m_nH,CONST_B,CONST_C,CONST_RHO,CONST_GR):
    if m_nH < 100:
        y = np.zeros((2,),dtype=np.complex128)
        tt = inner_core_sol_complex(core_data,m_nH) # is complex value
        y[1] =(tt[1] - ppgC[1,0] * tt[0] * m_nH) / ppgC[1,1]  # note in matlab the operator is \ instead of /
        y[0] = ppgC[0,1] * y[1] + ppgC[0,0] * tt[0] * m_nH
        y[1] = y[1] / y[0]
        y[0] = 1
    else:
        y = np.array([ppgC[0,1],1])
    g_s = core_data[-1, 4] * core_data[-1, 0] * CONST_R / CONST_GR
    rho_s = core_data[-1, 1] / CONST_RHO
    rc = core_data[-1, 0]
    tM = np.zeros((6, 3),dtype=np.complex128)
    tM[2, 0] = y[0]
    tM[3, 0] = rc * CONST_C * rho_s * y[0] / m_nH**2
    tM[5, 0] = y[1] + rc * CONST_B * rho_s / g_s * y[0] / m_nH
    tM[1, 1] = 1
    tM[0, 2] = 1
    tM[3, 2] = rc * CONST_C * rho_s * g_s / m_nH
    tM[5, 2] = rc * CONST_B * rho_s
    return tM

# %%
@njit
def DVP_S_dislocation(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_RHO,CONST_GR,m_nH):
        Hs = np.zeros((4,))
        Ls = np.zeros((4,))
        Ks = np.zeros((4,))
        ppgC = DVP_S_core(core_data,m_nH)
            
        matrix_dvp_1,matrix_dvp_2,matrix_dvp_3,ID_SF = DVP_S_cmd2surface(mantle_data,diff_eq6,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,m_nH,matrix_dim=6)
        if ID_SF == 0:
            matrix_dvp_g = matrix_dvp_1
            matrix_dvp_p = dvp_propagator(matrix_dvp_2,matrix_dvp_3)
        else:
            matrix_dvp_g = dvp_propagator(matrix_dvp_1,matrix_dvp_2)
            matrix_dvp_p = matrix_dvp_3
        tM = DVP_S_core_mantle_boundary(core_data,ppgC,m_nH,CONST_B,CONST_C,CONST_RHO,CONST_GR)
    
        # Dislocation
        Step = np.zeros((6, 4),dtype=np.complex128)
        c13 = mantle_data[m_source_layer+1, 6]
        c33 = mantle_data[m_source_layer+1, 3]
        c66 = mantle_data[m_source_layer+1, 5]
        c11 = mantle_data[m_source_layer+1, 2]
        c12 = c11 - 2 * c66

        Step[4, 0] = c66 / 2
        Step[1, 1] = -m_nH / 2
        Step[0, 2] = c13 / c33
        Step[3, 2] = -(c33 * (c11 + c12) - 2 * c13**2) / (c33 * m_nH)
        Step[4, 2] = -Step[3, 2] / 2 * m_nH
        Step[0, 3] = 1
        
        if m_nH == 1:
            s11 = matrix_dvp_p[:3,:3]
            s12 = matrix_dvp_p[:3,3:]
            s21 = matrix_dvp_p[3:,:3]
            s22 = matrix_dvp_p[3:,3:]
            pa = dvp_propagator(matrix_dvp_g,matrix_dvp_p)
            p11 = pa[:3,:3]
            p12 = pa[:3,3:]
            p21 = pa[3:,:3]
            p22 = pa[3:,3:]
            t0 = tM[3:,:2] - p21 @ tM[:3,:2]
            Ta_p = np.linalg.solve(s22,Step[3:,1:]-s21 @ Step[:3,1:]) 
            Ua_p = s11 @ Step[:3,1:] + s12 @ Ta_p
            t1 = -p22 @ Ta_p
            tx = np.zeros((3,3),dtype=np.complex128)
            tx[:2,:] = np.linalg.solve(t0[1:3,:],t1[1:3,:])
            U = p11 @ tM[:3,:3] @ tx + p12 @ Ta_p
            c3 = U[2,:] + Ua_p[2,:]
            Ua = np.zeros((3,3),dtype=np.complex128)
            Ua[:2,:] = U[:2,:] + c3 + Ua_p[:2,:]
            #Ua = np.vstack((U[:2,:] + c3 + Ua_p[:2,:],np.zeros((1,3))))
            if ID_SF == 0: # I skipped inter because they are not used in the following.
                Urf = np.linalg.solve(matrix_dvp_3[:3,:3],Ua) 
                #Trf = matrix_dvp_3[3:,:3] @ Urf
            else:
                Urs = np.linalg.solve(matrix_dvp_p[:3,:3],Ua)
                Trs = matrix_dvp_p[3:,:3] @ Urs
                Urs = Urs - Step[:3,1:]
                Trs = Trs - Step[3:,1:]
                Urf = np.linalg.solve(matrix_dvp_2[:3,:3],(Urs - matrix_dvp_2[:3,3:] @ Trs))
            Hs[1:] = Urf[0,:].real
            Ls[1:] = Urf[1,:].real
            Ks[1:] = -Urf[2,:].real
        else: 
            Cx, dx = matrix_cd(matrix_dvp_g,matrix_dvp_p,tM,Step)
            dx0 = np.linalg.solve(Cx,dx)
            if ID_SF == 0:
                if (depth_f == 0) and (depth_s == 0):
                    sstep = np.zeros((12,4),dtype=np.complex128)
                    sstep[6:9,:] = Step
                    dx2 = dx0 + sstep
                else:
                    Cx1, dx1 = matrix_cd_internal0(matrix_dvp_2,matrix_dvp_3,dx0[6:9,:] + Step[3:,:])
                    dx2 = np.linalg.solve(Cx1,dx1)
                Hs = dx2[3,:].real
                Ls = dx2[4,:].real
                Ks = -dx2[5,:].real
            else:
                Cx1, dx1 = matrix_cd_internal1(matrix_dvp_1,matrix_dvp_2,tM,dx0[6:9,:])
                dx2 = np.linalg.solve(Cx1,dx1)
                Hs = dx2[3,:].real
                Ls = dx2[4,:].real
                Ks = -dx2[5,:].real
        return Hs, Ls, Ks

# %%
@njit
def DVP_S0_core(core_data):
    ppgC = np.eye(3)
    for i in range(core_data.shape[0]-1):
        s1 = core_data[i+1,0]
        s0 = core_data[i,0]
        rho = core_data[i+1,1]
        k = core_data[i+1,4]
        lamb = core_data[i+1,2]
        p = np.sqrt(4 * rho * k / lamb)
        G0 = 4 * np.pi * CONST_G * rho
        pr0 = p * s0 * CONST_R
        pr1 = p * s1 * CONST_R
        j1, y1, j2, y2 = spherical_bessel(pr0)
        B = np.array([[j1, y1, 0],
                    [lamb * (j2 + 2 * j1), lamb * (y2 + 2 * y1), 0],
                    [-G0 * np.sin(pr0) / pr0 / p, G0 * np.cos(pr0) / pr0 / p, G0]])
        ppgC = np.linalg.solve(B,ppgC) #solve is faster and more accurate than np.linalg.inv(B) @ ppgC
            
        j1, y1, j2, y2 = spherical_bessel(pr1)
        B = np.array([[j1, y1, 0],
                    [lamb * (j2 + 2 * j1), lamb * (y2 + 2 * y1), 0],
                    [-G0 * np.sin(pr1) / pr1 / p, G0 * np.cos(pr1) / pr1 / p, G0]])
        ppgC = B @ ppgC
    return ppgC

# %%
@njit
def DVP_S0_dislocation(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_GR,CONST_LAMBDA):
        # love numbers for dislocation when degree == 0
        Hs = np.zeros(4,)
        ppgC = DVP_S0_core(core_data)
        matrix_dvp_1,matrix_dvp_2,matrix_dvp_3,ID_SF = DVP_S_cmd2surface(mantle_data,diff_eq4,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,m_nH=1,matrix_dim=4)
        # regarding core
        if ID_SF == 0:
            matrix_dvp_g = matrix_dvp_1
            matrix_dvp_p = dvp_propagator(matrix_dvp_2,matrix_dvp_3)
        else:
            matrix_dvp_g = dvp_propagator(matrix_dvp_1,matrix_dvp_2)
            matrix_dvp_p = matrix_dvp_3
        
        b0 = inner_core_sol_0(core_data)
        tc = (ppgC @ b0).astype(np.complex128)
        tM = np.zeros((4,1),dtype=np.complex128)
        tM[0] = tc[0] / CONST_R
        tM[1] = tc[1] / CONST_R / CONST_LAMBDA
        tM[2] = tc[2] / CONST_GR / CONST_R
        tM[3] = tM[2] 
        Step = np.zeros((2,2),dtype=np.complex128)
        c13 = mantle_data[m_source_layer+1,6]
        c33 = mantle_data[m_source_layer+1,3]
        c66 = mantle_data[m_source_layer+1,5]
        c11 = mantle_data[m_source_layer+1,2]
        c12 = c11 - 2 * c66
        Step[0,0] = c13 / c33
        Step[1,0] = -(c33 * (c11 + c12) - 2 * c13**2) / c33
        Step[0,1] = 1

        p11 = matrix_dvp_g[:2,:2]
        p12 = matrix_dvp_g[:2,2:]
        p21 = matrix_dvp_g[2:,:2]
        p22 = matrix_dvp_g[2:,2:]
        s11 = matrix_dvp_p[:2,:2]
        tl = p11 @ tM[:2] + np.linalg.solve(p22.T,p12.T).T @ (tM[2:] - p21 @ tM[:2]) 
        A = s11 @ tl
        d = s11 @ Step
        x = -d[1,:] / A[1,0]
        if ID_SF == 0:
            UL = np.zeros((2,2),dtype=np.complex128)
            UL[0,:] = A[0,0] * x + d[0,:] 
            Uf = np.linalg.solve(matrix_dvp_3[:2,:2],UL)[0]
            #Tf = matrix_dvp_3[2:,:2] @ Uf
            Hs[2:] = Uf.real
        else:
            Uc = x * tM[:2]
            Tc = x * tM[2:]
            Tf = np.linalg.solve(matrix_dvp_1[:2,:2], (Tc - matrix_dvp_3[2:,:2] @ Uc)) #matrix_dvp_1[1,1] / (Tc - matrix_dvp_3[2:,:2] @ Uc) may be wrong because the left should be a matrix instead of a value
            Uf = matrix_dvp_3[:2,:2] @ Uc + matrix_dvp_3[:2,2:] @ Tf
            Hs[2:] = Uf[0,:].real
        return Hs

# %%
@njit
def find_degree1_Tmode(mantle_data,depth_s,depth_f,m_source_layer):
    rs = 1 - depth_s / CONST_R
    rf = 1 - depth_f / CONST_R
    tmp2,tmp3,tmp4 = mantle_data[:,0]**2, mantle_data[:,0]**3,mantle_data[:,0]**4
        # from CM boundary to source
    m_nM = mantle_data.shape[0]
    s21 = tmp2[:m_source_layer+1]
    s31 = tmp3[:m_source_layer+1]
    s41 = tmp4[:m_source_layer+1]
    
    s22 = np.zeros(s21.shape)
    s32 = np.zeros(s21.shape)
    s42 = np.zeros(s21.shape)
    s22[:-1] = tmp2[1:m_source_layer+1]
    s22[-1] = rs**2
    s32[:-1] = tmp3[1:m_source_layer+1]
    s32[-1] = rs**3
    s42[:-1] = tmp4[1:m_source_layer+1]
    s42[-1] = rs**4
    
    ''' numba does not support these
    s22 = np.hstack((tmp2[1:m_source_layer+1],rs**2))
    s32 = np.hstack((tmp3[1:m_source_layer+1],rs**3))
    s42 = np.hstack((tmp4[1:m_source_layer+1],rs**4))
    '''
    x1 = (mantle_data[1:m_source_layer+2,1] * (2/3) * (s32 - s31) / (s22 - s21) * (s42 - s41) / 4).sum()
        # from source to surface
    s22 = tmp2[m_source_layer+1:]
    s32 = tmp3[m_source_layer+1:]
    s42 = tmp4[m_source_layer+1:]
    
    if (rs != mantle_data[m_source_layer+1,0]):
        s21 = np.zeros(s22.shape)
        s31 = np.zeros(s22.shape)
        s41 = np.zeros(s22.shape)
        s21[0] = rs**2
        s31[0] = rs**3
        s41[0] = rs**4
        s21[1:] = tmp2[m_source_layer+1:-1]
        s31[1:] = tmp3[m_source_layer+1:-1]
        s41[1:] = tmp4[m_source_layer+1:-1]
    else:
        s21 = tmp2[m_source_layer:-1]
        s31 = tmp3[m_source_layer:-1]
        s41 = tmp4[m_source_layer:-1]
    ''' numba does not support these
    s21 = np.hstack((tmp2[m_source_layer:m_nM-2],rs**2))
    s31 = np.hstack((tmp3[m_source_layer:m_nM-2],rs**3))
    s41 = np.hstack((tmp4[m_source_layer:m_nM-2],rs**4))
    '''
    x2 = (mantle_data[m_source_layer+1:,1] * (2/3) * (s32 - s31) / (s22 - s21) * (s42 - s41) / 4).sum()
        
    dU = 0.5 / rs
    b2 = x1 / (x1 + x2) * dU
    if rs <= b2:
        Lt32 = b2 * rf
    else:
        b1 = b2 - dU
        Lt32 = b1 * rf
    return Lt32
@njit
def DVP_T(mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_LAMBDA,m_nH):
    if m_nH == 0:
        raise ValueError("No toroidal solution for degree 0.")
    UN = np.zeros((2,)) # UN and TN are not used
    TN = np.zeros((2,))
    Lt = np.zeros((2,))
    if m_nH == 1:
        UN[1] = find_degree1_Tmode(mantle_data,depth_s,depth_f,m_source_layer)
        return Lt
    else:
        Step = np.array([[0,m_nH/2],
                        [mantle_data[m_source_layer+1,5]/2, 0]],dtype=np.complex128)
        matrix_dvp_1,matrix_dvp_2,matrix_dvp_3,ID_SF = DVP_T_cmd2surface(mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,m_nH)
        if ID_SF == 0:
            matrix_dvp_g = matrix_dvp_1
            matrix_dvp_p = dvp_propagator(matrix_dvp_2,matrix_dvp_3)
        else:
            matrix_dvp_g = dvp_propagator(matrix_dvp_1,matrix_dvp_2)
            matrix_dvp_p = matrix_dvp_3
        tM = np.array([1,0],dtype=np.complex128).reshape(-1,1)
        Cx, dx = matrix_cd(matrix_dvp_g,matrix_dvp_p,tM,Step)
        dx0 = np.linalg.solve(Cx,dx)
        if ID_SF == 0:
            if (depth_f == 0) & (depth_s == 0):
                sstep = np.zeros((4,2),dtype=np.complex128)
                sstep[2:4,:] = Step
                dx2 = dx0 + sstep
            else:
                Cx1, dx1 = matrix_cd_internal0(matrix_dvp_2,matrix_dvp_3,np.ascontiguousarray(dx0[2,:] + Step[1,:]).reshape(1,-1))
                dx2 = np.linalg.solve(Cx1,dx1)
        else:
            Cx1, dx1 = matrix_cd_internal1(matrix_dvp_1,matrix_dvp_2,tM,np.ascontiguousarray(dx0[2,:]).reshape(1,-1))
            dx2 = np.linalg.solve(Cx1,dx1)
        UN = dx2[1,:].real
        TN = dx2[2,:].real * CONST_LAMBDA
        Lt = UN
        return Lt 

# %%
def DVP_S0_tls(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_A,CONST_B,CONST_C,CONST_GR,CONST_LAMBDA):
    # Tide, loading, shear love numbers when degree == 0
    ppgC = DVP_S0_core(core_data)
    matrix_dvp_1,matrix_dvp_2,matrix_dvp_3,ID_SF = DVP_S_cmd2surface(mantle_data,diff_eq4,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,m_nH=1,matrix_dim=4)
    matrix_dvp_g = dvp_propagator(dvp_propagator(matrix_dvp_1,matrix_dvp_2),matrix_dvp_3)
    b0 = inner_core_sol_0(core_data)
    tc = (ppgC @ b0).astype(np.complex128)
    tM = np.zeros((4,1),dtype=np.complex128)
    tM[0] = tc[0] / CONST_R
    tM[1] = tc[1] / CONST_R / CONST_LAMBDA
    tM[2] = tc[2] / CONST_GR / CONST_R
    tM[3] = tM[2] 
    x = matrix_dvp_g[1,1] * tM[1] + matrix_dvp_g[1,0] * tM[0]
    c = -CONST_A / x # np.linalg.solve(x.T,-CONST_A.T).T
    Hs = c * (matrix_dvp_g[0,1] * tM[1] + matrix_dvp_g[0,0] * tM[0]) # only loading has conponent. Just one element. no need to do matrix multiplication
    return Hs

def DVP_S_tls(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_A,CONST_B,CONST_C,CONST_RHO,CONST_GR,m_nH):
    Hs = np.zeros((3,))
    Ls = np.zeros((3,))
    Ks = np.zeros((3,))
    ppgC = DVP_S_core(core_data,m_nH)
    matrix_dvp_1,matrix_dvp_2,matrix_dvp_3,ID_SF = DVP_S_cmd2surface(mantle_data,diff_eq6,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,m_nH,matrix_dim=6)
        
    matrix_dvp_g = dvp_propagator(dvp_propagator(matrix_dvp_1,matrix_dvp_2),matrix_dvp_3)
    tM = DVP_S_core_mantle_boundary(core_data,ppgC,m_nH,CONST_B,CONST_C,CONST_RHO,CONST_GR)
                # Tide / load / shear
    Ta = np.zeros((3, 3),dtype=np.complex128)
    Ta[2, 0] = -(2 * m_nH + 1)
    Ta[0, 1] = -(2 * m_nH + 1) * CONST_A / m_nH
    Ta[2, 1] = -(2 * m_nH + 1)
        
    if m_nH == 1:
        phi_r = -1
        p11 = matrix_dvp_g[:3,:3]
        p12 = matrix_dvp_g[:3,3:]
        p21 = matrix_dvp_g[3:,:3]
        p22 = matrix_dvp_g[3:,3:]
        t0 = tM[3:,:2] - p21 @ tM[:3,:2]
        Ta[0, 2] = -3 * CONST_A
        Ta[1, 2] = -Ta[0, 2] / 2
        rigid = np.array([[1.0], [1.0], [-1.0]],dtype=np.complex128)
        t1 = p22 @ Ta[:,1:3]
        c1 = np.zeros((3,2),dtype=np.complex128)
        c1[:2,:] = np.linalg.solve(t0[1:3,:], t1[1:,:])
        c2 = p11 @ tM[:3, :3] @ c1 + p12 @ Ta[:, 1:]
        cc = (phi_r - c2[2]) / rigid[2]
        c3 = c2 + rigid * cc

        Hs[1] = c3[0,0].real # it seems like c3[:3,0] is real in matlab version. For computation efficiency with numba, I make all complex values
        Ls[1] = c3[1,0].real
        Ks[1] = -c3[2,0].real - 1
        Hs[2] = c3[0,1].real
        Ls[2] = c3[1,1].real
        Ks[2] = (-c3[2,1]).real
    else:
        Ta[1, 2] = (2 * m_nH + 1) * CONST_A / (m_nH * (m_nH + 1))
        if m_nH <= 100:
            t0 = tM[3:,:] - matrix_dvp_g[3:,:3] @ tM[:3,:]
            t1 = np.linalg.solve((t0 @ matrix_dvp_g[3:,3:]).T,(matrix_dvp_g[:3,:3] @ tM[:3:]).T).T + matrix_dvp_g[:3,3:]
            u = t1 @ Ta
        else:
            u = matrix_dvp_g[:3,3:] @ Ta
        Hs = u[0,:].real
        Ls = u[1,:].real
        Ks[:2] = -u[2,:2].real
        Ks[2] = -u[2,2].real - m_nH
    return Hs, Ls, Ks

# %%
#@njit
def cal_love_tls(core_data,mantle_data,depth_s, depth_f, degree_b, degree_e, degree_i,CONST_RHO,CONST_LAMBDA,CONST_GR,CONST_A,CONST_B,CONST_C):
    m_source_layer = find_layer(mantle_data[:,0],depth_s)
    m_field_layer = find_layer(mantle_data[:,0],depth_f)
    if degree_b == 0:
        Hs = DVP_S0_tls(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_A,CONST_B,CONST_C,CONST_GR,CONST_LAMBDA)
        iter_start = degree_b + degree_i
        H22 = Hs # not sure whether it is H22
        #H33 = Hs[3]
    else:
        iter_start = degree_b
        H22 = None
        #H33 = None
            
    if degree_i == 1:
        degree_list = np.arange(iter_start,degree_e+degree_i,degree_i,dtype=np.int64)
    else:
        degree_list = np.arange(iter_start,degree_e,degree_i,dtype=np.int64)
    Hs_total = np.zeros(((degree_list.shape[0],3)))
    Ls_total = np.zeros(((degree_list.shape[0],3)))
    Ks_total = np.zeros(((degree_list.shape[0],3)))
        
    for i in range(degree_list.shape[0]):
        Hs, Ls, Ks = DVP_S_tls(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_A,CONST_B,CONST_C,CONST_RHO,CONST_GR, degree_list[i])

        Hs_total[i] = Hs
        Ls_total[i] = Ls
        Ks_total[i] = Ks
    return Hs_total, Ls_total, Ks_total, H22, degree_list

# %%
@njit
def cal_love_dislocation(core_data,mantle_data,depth_s, depth_f, degree_b, degree_e, degree_i,CONST_RHO,CONST_LAMBDA,CONST_GR,CONST_A,CONST_B,CONST_C):
    m_source_layer = find_layer(mantle_data[:,0],depth_s)
    m_field_layer = find_layer(mantle_data[:,0],depth_f)
    if degree_b == 0:
        Hs = DVP_S0_dislocation(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_GR,CONST_LAMBDA)
        iter_start = degree_b + degree_i
        H22 = Hs[2]
        H33 = Hs[3]
    else:
        iter_start = degree_b
        H22 = None
        H33 = None
    if degree_i == 1:
        degree_list = np.arange(iter_start,degree_e+degree_i,degree_i,dtype=np.int64)
    else:
        degree_list = np.arange(iter_start,degree_e,degree_i,dtype=np.int64)
    Hs_total = np.zeros(((degree_list.shape[0],4)))
    Ls_total = np.zeros(((degree_list.shape[0],4)))
    Ks_total = np.zeros(((degree_list.shape[0],4)))
    Lt_total = np.zeros(((degree_list.shape[0],2)))
        
    for i in range(degree_list.shape[0]):
        Hs, Ls, Ks = DVP_S_dislocation(core_data,mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_RHO,CONST_GR, degree_list[i])
        Lt = DVP_T(mantle_data,depth_s,depth_f,m_source_layer,m_field_layer,CONST_B,CONST_C,CONST_LAMBDA,degree_list[i])
        Hs_total[i] = Hs
        Ls_total[i] = Ls
        Ks_total[i] = Ks
        Lt_total[i] = Lt
    return Hs_total, Ls_total, Ks_total,Lt_total, H22,H33, degree_list

# %%
class Read_models:
    def __init__(self,core_file,mantle_file,normalized=True):
        try:
            file = np.loadtxt(core_file,skiprows=1)[:,1:]
            self.core_data = np.zeros((file.shape[0], 5))
            self.core_data[:,:4] = file
            self.core_file = os.path.basename(core_file)
        except FileNotFoundError:
            raise FileNotFoundError("Cannot open CoreFile!")
        try:
            file = np.loadtxt(mantle_file,skiprows=1)[:,1:]
            self.mantle_data = np.zeros((file.shape[0], 8))
            self.mantle_file = os.path.basename(mantle_file)
            if file.shape[1] == 4:  # isotropic
                miu = file[:,2]
                Lambda = file[:,3]
                x = Lambda + 2.0 * miu
                self.mantle_data[:, 0] = file[:,0]
                self.mantle_data[:, 1] = file[:,1]
                self.mantle_data[:, 2] = x
                self.mantle_data[:, 3] = x
                self.mantle_data[:, 4] = miu
                self.mantle_data[:, 5] = miu
                self.mantle_data[:, 6] = Lambda
            else:  # transversely isotropic
                self.mantle_data[:, :7] = file 
        except FileNotFoundError:
            raise FileNotFoundError("Cannot open MantleFile for Earth mantle!")
      
        self.core_data[:,4],self.mantle_data[:,7] = self.cal_gravity()
        self.CONST_RHO = self.core_data[0, 1]
        self.CONST_LAMBDA = self.core_data[0, 2]
        self.CONST_GR = self.mantle_data[-1, 7]  # last layer gR
        self.CONST_A = self.CONST_GR**2 / (4 * np.pi * CONST_G * self.CONST_LAMBDA)
        self.CONST_B = 4 * np.pi * CONST_G * self.CONST_RHO * CONST_R / self.CONST_GR
        self.CONST_C = self.CONST_RHO * self.CONST_GR * CONST_R / self.CONST_LAMBDA
        
        self.normalized = normalized
        if normalized:
            self.core_normalize()
            self.mantle_normalize()
            
    def cal_gravity(self):
    # Core gravity calculation
        v0 = 0
        mass = 0
        gravity_core = np.zeros(self.core_data.shape[0],)
        gravity_mantle = np.zeros(self.mantle_data.shape[0],)
        for i in range(gravity_core.shape[0]):
            r = self.core_data[i, 0]
            v1 = 4 * np.pi * r**3 / 3
            mass += (v1 - v0) * self.core_data[i, 1]
            gravity_core[i] = CONST_G * mass / r**3
            v0 = v1
    # Mantle gravity
        self.mantle_data[0, 7] = self.core_data[-1, 4] * self.core_data[-1, 0]
        for i in range(1, gravity_mantle.shape[0]):
            r = self.mantle_data[i, 0]
            v1 = 4 * np.pi * r**3 / 3
            mass += (v1 - v0) * self.mantle_data[i, 1]
            gravity_mantle[i] = CONST_G * mass / r**2
            v0 = v1
        return gravity_core, gravity_mantle
    
    def core_normalize(self):
        self.core_data[:,0] = self.core_data[:,0] / CONST_R

    def mantle_normalize(self):
        self.mantle_data[:,0] = self.mantle_data[:,0] / CONST_R
        self.mantle_data[:,1] = self.mantle_data[:,1] / self.CONST_RHO
        self.mantle_data[:,2:-1] = self.mantle_data[:,2:-1] / self.CONST_LAMBDA
        self.mantle_data[:,-1] = self.mantle_data[:,-1] / self.CONST_GR

# %%

class DLN:
    """
        Container to compute love numbers
    """
    def __init__(self, CM_models):
        """
        Parameters:
            CM_models: models of core and mantle. should be generated by Read_models()
        """
        self.core_data = CM_models.core_data
        self.mantle_data = CM_models.mantle_data
        self.core_file = CM_models.core_file
        self.mantle_file = CM_models.mantle_file
        self.CONST_RHO = CM_models.CONST_RHO
        self.CONST_LAMBDA = CM_models.CONST_LAMBDA
        self.CONST_GR = CM_models.CONST_GR
        self.CONST_A = CM_models.CONST_A
        self.CONST_B = CM_models.CONST_B
        self.CONST_C = CM_models.CONST_C
    
    def loves_tls(self, depth_s, depth_f, degree_b=0, degree_e=1000, degree_i=1):
        """IMPORTANT: I HAVE NOT YET VALIDATED THIS FUNCTION!!!
        Function to compute love number for tide, load and shear.
        
        
        Parameters:
            depth_s : float. Depth of source in km
            depth_f : float. Depth of source in km
            degree_b: integer. lowest degree
            degree_e: integer. highest degree
            degree_i: integer. interval of degree
        
        return:
            loves: love numbers about Hs, Ls, Ks, H22 and a list of degree for these
        
        """ 
        self.depth_s = depth_s # in km
        self.depth_f = depth_f  # in km
        self.degree_b = degree_b # lowest harmonic degree
        self.degree_e = degree_e #@ highest harmonic degree
        self.degree_i = degree_i # degree increment
        loves = {'Hs':None,'Ls':None,'Ks':None,'H22':None}
        loves['Hs'], loves['Ls'], loves['Ks'],loves['H22'], loves['degree_list'] = cal_love_tls(self.core_data,self.mantle_data,depth_s * 1000, depth_f * 1000, degree_b, degree_e, degree_i,self.CONST_RHO,self.CONST_LAMBDA,self.CONST_GR,self.CONST_A,self.CONST_B,self.CONST_C)
        return loves
    
    def loves_dislocation(self, depth_s, depth_f, degree_b=0, degree_e=1000, degree_i=1):
        """
        Function to compute love number for dislocation.
        
        Parameters:
            depth_s : float. Depth of source in km
            depth_f : float. Depth of source in km
            degree_b: integer. lowest degree
            degree_e: integer. highest degree
            degree_i: integer. interval of degree
        
        return:
            loves: love numbers about Hs, Ls, Ks, Lt, H22, H33 and a list of degree for these
        
        """
        self.depth_s = depth_s # in km
        self.depth_f = depth_f  # in km
        self.degree_b = degree_b # lowest harmonic degree
        self.degree_e = degree_e #@ highest harmonic degree
        self.degree_i = degree_i # degree increment
        loves = {'Hs':None,'Ls':None,'Ks':None,'Lt':None,'H22':None,'H33':None,'degree_list':None}
        loves['Hs'], loves['Ls'], loves['Ks'], loves['Lt'],loves['H22'], loves['H33'], loves['degree_list'] = cal_love_dislocation(self.core_data,self.mantle_data,depth_s * 1000, depth_f * 1000, degree_b, degree_e, degree_i,self.CONST_RHO,self.CONST_LAMBDA,self.CONST_GR,self.CONST_A,self.CONST_B,self.CONST_C)
        return loves


