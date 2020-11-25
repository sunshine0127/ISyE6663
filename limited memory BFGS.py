# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:57:37 2020

@author: Lenovo
"""

import numpy as np
import scipy.optimize

def f_1(x):
    ndim_1=x.shape[0];
    f1_value=0;
    for i in range(int(ndim_1/2)):
        f1_value=f1_value+(10*(x[2*i+1]-x[2*i]**2))**2+(1-x[2*i])**2
    return(f1_value)
    
def f_2(x):
    ndim_2=x.shape[0];
    f2_value=0;
    for i in range(int(ndim_2/4)):
        f2_value=f2_value+(x[4*i]+10*x[4*i+1])**2+5*(x[4*i+2]-x[4*i+3])**2+(x[4*i+1]-2*x[4*i+2])**4+10*(x[4*i]-x[4*i+3])**4
    return(f2_value)
        
def deri_f_1(x):
    ndim_1=x.shape[0];
    df1_value=np.zeros(ndim_1);
    for i in range(int(ndim_1/2)):
        df1_value[2*i]=2*x[2*i]-2+400*(x[2*i])**3-400*x[2*i]*x[2*i+1]
        df1_value[2*i+1]=200*(x[2*i+1]-x[2*i]**2)
    return(df1_value)
    
    
def deri_f_2(x):
    ndim_2=x.shape[0];
    df2_value=np.zeros(ndim_2);
    for i in range(int(ndim_2/4)):
        df2_value[4*i]=2*x[4*i]+20*x[4*i+1]+40*(x[4*i]-x[4*i+3])**3
        df2_value[4*i+1]=20*x[4*i]+200*x[4*i+1]+4*(x[4*i+1]-2*x[4*i+2])**3
        df2_value[4*i+2]=10*(x[4*i+2]-x[4*i+3])+8*(2*x[4*i+2]-x[4*i+1])**3
        df2_value[4*i+3]=10*(x[4*i+3]-x[4*i+2])+40*(x[4*i+3]-x[4*i])**3
    return(df2_value)
    
#L_BFGS
#L-BFGS for Problem 1
n=1600
x_0=np.zeros(n)
#initialized
for i in range(int(n/2)):
    x_0[2*i]=-1.2
    x_0[2*i+1]=1   
s=1
b=0.5
m=10
sigma=0.5
f_0=f_1(x_0)
x_temp=x_0
p_list=[]
q_list=[]
t=sum(deri_f_1(x_0)**2)
for i in range(m):
    x_last=x_temp+0
    deri_temp=deri_f_1(x_temp);
    alpha_temp=s
    phi_temp=f_1(x_temp-s*deri_temp)-f_1(x_temp)
    while phi_temp > -0.5*alpha_temp*sum(deri_temp**2):
        alpha_temp=b*alpha_temp
        phi_temp=f_1(x_temp-alpha_temp*deri_temp)-f_1(x_temp)
    x_temp=x_temp-alpha_temp*deri_temp
    p_list.append(x_temp-x_last)
    q_list.append(deri_f_1(x_temp)-deri_f_1(x_last))
    
while sum(deri_f_1(x_temp)**2)>= 1e-6*(t+1):
    x_last=x_temp+0
    rho_temp=np.zeros(m)
    for i in range(m):
        rho_temp[i]=1/sum(p_list[i]*q_list[i])
    u=deri_f_1(x_temp)
    for i in range(m):
        alpha=rho_temp[i]*sum(u*p_list[i])
        u=u-alpha*q_list[i]
    gamma=sum(p_list[m-1]*q_list[m-1])/sum(q_list[m-1]*q_list[m-1])
    r=gamma*u
    for i in range(m):
        beta_temp=rho_temp[i]*sum(r*q_list[i])
        r=r+(rho_temp[i]*sum(u*p_list[i])-beta_temp)*p_list[i]
    deri_temp=-r
    phi_temp=f_1(x_temp+s*deri_temp)-f_1(x_temp)
    alpha_temp=s
    while sum(deri_f_1(x_temp+alpha_temp*deri_temp)*deri_temp) > 0.9*sum(deri_f_1(x_temp)*deri_temp):
        alpha_temp=alpha_temp*b
        phi_temp=f_1(x_temp+alpha_temp*deri_temp)-f_1(x_temp)
    x_temp=x_temp+alpha_temp*deri_temp
    print(x_temp)
    p_list.remove(p_list[0])
    q_list.remove(q_list[0])
    p_list.append(x_temp-x_last)
    q_list.append(deri_f_1(x_temp)-deri_f_1(x_last))
    


