# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:57:15 2020

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
    
    
#DFP
#DFP for Problem 1
n=800
x_0=np.zeros(n)
#initialized
for i in range(int(n/2)):
    x_0[2*i]=-1.2
    x_0[2*i+1]=1   
s=1
b=0.8
sigma=0.5
f_0=f_1(x_0)
x_temp=x_0
H_temp=np.mat(np.identity(n))
t=sum(deri_f_1(x_0)**2)
sum1=0
while sum(deri_f_1(x_temp)**2)>= 1e-3*(t+1):
    x_last=x_temp-0
    deri_temp=np.mat(deri_f_1(x_temp))
    d_temp=np.array(-deri_temp*H_temp.T)[0]
    phi_temp=f_1(x_temp+s*d_temp)-f_1(x_temp)
    alpha_temp=s
    while sum(deri_f_1(x_temp+alpha_temp*d_temp)*d_temp) > 0.4*sum(deri_f_1(x_temp)*d_temp):
        alpha_temp=b*alpha_temp
        phi_temp=f_1(x_temp+alpha_temp*d_temp)-f_1(x_temp)
    x_temp=x_temp+alpha_temp*d_temp
    p=np.mat(x_temp-x_last)
    q=np.mat(deri_f_1(x_temp)-deri_f_1(x_last))
    p_temp=(p*q.T)[0,0]
    q_temp=(q*H_temp*q.T)[0,0]
    H_temp=H_temp+p.T*p/p_temp-H_temp*q.T*q*H_temp/q_temp
    sum1=sum1+1



#DFP for Problem 2
n=64
x_0=np.zeros(n)
#initialized
for i in range(int(n/4)):
    x_0[4*i]=3
    x_0[4*i+1]=-1
    x_0[4*i+2]=0
    x_0[4*i+3]=1

#choose s=1,beta=0.5,sigma=0.5
s=1
b=0.5
sigma=0.5
f_0=f_2(x_0)
x_temp=x_0
H_temp=np.mat(np.identity(n))
t=sum(deri_f_2(x_0)**2)
sum2=0
while sum(deri_f_2(x_temp)**2)>= 1e-6*(t+1):
    x_last=x_temp-0
    deri_temp=np.mat(deri_f_2(x_temp))
    d_temp=np.array(-deri_temp*H_temp.T)[0]
    phi_temp=f_2(x_temp+s*d_temp)-f_2(x_temp)
    alpha_temp=s
    while sum(deri_f_2(x_temp+alpha_temp*d_temp)*d_temp) > 0.3*sum(deri_f_2(x_temp)*d_temp):
        alpha_temp=b*alpha_temp
        phi_temp=f_2(x_temp+alpha_temp*d_temp)-f_2(x_temp)
    x_temp=x_temp+alpha_temp*d_temp
    p=np.mat(x_temp-x_last)
    q=np.mat(deri_f_2(x_temp)-deri_f_2(x_last))
    p_temp=(p*q.T)[0,0]
    q_temp=(q*H_temp*q.T)[0,0]
    H_temp=H_temp+p.T*p/p_temp-H_temp*q.T*q*H_temp/q_temp
    sum2=sum2+1