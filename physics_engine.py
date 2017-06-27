from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import time
from math import sin, cos, radians, pi

# 1000 one-millisecond time steps
total_state=1000;
# 5 features on the state [mass,x,y,x_vel,y_vel]
fea_num=5;
# G 
#G = 6.67428e-11;
G=10;
# time step
diff_t=0.000001;

def init(total_state,n_body,fea_num,orbit):
  data=np.zeros((total_state,n_body,fea_num),dtype=float);
  if(orbit):
    print("Not yet");exit(1);
  else:
    for i in range(n_body):
      data[0][i][0]=np.random.rand()*8.98+0.02;
      distance=np.random.rand()*90.0+10.0;
      theta=np.random.rand()*360;
      theta_rad = pi/2 - radians(theta);    
      data[0][i][1]=distance*cos(theta_rad);
      data[0][i][2]=distance*sin(theta_rad);
      data[0][i][3]=np.random.rand()*6.0-3.0;
      data[0][i][4]=np.random.rand()*6.0-3.0;
  return data;      

def get_f(reciever,sender):
  diff=reciever[1:3]-sender[1:3];
  return G*reciever[0]*sender[0]*diff/(np.linalg.norm(diff)**3);
 
def calc(cur_state,n_body):
  next_state=np.zeros((n_body,fea_num),dtype=float);
  f_mat=np.zeros((n_body,n_body,2),dtype=float);
  f_sum=np.zeros((n_body,2),dtype=float);
  acc=np.zeros((n_body,2),dtype=float);
  for i in range(n_body):
    for j in range(i+1,n_body):
      if(j!=i):
        f_mat[i,j]+=get_f(cur_state[i][:3],cur_state[j][:3]);  
    f_mat[j,i]-=f_mat[i,j];  
    f_sum[i]=np.sum(f_mat[i,:]); 
    acc[i]=f_sum[i]/cur_state[i][0];
    next_state[i][0]=cur_state[i][0];
    next_state[i][3:5]=cur_state[i][3:5]+acc[i]*diff_t;
    next_state[i][1:3]=cur_state[i][1:3]+next_state[i][3:5]*diff_t;
  return next_state;

def gen(n_body,orbit):
  # initialization on just first state
  data=init(total_state,n_body,fea_num,orbit);
  for i in range(1,total_state):
    data[i]=calc(data[i-1],n_body);
  return data;

if __name__=='__main__':
  gen(3,False);
