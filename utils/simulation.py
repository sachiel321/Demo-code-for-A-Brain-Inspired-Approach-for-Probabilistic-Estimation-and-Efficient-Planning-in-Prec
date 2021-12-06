# -*- coding: utf-8 -*-
"""
@Author: YYM
@Institute: CASIA
"""
import math
import numpy as np

#define point function
class Point:
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z
        # the original location
        self.x_o = x
        self.y_o = y
        self.z_o = z
    def getx(self):
        return self.x
    def gety(self):
        return self.y 
    def getz(self):
        return self.z 
    def reset(self):
        self.x = self.x_o
        self.y = self.y_o
        self.z = self.z_o
        self.z = self.z_o
    def move(self,delt_x=0,delt_y=0,delt_z=0):
        self.x += delt_x
        self.y += delt_y
        self.z += delt_z

#the function to get length
def getlen(p1,p2):
    x=p1.getx()-p2.getx()
    y=p1.gety()-p2.gety()
    z=p1.getz()-p2.getz()
    return math.sqrt(x**2+y**2+z**2)

#resolution ratio
ratio_circle = 50
ratio_height = 20
#model parameters
r_c = 10 #radius of cylinder
height_c = 50 #height of cylinder
bias_zc = 50 #the initial height of cylinder
r_a = 10 #radius of annulus
height_a = 50 #height of annulus
bias_za = 0 #the initial height of annulus

class simulate:
    def __init__(self, x_a, y_a, z_a, x_b, y_b, z_b,
                ra=r_c,rb=r_a,M=80,ha=height_c,hb=height_a):        
        self.pa = Point(x_a,y_a,z_a) #center of cylinder
        self.pb = Point(x_b,y_b,z_b) #center of annulus
        self.M = M
        self.ra = ra
        self.rb = rb
        self.ha = ha
        self.hb = hb
        self.f_x = 0
        self.f_y = 0
        self.isbottom = 0
        self.observation_space = 6
        self.action_space = 3
        self.info = None
    
    def step(self,action):
        
        delt_x = action[0].item()
        delt_y = action[1].item()
        delt_z = action[2].item()
        noise_x = np.clip(0.1*np.random.randn(),-0.2,0.2)
        noise_y = np.clip(0.1*np.random.randn(),-0.2,0.2)
        noise_z = np.clip(0.1*np.random.randn(),-0.2,0.2)

        self.pa.move(delt_x+noise_x,delt_y+noise_y,delt_z+noise_z)

        temp_x = self.pa.getx() - self.pb.gety()
        temp_y = self.pa.gety() - self.pb.gety()
        len_ab = getlen(self.pa,self.pb)
        sin_theta = temp_y/len_ab
        cos_theta = temp_x/len_ab
        len_in = len_ab - (self.rb-self.ra)
        f = -self.M * len_in
        self.f_x = f * cos_theta
        self.f_y = f * sin_theta

        if abs(self.pa.getz() - (self.pb.getz()))>=10: #dangerous detection
            self.isbottom = -1 - np.sqrt(self.f_x*self.f_x+self.f_y*self.f_y)*0.002

        elif abs(self.pa.getz() - (self.pb.getz()))<10: #bottom detection
            self.isbottom = 1 - np.sqrt(self.f_x*self.f_x+self.f_y*self.f_y)*0.002


        return np.array([self.f_x, self.f_y, self.pa.getz(),delt_x,delt_y,delt_z]).astype(np.float32), self.isbottom, False, self.info
    
    def reset(self):
        self.pa.reset()
        self.pb.reset()
        self.f_x = 0
        self.f_y = 0
        self.isbottom = 0
        return np.array([self.f_x, self.f_y, self.pa.getz(),0,0,0]).astype(np.float64)

def evaluate(f_x,f_y,x_z,dt=0.1):
    delt_xt = f_x * 0.01
    delt_yt = f_y * 0.01
    if x_z - 5 > 0:
        delt_zt = 5
    else:
        delt_zt = x_z
    return delt_xt, delt_yt,delt_zt


# sim = simulate(0,0,1,0,0,0)

# state = sim.step(0,0,0.3)

# print(state)