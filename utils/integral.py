#-*- coding:utf-8 -*-
import math
import numpy as np

#定义标准正态分布概率密度函数
class normalpdf():
    def __init__(self):
        self.u = 0
        self.sigma = 1
    
    def change_pram(self,u,sigma):
        self.u = u
        self.sigma = sigma

    def Normal_pdf(self,x):
        result = 1/(math.sqrt(2*math.pi)*self.sigma)*math.exp(-(x-self.u)**2/(2*self.sigma**2))
        return result

#定义复合辛普森法求积分
def Simpson(func,a,b,eps=1e-10):
    '''
    :param func: 被积函数
    :param a: 积分下限
    :param b: 积分上限
    :param eps: 计算精度，默认为1e-10
    :return: 返回积分结果
    '''

    #定义一个字典，用来存储被积函数(x,f(x))，避免重复计算
    func_result_dict = {}

    def f(x):
        if func_result_dict.get(float(x)) is None:
            r = func(float(x))
            func_result_dict[float(x)] = r
        else:
            r = func_result_dict[float(x)]
        return r

    #辛普森函数
    def Sn(a,b,n):
        '''
        :param a: 积分下限
        :param b: 积分上限
        :param n: 将区间[a,b]划为n等分
        :return: 返回积分结果
        '''
        sum_result = 0
        half_h = (b-a)/(2*n)
        for k in range(n):
            #k=0的时候，f(a+2kh)=f(a),后面需要再减去f(a)
            sum_result += 2*f(a+2*k*half_h)
            sum_result += 4 * f(a + (2 * k + 1) * half_h)
        sum_result = (sum_result+f(b)-f(a))*half_h/3
        return sum_result

    #依次计算S1,S2,S4,S8...当相邻的精度小于eps时退出循环，返回S4n的结果
    i = 1
    S2n = Sn(a,b,i)
    S4n = Sn(a,b,2*i)
    while abs(S4n-S2n) > eps:
        i += 1
        S2n = S4n
        S4n = Sn(a,b,2**i)
    return S4n