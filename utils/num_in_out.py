import math
import numpy as np
import torch
import time
from utils.integral import Simpson, normalpdf

def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#seed_everything(9)

NormalPDF = normalpdf()

#Integral of a normal function cdfd(a,b,u,sigma)
def pdf(x):
  return math.exp(-(x) ** 2 / (2)) / (math.sqrt(2 * math.pi))
 
def sum_fun_xk(xk, func):
  return sum([func(each) for each in xk])
 
def integral(a, b, n, func):
  h = (b - a)/float(n)
  xk = [a + i*h for i in range(1, n)]
  return h/2 * (func(a) + 2 * sum_fun_xk(xk, func) + func(b))
 
def cdfd(a,b,u,o):
  return integral((a-u)/o,(b-u)/o,10000,pdf)

# def get_data(m_in_batch,u_in_batch,sigma_in_batch,f_in_batch):

#     M = 80 #60-120
#     #calculate input and output
#     tao = 1
#     dt = 0.1 #real phy time speed is um/s
#     #m_in = np.random.rand(1)*3 #mN/um in range 3
#     #u_in = 4
#     #sigma_in = 20
#     #f_in = np.random.rand(1)*100 #-100-100uN
#     batch_size = m_in_batch.shape[0]
#     m_out_batch = np.zeros_like(m_in_batch)
#     u_out_batch = np.zeros_like(u_in_batch)
#     for i in range(batch_size):
#       m_in = m_in_batch[i]
#       u_in = u_in_batch[i]
#       sigma_in = sigma_in_batch[i]
#       f_in = f_in_batch[i]
#       if f_in > 0 or f_in < 0 or m_in == 0:
#         sigma0 = 10
#         m_dot_max = 10 #Maximum acceleration
#         if sigma_in == 0:
#           sigma_in = np.random.rand()
#         temp_gamma = cdfd(-sigma0,sigma0,0,sigma_in)/cdfd(-sigma0,sigma0,0,sigma0)

#         gamma = min(temp_gamma,1)

#         p1 = -M * m_in/(2*(f_in+gamma*u_in))

#         k = m_in/(2*p1)

#         temp_m_dot = (4*p1*p1-2)*k

#         m_dot = min(temp_m_dot,m_dot_max)
#         m_dot = max(temp_m_dot,-m_dot_max)

#         m_out = m_dot * dt + m_in

#         u_out = f_in - M * m_out * dt

    
#         if np.isnan(m_out) or np.isnan(u_out):
#           print("data error")
#       else:
#         m_out = 0.5 * m_in
#         u_out = f_in - M * m_out * dt
      
#       m_out_batch[i] = m_out
#       u_out_batch[i] = u_out

#     return m_out_batch,u_out_batch

class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()
    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 20.0
        self.output = 0.0
    def update(self, feedback_value):
        error = self.SetPoint - feedback_value
        #print(f'NextState:{self.SetPoint}||State:{feedback_value}||error:{error}')
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error#比例
            self.ITerm += error * delta_time#积分
            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time
            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
        return self.output
    def setKp(self, proportional_gain):
        self.Kp = proportional_gain
    def setKi(self, integral_gain):
        self.Ki = integral_gain
    def setKd(self, derivative_gain):
        self.Kd = derivative_gain
    def setWindup(self, windup):
        self.windup_guard = windup
    def setSampleTime(self, sample_time):
        self.sample_time = sample_time

pid1 = PID(P=0.0008,I=0.000,D=0.000)
pid1.clear()
pid1.setSampleTime(0.1)
pid1.SetPoint = 0

def get_data(m_in,u_in,sigma_in,f_in):

  M = 80 #60-120
  #calculate input and output
  tao = 1
  dt = 0.1 #real phy time speed is um/s
  #m_in = np.random.rand(1)*3 #mN/um in range 3
  #u_in = 4
  #sigma_in = 20
  #f_in = np.random.rand(1)*100 #-100-100uN

  if abs(f_in) >60 and m_in != 0:
    sigma0 = 10
    m_dot_max = 10 #Maximum acceleration
    if sigma_in == 0:
      sigma_in = np.random.rand()
    temp_gamma = cdfd(-sigma0,sigma0,0,sigma_in)/cdfd(-sigma0,sigma0,0,sigma0)

    gamma = min(temp_gamma,1)

    m_out = m_in + pid1.update(-(f_in+gamma*u_in))

    u_out = f_in - M * m_out * dt
  else:
    m_out = m_in + pid1.update(-(f_in))

    u_out = f_in - M * m_out * dt

  return m_out,u_out


def get_data_prefrontal(f_xt,f_yt,u_x,u_y,sigma_x,sigma_y,m_dot_z,belta,lenth,x_sum,v_max,a_max):
    rca = 0.14
    dt = 0.1
    m_dot_zt_true = 0

    if x_sum > (lenth-(v_max*v_max)/(2*a_max)):
      # 减速阶段
      flag = 1
      if x_sum <lenth:
        delt_v = -a_max*dt
      else:
        if m_dot_z < 0:
          delt_v = a_max*dt
        else:
          delt_v = -a_max*dt
    elif x_sum < (v_max*v_max)/(2*a_max):
      # 加速阶段
      flag = 0
      delt_v = a_max*dt
    else:
      # 匀速阶段
      flag = 0
      if m_dot_z <= v_max:
        delt_v = (10*(1-m_dot_z/v_max) - (abs(f_xt)+abs(f_yt))/1000) * a_max * dt
      else:
        delt_v = 0

    m_dot_zt_true = m_dot_z + delt_v

    return flag, m_dot_zt_true

def get_horizontal_force(x_sum,lenth,param_x=1200,param_y=1200):
  progress_rate = x_sum/lenth

  f_xt = (np.random.rand(1)*param_x) * progress_rate 
  f_yt = (np.random.rand(1)*param_y) * progress_rate 
  return f_xt,f_yt

def feature_normalize(data,dim=0):
    mu = np.mean(data,axis=dim)
    std = np.std(data,axis=dim)
    mu = np.expand_dims(mu, dim)
    std = np.expand_dims(std, dim)
    return (data - mu)/std


if __name__ == "__main__":
  # possion_num = 50
  # dt = 0.1
  # #num = 100000
  # num = 50
  # N_step = 3
  # data =np.loadtxt("prefrontal_data.txt").reshape(-1,350,7)
  # print(data.shape)
  # data_pre_spike = np.zeros([data.shape[0],350,17,4,50])
  # data_pre = np.zeros([data.shape[0],350,17])
  # m_x = data[:,:,0]

  # m_y = data[:,:,1]
  # f_x = data[:,:,2]
  # f_y = data[:,:,3]
  # delt_z = data[:,:,4]
  # x_sum = data[:,:,5]
  # lenth = data[:,:,6]

  # opts = np.ones([data.shape[0],350,N_step,4])
  # opts[:,:,1,:] = opts[:,:,1,:]*2
  # opts[:,:,2,:] = opts[:,:,2,:]*3
  # opts[:,:,:,0] = opts[:,:,:,0]*0.7189618*delt_z.reshape(data.shape[0],350,1)
  # opts[:,:,:,1] = opts[:,:,:,1]*1.287569*delt_z.reshape(data.shape[0],350,1)
  # opts[:,:,:,2] = opts[:,:,:,2]*1.3333306*delt_z.reshape(data.shape[0],350,1)
  # opts[:,:,:,3] = opts[:,:,:,3]*0.5672474*delt_z.reshape(data.shape[0],350,1)

  # opts_ux = np.zeros([data.shape[0],350,N_step])
  # opts_uy = np.zeros([data.shape[0],350,N_step])

  # for i in range(data.shape[0]):
  #   print(i)
  #   for j in range(350):
  #     for k in range(N_step):
  #         opts_mx,opts_ux[i,j,k] = get_data(m_x[i,j],opts[i,j,k,0],opts[i,j,k,2],f_x[i,j]) #opts_mx:[-4,4] opts_ux:[-180,180]
  #         opts_my,opts_uy[i,j,k] = get_data(m_y[i,j],opts[i,j,k,1],opts[i,j,k,3],f_y[i,j]) #opts_mx:[-4,4] opts_ux:[-180,180]

  #     u_x = opts_ux[:]
  #     u_y = opts_uy[:]
  #     sigma_x = opts[:,2]
  #     sigma_y = opts[:,3]
  #   if i % 100 == 0:
  #     data_pre[:,:,0] = f_x
  #     data_pre[:,:,1] = f_y
  #     data_pre[:,:,2:5] = opts_ux
  #     data_pre[:,:,5:8] = opts_uy
  #     data_pre[:,:,8:11] = opts[:,:,:,2]
  #     data_pre[:,:,11:14] = opts[:,:,:,3]
  #     data_pre[:,:,14] = lenth
  #     data_pre[:,:,15] = x_sum
  #     data_pre[:,:,16] = delt_z
  #     np.savetxt('data_pre.npy',data_pre.reshape(-1,17))
  #     print('saved data pre successfully')

  # data_pre_spike[:,:,0,:,:] = (np.random.rand(data.shape[0],350,4,50) < (f_x.reshape(data.shape[0],350,1,1))/200).astype(np.int)
  # data_pre_spike[:,:,1,:,:] = (np.random.rand(data.shape[0],350,4,50) < (f_y.reshape(data.shape[0],350,1,1))/200).astype(np.int)
  # data_pre_spike[:,:,2:5,:,:] = (np.random.rand(data.shape[0],350,3,4,50) < (feature_normalize(opts_ux,dim=2).reshape(data.shape[0],350,3,1,1))).astype(np.int)
  # data_pre_spike[:,:,5:8,:,:] = (np.random.rand(data.shape[0],350,3,4,50) < (feature_normalize(opts_uy,dim=2).reshape(data.shape[0],350,3,1,1))).astype(np.int)
  # data_pre_spike[:,:,8:11,:,:] = (np.random.rand(data.shape[0],350,3,4,50) < (feature_normalize(opts[:,:,:,2],dim=2).reshape(data.shape[0],350,3,1,1))).astype(np.int)
  # data_pre_spike[:,:,11:14,:,:] = (np.random.rand(data.shape[0],350,3,4,50) < (feature_normalize(opts[:,:,:,3],dim=2).reshape(data.shape[0],350,3,1,1))).astype(np.int)
  # data_pre_spike[:,:,14,:,:] = (np.random.rand(data.shape[0],350,4,50) < (feature_normalize(lenth,dim=1).reshape(data.shape[0],350,1,1))).astype(np.int)
  # data_pre_spike[:,:,15,:,:] = (np.random.rand(data.shape[0],350,4,50) < (feature_normalize(x_sum,dim=1).reshape(data.shape[0],350,1,1))).astype(np.int)
  # data_pre_spike[:,:,16,:,:] = (np.random.rand(data.shape[0],350,4,50) < (feature_normalize(delt_z,dim=1).reshape(data.shape[0],350,1,1))).astype(np.int)

  # np.savetxt('data_pre_spike.npy',data_pre_spike.reshape(-1,50))

  ###########test###############
  # m_dot_z = 100
  # belta = 0.5
  # dt = 0.1
  # lenth = 1000
  # x_sum = 500
  # delt_z = m_dot_z
  
  # for i in range(1000):
  #   f_xt = np.random.rand(1)+i
  #   f_yt = np.random.rand(1)+i
  #   delt_z = np.array(delt_z)
  #   opts = np.array([0.7189618*delt_z, 1.287569*delt_z, 1.3333306*delt_z, 0.5672474*delt_z])
  #   opts = np.array([opts*(i+1) for i in range(3)])

  #   opts_ux = opts_uy = np.zeros_like(opts)
  #   m_x = np.random.rand(1)*8 - 4 # [-4,4]
  #   m_y = np.random.rand(1)*8 - 4 # [-4,4]
  #   for k in range(3):
  #       opts_mx,opts_ux[k,0] = get_data(m_x,opts[k,0],opts[k,2],f_xt) #opts_mx:[-4,4] opts_ux:[-180,180]
  #       opts_my,opts_uy[k,0] = get_data(m_y,opts[k,1],opts[k,3],f_yt) #opts_mx:[-4,4] opts_ux:[-180,180]
  #       m_x = opts_mx
  #       m_y = opts_my

  #   u_x = opts_ux[:,0]
  #   u_y = opts_uy[:,1]
  #   sigma_x = opts[:,2]
  #   sigma_y = opts[:,3]
    
  #   flag, m_dot_zt_true = get_data_prefrontal(f_xt,f_yt,u_x,u_y,sigma_x,sigma_y,m_dot_z,belta,lenth,x_sum,100,20)
  #   print(f_xt,f_yt,m_dot_zt_true)

  m_z = 0
  m_dot_z = 0.1 #[0,80]
  belta = 0.5
  dt = 0.1
  lenth = 1000
  x_sum = 0
  delt_z = 1
  x_sum_store = []
  m_dot_z_store = []
  f_x_store = []
  f_y_store = []

  data_cere = np.zeros([10000,25,6])

  for i in range(10000):
      f_x_max = np.random.rand(1)*200 + 100
      f_y_max = np.random.rand(1)*200 + 100
      f_xt = f_x_max

      x_xt = f_xt/80

      delt_z = np.random.rand(1) * 50
      delt_z = np.array(delt_z)
      opts = np.array([0.7189618*delt_z, 1.287569*delt_z, 1.3333306*delt_z, 0.5672474*delt_z])

      opts_ux = np.zeros_like(opts)
      m_x = np.random.rand(1)*8 - 4 # [-4,4]
      m_y = np.random.rand(1)*8 - 4 # [-4,4]

      opts_mx,opts_ux = get_data(m_x,opts[0],opts[2],f_xt) #opts_mx:[-4,4] opts_ux:[-180,180]
      u_x = opts[0]
      sigma_x = opts[2]

      for j in range(25):
        data_cere[i,j,0] = m_x
        data_cere[i,j,1] = f_xt
        data_cere[i,j,2] = u_x
        data_cere[i,j,3] = sigma_x
        data_cere[i,j,4] = opts_mx
        data_cere[i,j,5] = opts_ux

        x_xt = x_xt - m_x * dt
        f_xt = x_xt * 80
        m_x = opts_mx

        delt_z = np.random.rand(1) * 50
        delt_z = np.array(delt_z)
        opts = np.array([0.7189618*delt_z, 1.287569*delt_z, 1.3333306*delt_z, 0.5672474*delt_z])

        opts_mx,opts_ux = get_data(m_x,opts[0],opts[2],f_xt) #opts_mx:[-4,4] opts_ux:[-180,180]

        u_x = opts[0]
        sigma_x = opts[2]

        opts_mx,opts_ux = get_data(m_x,opts[0],opts[2],f_xt) #opts_mx:[-4,4] opts_ux:[-180,180]
        if m_x>100:
          a = m_x
      if i % 100 == 1:
        print('ep: ',i)
        np.save('data_cere_withux.npy',data_cere)






