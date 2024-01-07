# TODO やりたいこと


from casadi import *
import numpy

class TrajectoryCreator:
    def __init__(self):
        self.state = []
        self.param = []
        self.N = 100      # ホライゾン離散化グリッド数 
        self.opti = casadi.Opti()
        self.M=0 # 機体の質量[kg]
        self.max_steer_acc=None # 操舵の角加速度[rad/s^2]
        self.max_steer_vel=None # 操舵の角速度[rad/s]
        self.max_steer_torque=None # 操舵のトルク[Nm]
        self.max_power=None # 機体の最大出力[W]
        self.max_cf=None # 機体の最大コーナリングフォース[N]
        self.sol = None
        self.X = None
        self.u = None
        self.z = None
        self.p = None
        self.T = None
        self.dt = None
    
    def _set_variable(self):
        nx = 8      # 状態空間の次元
        nu = 3      # 制御入力の次元
        np = 6      # パラメータの次元
        nz = 2      # 代数変数の次元
        self.X = self.opti.variable(nx,self.N*(len(self.state)-1)) # ほしいのは間の数なので-1
        self.u = self.opti.variable(nu,self.N*(len(self.state)-1))
        self.z = self.opti.variable(nz,self.N*(len(self.state)-1))
        self.p = self.opti.parameter(np) 
        self.T = self.opti.variable()
        self.dt = self.T/(self.N*(len(self.state)-1))
        self.M = self.p[0]
        self.max_steer_acc = self.p[1]
        self.max_steer_vel = self.p[2]
        self.max_steer_torque = self.p[3]
        self.max_power = self.p[4]
        self.max_cf = self.p[5]
        self.opti.set_value(self.M,self.param[0])
        self.opti.set_value(self.max_steer_acc,self.param[1])
        self.opti.set_value(self.max_steer_vel,self.param[2])
        self.opti.set_value(self.max_steer_torque,self.param[3])
        self.opti.set_value(self.max_power,self.param[4])
        self.opti.set_value(self.max_cf,self.param[5])
    
    
    def _add_point(self,x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot):
        self.state.append([x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot])
        # 指定したポイントの数だけfor分を回す
    def _problem(self,num):
        for i in range((self.N-1)*num,(self.N-1)*(num+1)+1): # self.N*(num+1)-1は差分方程式を使うために-1
            x = self.X[0,i]
            y = self.X[1,i]
            theta = self.X[2,i]
            V = self.X[3,i]
            beta = self.X[4,i]
            theta_dot = self.X[5,i]
            beta_dot = self.X[6,i]
            theta_ddot = self.X[7,i]
            
            Uw = self.u[0,i]
            Us = self.u[1,i]
            Utheta = self.u[2,i]
            
            cf = self.z[0,i]
            Power = self.z[1,i]
            
            self.opti.subject_to(self.dt>=0)
            self.opti.subject_to(self.X[0,i+1]==x+ self.dt*V*cos(beta))
            self.opti.subject_to(self.X[1,i+1]==y + self.dt*V*sin(beta))
            self.opti.subject_to(self.X[2,i+1]==theta + self.dt*theta_dot)
            self.opti.subject_to(self.X[3,i+1]==V + self.dt*Uw/self.M)
            self.opti.subject_to(self.X[4,i+1]==beta + self.dt*beta_dot)
            self.opti.subject_to(self.X[5,i+1]==theta_dot + self.dt*theta_ddot)
            self.opti.subject_to(self.X[6,i+1]==beta_dot +self.dt*Us)
            self.opti.subject_to(self.X[7,i+1]==theta_ddot + self.dt*Utheta)
            
            self.opti.subject_to(cf-self.M*V*beta_dot==0)
            self.opti.subject_to(Power-self.M*V*Uw==0)
            
            
            self.opti.subject_to(-self.max_steer_acc<=Us)
            self.opti.subject_to(Us<=self.max_steer_acc)
            self.opti.subject_to(-self.max_steer_vel<=beta_dot)
            self.opti.subject_to(beta_dot<=self.max_steer_vel)
            self.opti.subject_to(-self.max_steer_torque<=Uw)
            self.opti.subject_to(Uw<=self.max_steer_torque)
            self.opti.subject_to(-self.max_power<=Power)
            self.opti.subject_to(Power<=self.max_power)
            self.opti.subject_to(-self.max_cf<=cf)
            self.opti.subject_to(cf<=self.max_cf)
            self.opti.subject_to(-3<=Utheta)
            self.opti.subject_to(Utheta<=3)
            # self.opti.subject_to(0<=V)
        
    def _solve(self):
        self._set_variable()
        
        for i in range(len(self.state)-1): #終端に達したら終わりなので-1
            for k in range(len(self.state[i])):
                if self.state[i][k] !=None:
                    self.opti.subject_to(self.X[k,(self.N-1)*i]==self.state[i][k]) # 初期状態の設定
                if self.state[i+1][k] !=None:
                    self.opti.subject_to(self.X[k,(self.N-1)*(i+1)]==self.state[i+1][k]) # 終端状態の設定
            self._problem(i)
        
        
        
        self.opti.minimize(self.T)
        self.opti.solver('ipopt')
        # print(self.opti.debug.value(self.T))
        self.sol = self.opti.solve()
        for i in range(self.N*(len(self.state)-1)):
            print(self.X[3,i],self.sol.value(self.X[3,i]))
        
    def start_point(self,x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot):
        self._add_point(x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot)
    
    def pass_point(self,x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot):
        self._add_point(x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot)
    
    def end_point(self,x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot):
        self._add_point(x,y,theta,V,beta,theta_dot,beta_dot,theta_ddot)
        self._solve()

    
    
    def print_sol(self):
        x_opt = self.sol.value(self.X) # 結果の取得
        u_opt = self.sol.value(self.u) # 結果の取得
        z_opt = self.sol.value(self.z) # 結果の取得
        t_opt = self.sol.value(self.T) # 結果の取得
        dt_opt = numpy.linspace(0,t_opt,num =self.N*(len(self.state)-1))
        print(t_opt)

        # 結果のプロット
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x_opt[0,:],x_opt[1,:],'-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('position')
        plt.grid()
        
        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('theta [rad]')
        plt.title('theta over Time')
        plt.plot(dt_opt, x_opt[2,:],'-')
        plt.grid()
        

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity over Time')
        plt.plot(dt_opt, x_opt[3,:],'-')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Steering Angle [rad]')
        plt.title('Steering Angle over Time')
        plt.plot(dt_opt, x_opt[4,:],'-')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('theta_dot [rad/s]')
        plt.title('theta_dot over Time')
        plt.plot(dt_opt, x_opt[5,:],'-')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('beta_dot [rad/s]')
        plt.title('beta_dot over Time')
        plt.plot(dt_opt, x_opt[6,:],'-')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('cf [N]')
        plt.title('cf over Time')
        plt.plot(dt_opt, z_opt[0,:],'-')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Power [W]')
        plt.title('Power over Time')
        plt.plot(dt_opt, z_opt[1,:],'-')
        plt.grid()
        
        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Uw [Nm]')
        plt.title('Uw over Time')
        plt.plot(dt_opt, u_opt[0,:],'-')
        plt.grid()

        plt.show()

    

        
    
    def set_parameter(self,M,max_steer_acc,max_steer_vel,max_steer_torque,max_power,max_cf):
        self.param = [M,max_steer_acc,max_steer_vel,max_steer_torque,max_power,max_cf]

    
    
if __name__ == '__main__':
    trajectory_creator = TrajectoryCreator()
    trajectory_creator.set_parameter(10,10,10,10,10,10)
    trajectory_creator.start_point(0,0,0,0,0,0,0,0)
    trajectory_creator.pass_point(1,1.5,0,None,None,None,None,None)
    trajectory_creator.pass_point(2,2,0,None,None,None,None,None)
    # trajectory_creator.pass_point(2,2,0,0,0,0,0,0)
    trajectory_creator.end_point(4,4,0,0.0,0,0,0,0)
    trajectory_creator.print_sol()