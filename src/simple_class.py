# TODO やりたいこと


from casadi import *
import numpy

# x_dot = V*cos(beta)
# y_dot = V*sin(beta)

class TrajectoryCreator:
    def __init__(self):
        self.state = []
        self.N = 100      # ホライゾン離散化グリッド数 #TODO 経過点を増やした場合の合計にするか悩み中
        self.opti = casadi.Opti()
        self.M = 0 # 機体の質量[kg]
        self.sol = None
        self.X = 0
        self.u = 0
        self.z = 0
        self.p = 0 
        self.T = 0
        self.dt = 0
    
    def _set_variable(self):
        nx = 3      # 状態空間の次元
        nu = 1      # 制御入力の次元
        self.X = self.opti.variable(nx,self.N*(len(self.state)-1)) # ほしいのは間の数なので-1
        self.u = self.opti.variable(nu,self.N*(len(self.state)-1))
        self.T = self.opti.variable()
        self.dt = self.T/(self.N*(len(self.state)-1))
    
    def _add_point(self,x,y):
        self.state.append([x,y])
        # 指定したポイントの数だけfor分を回す
    def _problem(self,num):
        for i in range((self.N-1)*num,(self.N-1)*(num+1)):
            
            x = self.X[0,i]
            y = self.X[1,i]
            beta = self.X[2,i]
            V = self.u[0,i]
        
            self.opti.subject_to(self.dt>=0)
            self.opti.subject_to(self.X[0,i+1]==x+ self.dt*V*cos(beta))
            self.opti.subject_to(self.X[1,i+1]==y + self.dt*V*sin(beta))
            
            self.opti.subject_to(V>=0)
            self.opti.subject_to(V<=2.0)
    def _solve(self):
        self._set_variable()
        
        # 始点、経過点、終点の設定
        for i in range(len(self.state)-1): #終端に達したら終わりなので-1
            self._problem(i)
            
            # ここ（k要素）は悪くない
            for k in range(len(self.state[i])):
                if self.state[i][k] !=None:
                    self.opti.subject_to(self.X[k,self.N*i]==self.state[i][k]) # 初期状態の設定
                if self.state[i+1][k] !=None:
                    self.opti.subject_to(self.X[k,self.N*(i+1)-1]==self.state[i+1][k]) # 終端状態の設定
        
            
        self.opti.minimize(self.T)
        self.opti.solver('ipopt')
        # print(self.opti.debug.value(self.T))
        self.sol = self.opti.solve()
        
    def start_point(self,x,y):
        self._add_point(x,y)
    
    def pass_point(self,x,y):
        self._add_point(x,y)
    
    def end_point(self,x,y):
        self._add_point(x,y)
        self._solve()

    
    
    def print_sol(self):
        x_opt = self.sol.value(self.X) # 結果の取得
        u_opt = self.sol.value(self.u) # 結果の取得
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
        plt.plot(dt_opt, x_opt[2,:],'-')
        plt.xlabel('Time [s]')
        plt.ylabel('beta [rad]')
        plt.grid()

        plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity over Time')
        plt.plot(dt_opt, u_opt[0:],'-')
        plt.grid()

    

        plt.show()
    
    
if __name__ == '__main__':
    trajectory_creator = TrajectoryCreator()
    trajectory_creator.start_point(0,0)
    trajectory_creator.pass_point(1,1.5)
    # trajectory_creator.pass_point(2,2,0,0,0,0,0,0)
    trajectory_creator.end_point(2,2)
    trajectory_creator.print_sol()