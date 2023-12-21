from casadi import *
import numpy as np
def main():
     # ユーザーが設定する制約条件
    M=10
    max_force = 100 
    max_power = 100
    max_cf = 100
    
    # 状態変数の設定
    X =SX.sym('X')
    Y =SX.sym('Y')
    theta =SX.sym('theta')
    V =SX.sym('V')
    beta =SX.sym('beta')
    theta_dot =SX.sym('theta_dot')
    beta_dot =SX.sym('beta_dot')
    time =SX.sym('time')
    # 状態ベクトルに統合
    x = vertcat(X,Y,theta,V,beta,theta_dot,beta_dot,time)
    # 入力変数の設定
    Uw =SX.sym('Uw')
    Us =SX.sym('Us')
    Utheta =SX.sym('Utheta')
    # 入力ベクトルに統合
    u = vertcat(Uw,Us,Utheta)
    # 微分方程式の設定
    x_dot = vertcat(V*np.cos(beta),V*np.sin(beta),theta_dot,Uw/M,beta_dot,Utheta,Us,1)
    
    
    # 目的関数の設定(時間を最小化)
    obj = time
    
    # 制約条件の設定
    Cf = M*V*beta_dot
    Power = M*V*Uw

    
    # 制約条件の統合
    g = vertcat(Cf,Power,max_force,max_power,max_cf)
    
    # 最適問題の設定
    nlp = {'x':x,'f':obj,'g':g}
    opts = {"ipopt.linear_solver":"ma27"} # ソルバーの設定
    solver = nlpsol('solver','ipopt',nlp,opts) # 最適化問題の設定
    
    sol = solver (x0=[0,0,0,0,0,0,0],lbx=[-inf,-inf,-inf,-inf,-inf,-inf,-inf],ubx=[inf,inf,inf,inf,inf,inf,inf],lbg=[-inf,-inf,-inf,-inf,-inf],ubg=[inf,inf,inf,inf,inf]) # ソルバーの設定
    
    print('x_opt: ',sol['x'])
if __name__ == '__main__':
    main()