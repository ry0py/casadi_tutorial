from casadi import *
import numpy as np
def main():
    # 状態変数の設定
    X =SX.sym('X')
    Y =SX.sym('Y')
    theta =SX.sym('theta')
    V =SX.sym('V')
    beta =SX.sym('beta')
    theta_dot =SX.sym('theta_dot')
    beta_dot =SX.sym('beta_dot')
    # 状態ベクトルに統合
    x = vertcat(X,Y,theta,V,beta,theta_dot,beta_dot)
    # 入力変数の設定
    theta_ddot =SX.sym('theta_ddot')
    beta_ddot =SX.sym('beta_ddot')
    # 入力ベクトルに統合
    u = vertcat(theta_ddot,beta_ddot)
    # 微分方程式の設定
    x_dot = vertcat(V*np.cos(beta),V*np.sin(beta),theta_dot,theta_ddot,beta_dot,beta_ddot)
    
    
    # 目的関数の設定(時間を最小化)
    cost_time = SX.sym('cost_time') # TODO SXだとエラーが出る
    obj = cost_time
    
    # 制約条件の設定
    steer_acc_constraint = 1
    steer_speed_constraint = 1
    # ユーザーが設定する
    max_force = 100 
    max_power = 100
    max_cf = 100
    
    # 制約条件の統合
    g = vertcat(steer_acc_constraint,steer_speed_constraint,max_force,max_power,max_cf)
    
    # 最適問題の設定
    nlp = {'x':x,'f':obj,'g':g}
    opts = {"ipopt.linear_solver":"ma27"} # ソルバーの設定
    solver = nlpsol('solver','ipopt',nlp,opts) # 最適化問題の設定
    
    sol = solver (x0=[0,0,0,0,0,0,0],lbx=[-inf,-inf,-inf,-inf,-inf,-inf,-inf],ubx=[inf,inf,inf,inf,inf,inf,inf],lbg=[-inf,-inf,-inf,-inf,-inf],ubg=[inf,inf,inf,inf,inf]) # ソルバーの設定
    
    print('x_opt: ',sol['x'])
if __name__ == '__main__':
    main()