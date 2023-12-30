from casadi import *
import numpy as np



# 問題設定
T = 10.0     # ホライゾン長さ
N = 1000      # ホライゾン離散化グリッド数 
dt = T / N  # 離散化ステップ TODO 固定になるがトラクリはどうしているか調べる
nx = 8      # 状態空間の次元
nu = 3      # 制御入力の次元
np = 6      # パラメータの次元
nz = 2      # 代数変数の次元
opti = casadi.Opti() # 最適化問題を宣言

X = opti.variable(nx,N)  # 状態変数の宣言 X[0]~X[7] max 799 or [1,2]のようにアクセスできる
u = opti.variable(nu,N)  # 入力変数の宣言 u[0]~u[2]
z = opti.variable(nz,N)  # パラメータ変数の宣言 z[0]~z[5]
p = opti.parameter(np) # パラメータの宣言 

# パラメータの追加
M = p[0]
max_steer_acc = p[1]
max_steer_vel = p[2]
max_steer_torque = p[3]
max_power = p[4]
max_cf = p[5]

opti.set_value(M,10)
opti.set_value(max_steer_acc,1)
opti.set_value(max_steer_vel,1)
opti.set_value(max_steer_torque,1)
opti.set_value(max_power,1)
opti.set_value(max_cf,1)

J = 0 # コスト関数の初期化
for i in range(N-1):
    J += dt  # コスト関数の定義 TODO こレって意味あるの？
    # 状態変数の定義
    x = X[0,i]
    y = X[1,i]
    theta = X[2,i]
    V = X[3,i]
    beta = X[4,i]
    theta_dot = X[5,i]
    beta_dot = X[6,i]
    theta_ddot = X[7,i]
    Uw = u[0,i] 
    Us = u[1,i]
    Utheta = u[2,i]
    cf = z[0,i]
    Power = z[1,i]
    
    
    # 状態方程式を前進オイラー法で離散化して等式条件として制約条件に追加
    opti.subject_to(X[0,i+1]==x+ dt*V*cos(beta)) # x方向の位置
    opti.subject_to(X[1,i+1]==y + dt*V*sin(beta)) # y方向の位置
    opti.subject_to(X[2,i+1]==theta + dt*theta_dot) # 機体の角度
    opti.subject_to(X[3,i+1]==V + dt*Uw/M) # 速度
    opti.subject_to(X[4,i+1]==beta + dt*beta_dot) # 操舵の角度
    opti.subject_to(X[5,i+1]==theta_dot + dt*theta_ddot) # 機体の角速度
    opti.subject_to(X[6,i+1]==beta_dot + dt*Us) # 操舵の角速度
    opti.subject_to(X[7,i+1]==theta_ddot + dt*Utheta) # 機体の角加速度
    
    opti.subject_to(cf-M*V*beta_dot==0) # 横力の釣り合い?
    opti.subject_to(Power-M*V*Uw==0) # 機体の運動エネルギー?
    
    opti.subject_to(-max_steer_acc<=Us) # ステアの最大加速度下限
    opti.subject_to(Us<=max_steer_acc) # ステアの最大加速度上限
    opti.subject_to(-max_steer_vel<=beta_dot) # ステアの最大速度下限
    opti.subject_to(beta_dot<=max_steer_vel) # ステアの最大速度上限
    opti.subject_to(-max_steer_torque<=Uw) # ステアの最大トルク下限
    opti.subject_to(Uw<=max_steer_torque) # ステアの最大トルク上限
    opti.subject_to(-max_power<=Power) # ステアの最大トルク下限
    opti.subject_to(Power<=max_power) # ステアの最大トルク上限
    opti.subject_to(-max_cf<=cf) # ステアの最大トルク下限
    opti.subject_to(cf<=max_cf) # ステアの最大トルク上限

# 初期状態の設定
for k in range(nx):
    opti.subject_to(X[k,0]==0) # 初期状態の設定
# 終端状態の設定
opti.subject_to(X[0,N-1]==1) # 終端状態の設定
opti.subject_to(X[1,N-1]==1) # 終端状態の設定
opti.subject_to(X[2,N-1]==0) # 終端状態の設定
opti.subject_to(X[3,N-1]==0) # 終端状態の設定
opti.subject_to(X[4,N-1]==0) # 終端状態の設定
opti.subject_to(X[5,N-1]==0) # 終端状態の設定
opti.subject_to(X[6,N-1]==0) # 終端状態の設定
opti.subject_to(X[7,N-1]==0) # 終端状態の設定

opti.minimize(J) # 最小化問題として設定
opti.solver('ipopt') # 最適化問題を宣言
sol = opti.solve() # 最適化問題を解く 解けなかったら'Infeasible_Problem_Detected'というエラーが出る
x_opt = sol.value(X) # 結果の取得
u_opt = sol.value(u) # 結果の取得
z_opt = sol.value(z) # 結果の取得

# 結果のプロット
import matplotlib.pyplot as plt

t_grid = [dt*k for k in range(N)]
plt.figure()
plt.plot(x_opt[0,:],x_opt[1,:],'-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('position')
plt.grid()

plt.figure()
plt.plot(t_grid, x_opt[3,:],'-')
plt.grid()

# plt.figure()
# plt.step(t_grid[:-1], u_opt)
# plt.xlabel('Time [s]')
# plt.ylabel('Control Input')
# plt.title('Control Input over Time')
# plt.grid()

plt.show()

