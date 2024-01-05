from casadi import *
import numpy as np
 

# 問題設定
T = 10.0     # ホライゾン長さ
N = 100      # ホライゾン離散化グリッド数
dt = T / N  # 離散化ステップ
nx = 8      # 状態空間の次元
nu = 3      # 制御入力の次元

M =10 # 機体の質量[kg]

# 以下で非線形計画問題(NLP)を定式化
w = []    # 最適化変数を格納する list
w0 = []   # 最適化変数(w)の初期推定解を格納する list 状態xも出力uも両方入れる
lbw = []  # 最適化変数(w)の lower bound を格納する list 対応するリスト番号のところに数値を入れる
ubw = []  # 最適化変数(w)の upper bound を格納する list
J = 0     # コスト関数 
g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
lbg = []  # 制約関数(g)の lower bound を格納する list 等式制約は lower-bound と upper-bound を両方0にする
ubg = []  # 制約関数(g)の upper bound を格納する list

Xk = MX.sym('X0', nx) # 初期時刻の状態ベクトル x0
w += [Xk]             # x0 を 最適化変数 list (w) に追加
# 初期状態は given という条件を等式制約として考慮
lbw += [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ]  # 初期の境界値条件をあらかじめ記入しておく(見やすい)　リストに逐次追加していく 
ubw += [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ]   
w0 +=  [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ]  # x0 の初期推定解 TODO startpointでユーザーが決めれるように


# 離散化ステージ 0~N-1 までのコストと制約を設定
for k in range(N):
    Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
    w   += [Uk]                    # uk を最適化変数 list に追加
    lbw += [-1,-1,-1]                 # uk の lower-bound TODO 入力の制限を決める
    ubw += [1,1,1]                  # uk の upper-bound TODO 入力の制限を決める
    w0  += [0,0,0]                     # uk の初期推定解

    
    x = Xk[0]     # x[m]
    y = Xk[1]    # y[m]
    theta = Xk[2]    # 速度ベクトルの水平方向から見た角度[rad]
    V = Xk[3]     # 速度[m/s]（制御入力）
    beta = Xk[4]   # 角速度[rad/s]
    theta_dot = Xk[5] # 角速度[rad/s]
    beta_dot = Xk[6] # 操舵の角加速度[rad/s^2]
    theta_ddot = Xk[7] # 機体の角加速度[rad/s^2]
    
    Uw = Uk[0] # トルク？？？
    Us = Uk[1] # 推力？？？
    Utheta = Uk[2] # 操舵の加速度？？？
    
    
    # ステージコストのパラメータ 
    x_ref = [2.0, 2.0 , 0 , 0 , 0 , 0 , 0 , 0]  # 目標状態
    Q = [10.0, 10.0, 0.01 , 0.1 ,0.1 ,0.1 , 0.1 ,0.1]       # 状態への重み
    R = [1.0,1.0,1.0]                         # 制御入力への重み
    L = 0                             # ステージコスト
    for i in range(nx):
        L += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2 
    for i in range(nu):
        L += 0.5 * R[i] * Uk[i]**2
    J = J + L * dt                    # コスト関数にステージコストを追加

    # Forward Euler による離散化状態方程式
    Xk_next = vertcat(x + V*np.cos(beta) * dt,  #x
                      y + V*np.sin(beta) * dt, #y
                      theta + theta_dot * dt,# theta
                      V + Uw/M * dt, # V
                      beta + beta_dot * dt,
                      theta_dot + theta_ddot * dt,
                      beta_dot + Us,
                      theta_ddot + Utheta)
    Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
    w   += [Xk1]                       # xk+1 を最適化変数 list に追加
    lbw += [-inf, -inf, -inf,-inf,-inf,-inf,-inf,-inf]    # xk+1 の lower-bound （指定しない要素は -inf）TODO ここで制約条件を指定する
    ubw += [inf, inf, inf,inf,inf,inf,inf,inf]        # xk+1 の upper-bound （指定しない要素は inf）TODO ユーザーが指定しない場合はinfにする
    w0 += [0, 0, 0 , 0 , 0 , 0 , 0 , 0]         # xk+1 の初期推定解　TODO 一つ前のステップの状態を入れる？？？

    # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
    g   += [Xk_next-Xk1] # TODO 別に状態方程式のdtをかけてる項のみでも同じじゃね？？？
    lbg += [0, 0, 0, 0, 0, 0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定 ここは状態方程式で等式になってほしいから確定で0が入る
    ubg += [0, 0, 0, 0, 0, 0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    Xk = Xk1

# 終端コストのパラメータ 
x_ref = [2.0, 2.0, 0, 0, 0, 0, 0, 0]  # 目標状態
Q = [10.0, 10.0, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1]       # 状態への重み
Vf = 0                            # 終端コスト
for i in range(nx):
    Vf += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2 
for i in range(nu):
    Vf += 0.5 * R[i] * Uk[i]**2
J = J + Vf                        # コスト関数に終端コストを追加


# 非線形計画問題(NLP)
nlp = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)} 
# Ipopt ソルバー，最小バリアパラメータを0.001に設定
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt':{'mu_min':0.001}}) 
# SQP ソルバー（QPソルバーはqpOASESを使用），QPソルバーの regularization 無効，QPソルバーのプリント無効
# solver = nlpsol('solver', 'sqpmethod', nlp, {'max_iter':100, 'qpsol_options':{'enableRegularisation':False, 'printLevel':None}})

# NLPを解く
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# 解をプロット
x0_opt = w_opt[0::nx+nu]
x1_opt = w_opt[1::nx+nu]
x2_opt = w_opt[2::nx+nu]
x3_opt = w_opt[3::nx+nu]
x4_opt = w_opt[4::nx+nu]
x5_opt = w_opt[5::nx+nu]
x6_opt = w_opt[6::nx+nu]
x7_opt = w_opt[7::nx+nu]
u0_opt = w_opt[8::nx+nu]
u1_opt  = w_opt[9::nx+nu]
u2_opt  = w_opt[10::nx+nu]

tgrid = [dt*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.clf() # clear current figure
fig,axes = plt.subplots(2,3)
axes[0,0].plot(x0_opt, x1_opt, '--')
axes[0,0].set_title('x-y')
axes[0,2].plot(tgrid,x3_opt,'--')
axes[0,2].set_title('V')
axes[1,1].plot(tgrid,x4_opt,'--')
axes[1,1].set_title('beta')
axes[1,2].plot(tgrid,x6_opt,'--')
axes[1,2].set_title('beta_dot')
plt.figure(1)
plt.step(tgrid, vertcat(DM.nan(1), u1_opt), '-.')
plt.step(tgrid, vertcat(DM.nan(1), u2_opt), '-.')
plt.xlabel('t')
plt.legend(['y(x2)', 'beta(x3)','u1(V)','u2(beta_dot)'])
plt.plot(x1_opt, x2_opt, '--')
plt.figure(2)
plt.grid()
plt.show()