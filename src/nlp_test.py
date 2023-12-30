from casadi import *
import numpy as np
import math

# direct collocationであるが離散化のみ前進オイラー法を使用している。

# 問題設定
T = 10.0     # ホライゾン長さ
N = 100      # ホライゾン離散化グリッド数
dt = T / N  # 離散化ステップ
nx = 3      # 状態空間の次元
nu = 2      # 制御入力の次元

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
lbw += [0, 0, 0]  # 初期の境界値条件をあらかじめ記入しておく(見やすい)　リストに逐次追加していく ここはx[0],x[1],x[2]の初期値を順に記述してる
ubw += [0, 0, 0]   
w0 +=  [0, 0, 0]  # x0 の初期推定解


# 離散化ステージ 0~N-1 までのコストと制約を設定
for k in range(N):
    Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
    w   += [Uk]                    # uk を最適化変数 list に追加
    lbw += [-3.0,-math.pi/6]                 # uk の lower-bound
    ubw += [3.0,math.pi/6]                  # uk の upper-bound
    w0  += [0,0]                     # uk の初期推定解

    # 以下，cartpole のステージコストとダイナミクスを記述
    # cartpoleの運動方程式は(https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf)を参照した
    x = Xk[0]     # x[m]
    y = Xk[1]    # y[m]
    beta = Xk[2]    # 速度ベクトルの水平方向から見た角度[rad]
    V = Uk[0]     # 速度[m/s]（制御入力）
    beta_dot = Uk[1]   # 角速度[rad/s]
    
    # ステージコストのパラメータ 線形二次レギュレータ(Linear-Quadratic Regulator:LQR)を使用している
    x_ref = [2.0, 2.0, 0.0]# 目標状態
    Q = [1.0, 1.0, 0.01]       # 状態への重み
    R = [1.0,1.0]                         # 制御入力への重み
    L = 0                             # ステージコスト
    for i in range(nx):
        L += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2 
    for i in range(nu):
        L += 0.5 * R[i] * Uk[i]**2
    J = J + L * dt                    # コスト関数にステージコストを追加

    # Forward Euler による離散化状態方程式
    Xk_next = vertcat(x + V*np.cos(beta) * dt, 
                      y + V*np.sin(beta) * dt,
                      beta + beta_dot * dt)
    Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
    w   += [Xk1]                       # xk+1 を最適化変数 list に追加
    lbw += [-inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
    ubw += [inf, inf, inf]        # xk+1 の upper-bound （指定しない要素は inf）
    w0 += [0.0, 0.0, 0.0]         # xk+1 の初期推定解

    # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
    g   += [Xk_next-Xk1]
    lbg += [0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定 
    ubg += [0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    Xk = Xk1

# 終端コストのパラメータ 
x_ref = [2.0, 2.0, 0.0]  # 目標状態
Q = [10.0, 10.0, 0.01]       # 状態への重み
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
x1_opt = w_opt[0::5]
x2_opt = w_opt[1::5]
x3_opt = w_opt[2::5]
u1_opt = w_opt[3::5]
u2_opt  = w_opt[4::5]

tgrid = [dt*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x3_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u1_opt), '-.')
plt.step(tgrid, vertcat(DM.nan(1), u2_opt), '-.')
plt.xlabel('t')
plt.legend(['y(x2)', 'beta(x3)','u1(V)','u2(beta_dot)'])
plt.figure(2)
plt.plot(x1_opt, x2_opt, '--')

plt.grid()
plt.show()