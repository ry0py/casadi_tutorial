from casadi import *
import math

# 問題設定
T = 5.0     # ホライゾン長さ
N = 100      # ホライゾン離散化グリッド数
dt = T / N  # 離散化ステップ
nx = 4      # 状態空間の次元
nu = 1      # 制御入力の次元

# 以下で非線形計画問題(NLP)を定式化
w = []    # 最適化変数を格納する list
w0 = []   # 最適化変数(w)の初期推定解を格納する list
lbw = []  # 最適化変数(w)の lower bound を格納する list
ubw = []  # 最適化変数(w)の upper bound を格納する list
J = 0     # コスト関数 
g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
lbg = []  # 制約関数(g)の lower bound を格納する list
ubg = []  # 制約関数(g)の upper bound を格納する list

Xk = MX.sym('X0', nx) # 初期時刻の状態ベクトル x0
w += [Xk]             # x0 を 最適化変数 list (w) に追加
# 初期状態は given という条件を等式制約として考慮
lbw += [0., 0., 0., 0.]  # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
ubw += [0, 0, 0, 0]  # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
w0 +=  [0, 0, 0, 0]  # x0 の初期推定解


# 離散化ステージ 0~N-1 までのコストと制約を設定
for k in range(N):
    Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
    w   += [Uk]                    # uk を最適化変数 list に追加
    lbw += [-5.0]                 # uk の lower-bound
    ubw += [5.0]                  # uk の upper-bound
    w0  += [0]                     # uk の初期推定解

    # 以下，cartpole のステージコストとダイナミクスを記述
    # cartpoleの運動方程式は(https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf)を参照した
    y = Xk[0]     # cart の水平位置[m]
    th = Xk[1]    # pole の傾き角[rad]
    dy = Xk[2]    # cart の水平速度[m/s]
    dth = Xk[3]   # pole の傾き角速度[rad/s]
    f = Uk[0]     # cart を押す力[N]（制御入力）
    # ステージコストのパラメータ 
    x_ref = [0.0, math.pi, 0.0, 0.0]  # 目標状態
    Q = [1.0, 10.0, 0.01, 0.01]       # 状態への重み
    R = [1.0]                         # 制御入力への重み
    L = 0                             # ステージコスト
    for i in range(nx):
        L += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2 
    for i in range(nu):
        L += 0.5 * R[i] * Uk[i]**2
    J = J + L * dt                    # コスト関数にステージコストを追加

    # cartpole の物理パラメータ 
    mc = 2.0   # cart の質量[kg]
    mp = 0.2   # pole の質量[kg]
    l = 0.5    # pole の長さ[m]
    ga = 9.81   # 重力加速度[m/s2]
    # cart の水平加速度
    ddy = (f+mp*sin(th)*(l*dth*dth+ga*cos(th))) / (mc+mp*sin(th)*sin(th)) 
    # pole の傾き角加速度
    ddth = (-f*cos(th)-mp*l*dth*dth*cos(th)*sin(th)-(mc+mp)*ga*sin(th)) / (l * (mc+mp*sin(th)*sin(th))) 
    # Forward Euler による離散化状態方程式
    Xk_next = vertcat(y + dy * dt, 
                      th + dth * dt,
                      dy + ddy * dt,
                      dth + ddth * dt)
    Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
    w   += [Xk1]                       # xk+1 を最適化変数 list に追加
    lbw += [-10., -inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
    ubw += [10., inf, inf, inf]        # xk+1 の upper-bound （指定しない要素は inf）
    w0 += [0.0, 0.0, 0.0, 0.0]         # xk+1 の初期推定解

    # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
    g   += [Xk_next-Xk1]
    lbg += [0, 0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    ubg += [0, 0, 0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    Xk = Xk1

# 終端コストのパラメータ 
x_ref = [0.0, math.pi, 0.0, 0.0]  # 目標状態
Q = [1.0, 10.0, 0.01, 0.01]       # 状態への重み
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
x4_opt = w_opt[3::5]
u_opt  = w_opt[4::5]

tgrid = [dt*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.plot(tgrid, x3_opt, '-')
plt.plot(tgrid, x4_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['y(x1)','th(x2)', 'dy(x3)', 'dth(x4)','u'])
plt.grid()
plt.show()