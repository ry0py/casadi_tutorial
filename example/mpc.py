from casadi import *
import math
import numpy as np


# cartpole のダイナミクスを記述
# cartpoleの運動方程式は(https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf)を参照した
class CartPole:
    def __init__(self):
        self.mc = 2.0   # cart の質量[kg]
        self.mp = 0.2   # pole の質量[kg]
        self.l = 0.5    # pole の長さ[m]
        self.ga = 9.81   # 重力加速度[m/s2]

    def dynamics(self, x, u):
        mc = self.mc
        mp = self.mp
        l = self.l
        ga = self.ga
        y = x[0]     # cart の水平位置[m]
        th = x[1]    # pole の傾き角[rad]
        dy = x[2]    # cart の水平速度[m/s]
        dth = x[3]   # pole の傾き角速度[rad/s]
        f = u[0]     # cart を押す力[N]（制御入力）
        # cart の水平加速度
        ddy = (f+mp*sin(th)*(l*dth*dth+ga*cos(th))) / (mc+mp*sin(th)*sin(th)) 
        # pole の傾き角加速度
        ddth = (-f*cos(th)-mp*l*dth*dth*cos(th)*sin(th)-(mc+mp)*ga*sin(th)) / (l * (mc+mp*sin(th)*sin(th))) 
        return dy, dth, ddy, ddth

# コスト関数を記述
class CostFunction:
    def __init__(self):
        self.nx = 4
        self.nu = 1
        self.x_ref = [0.0, math.pi, 0.0, 0.0]   # 目標状態
        # ステージコストのパラメータ 
        self.Q  = [2.5, 10.0, 0.01, 0.01]       # 状態への重み
        self.R  = [0.1]                         # 制御入力への重み
        # 終端コストのパラメータ 
        self.Qf = [2.5, 10.0, 0.01, 0.01]       # 状態への重み

    # ステージコスト
    def stage_cost(self, dt, x, u):
        L = 0                             
        for i in range(self.nx):
            L += 0.5 * self.Q[i] * (x[i]-self.x_ref[i])**2 
        for i in range(self.nu):
            L += 0.5 * self.R[i] * u[i]**2
        return dt * L

    # 終端コスト
    def terminal_cost(self, x):
        Vf = 0
        for i in range(self.nx):
            Vf += 0.5 * self.Q[i] * (x[i]-self.x_ref[i])**2 
        return Vf


class MPC:
    def __init__(self):
        # 問題設定
        T = 1.0     # ホライゾン長さ (MPCなので短め)
        N = 20      # ホライゾン離散化グリッド数 (MPCなので荒め)
        dt = T / N  # 離散化ステップ
        nx = 4      # 状態空間の次元
        nu = 1      # 制御入力の次元
        cartpole = CartPole() # cartpole のダイナミクス
        cost_function = CostFunction() # コスト関数

        # 以下で非線形計画問題(NLP)を定式化
        w = []    # 最適化変数を格納する list
        w0 = []   # 最適化変数(w)の初期推定解を格納する list
        lbw = []  # 最適化変数(w)の lower bound を格納する list
        ubw = []  # 最適化変数(w)の upper bound を格納する list
        J = 0     # コスト関数 
        g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
        lbg = []  # 制約関数(g)の lower bound を格納する list
        ubg = []  # 制約関数(g)の upper bound を格納する list
        lam_x0 = []  # 制約 lbw<w<ubw のラグランジュ乗数
        lam_g0 = []  # 制約 lbg<g<ubg のラグランジュ乗数

        Xk = MX.sym('X0', nx) # 初期時刻の状態ベクトル x0
        w += [Xk]             # x0 を 最適化変数 list (w) に追加
        # 初期状態は given という条件を等式制約として考慮
        lbw += [0, 0, 0, 0]  # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        ubw += [0, 0, 0, 0]      # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        w0 +=  [0, 0, 0, 0]      # x0 の初期推定解
        lam_x0 += [0, 0, 0, 0]    # ラグランジュ乗数の初期推定解

        # 離散化ステージ 0~N-1 までのコストと制約を設定
        for k in range(N):
            Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
            w   += [Uk]                   # uk を最適化変数 list に追加
            lbw += [-15.0]                # uk の lower-bound
            ubw += [15.0]                 # uk の upper-bound
            w0  += [0]                    # uk の初期推定解
            lam_x0 += [0]                 # ラグランジュ乗数の初期推定解

            # ステージコスト
            J = J + cost_function.stage_cost(dt, Xk, Uk) # コスト関数にステージコストを追加

            # Forward Euler による離散化状態方程式
            dXk = cartpole.dynamics(Xk, Uk)
            Xk_next = vertcat(Xk[0] + dXk[0] * dt, 
                              Xk[1] + dXk[1] * dt,
                              Xk[2] + dXk[2] * dt,
                              Xk[3] + dXk[3] * dt)
            Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
            w   += [Xk1]                       # xk+1 を最適化変数 list に追加
            lbw += [-inf, -inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
            ubw += [inf, inf, inf, inf]        # xk+1 の upper-bound （指定しない要素は inf）
            w0 += [0.0, 0.0, 0.0, 0.0]         # xk+1 の初期推定解
            lam_x0 += [0, 0, 0, 0]              # ラグランジュ乗数の初期推定解

            # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
            g   += [Xk_next-Xk1]
            lbg += [0, 0, 0, 0]     # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
            ubg += [0, 0, 0, 0]     # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
            lam_g0 += [0, 0, 0, 0]   # ラグランジュ乗数の初期推定解
            Xk = Xk1

        # 終端コスト 
        J = J + cost_function.terminal_cost(Xk) # コスト関数に終端コストを追加

        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        # 非線形計画問題(NLP)
        self.nlp = {'f': self.J, 'x': self.w, 'g': self.g} 
        # Ipopt ソルバー，最小バリアパラメータを0.1，最大反復回数を5, ウォームスタートをONに
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':5, 'mu_min':0.1, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}}) 
        # self.solver = nlpsol('solver', 'scpgen', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'qpsol':'qpoases', 'print_time':False, 'print_header':False, 'max_iter':5, 'hessian_approximation':'gauss-newton', 'qpsol_options':{'print_out':False, 'printLevel':'none'}}) # print をオフにしたいがやり方がわからない


    def init(self, x0=None):
        if x0 is not None:
            # 初期状態についての制約を設定
            self.lbx[0:4] = x0
            self.ubx[0:4] = x0
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

    def solve(self, x0):
        # 初期状態についての制約を設定
        nx = x0.shape[0]
        self.lbx[0:nx] = x0
        self.ubx[0:nx] = x0
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()
        return np.array([self.x[4]]) # 制御入力を return



# Closed-loop シミュレーション
sim_time = 10.0 # 10秒間のシミュレーション
sampling_time = 0.001 # 0.001秒（1ms）のサンプリング周期
sim_steps = math.floor(sim_time/sampling_time)
xs = []
us = []
cartpole = CartPole()
mpc = MPC()
mpc.init()
x = np.zeros(4)
for step in range(sim_steps):
    if step%(1/sampling_time)==0:
        print('t =', step*sampling_time)
    u = mpc.solve(x)
    xs.append(x)
    us.append(u)
    x1 = x + sampling_time * np.array(cartpole.dynamics(x, u))
    x = x1


# シミュレーション結果をプロット
xs1 = [x[0] for x in xs]
xs2 = [x[1] for x in xs]
xs3 = [x[2] for x in xs]
xs4 = [x[3] for x in xs]
tgrid = [sampling_time*k for k in range(sim_steps)]

import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, xs1, '--')
plt.plot(tgrid, xs2, '-')
plt.plot(tgrid, xs3, '-')
plt.plot(tgrid, xs4, '-')
plt.step(tgrid, us, '-.')
plt.xlabel('t')
plt.legend(['y(x1)','th(x2)', 'dy(x3)', 'dth(x4)','u'])
plt.grid()
plt.show()