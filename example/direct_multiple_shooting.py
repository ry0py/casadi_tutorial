### copy https://helve-blog.com/posts/python/casadi-direct-multiple-shooting/


import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

Ts = 1.0 # サンプリング周期[s]
N = 20 # サンプリング点数

m = 1 # 質量[kg]

# 運動方程式を微分方程式で記述する
z = ca.MX.sym('z', 4) # x, y, x_dot, y_dot
u = ca.MX.sym("u", 2)
zdot = ca.vertcat(z[2], z[3], u[0]/m,u[1]/m)

# 初期・終端状態
z0 = [0, 0, 0, 0]
zf = [20, 15, 0, 0]

# Objective term
L = (z[0]-zf[0])**2 + (z[1]-zf[1])**2 + u[0]**2 + u[1]**2

dae = {'x': z,
       'p': u,
       'ode': zdot,
       'quad':L}
opts = {'tf':Ts}
F = ca.integrator('F', 'cvodes', dae, opts)

w = []; lbw = []; ubw = []; G = []; J = 0
# 変数とその上下限、等式制約、評価関数

Xk = ca.MX.sym('X', 4)
w += [Xk]
lbw += z0
ubw += z0

for i in range(N):
    Uk = ca.MX.sym(f'U{i}', 2)
    w += [Uk]
    lbw += [-1, -1]
    ubw += [1, 1]

    Fk = F(x0=Xk, p=Uk) # Tsだけシミュレーションする（微分方程式を解く）
    J += Fk["qf"]

    Xk = ca.MX.sym(f'X{i+1}', 4)
    w += [Xk]
    G += [Fk['xf'] - Xk]
    if i != N-1:
        lbw += [-100, -100, -100, -100]
        ubw += [100, 100, 100, 100]
    else:
        # 最終ステップでは、位置誤差±1, 速度誤差±1以下
        lbw += zf
        ubw += zf

prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*G)}
option = {"ipopt.print_level": 4}
solver = ca.nlpsol('solver', 'ipopt', prob, option)
sol = solver(x0=0, lbx=lbw, ubx=ubw, lbg=0, ubg=0)

x_star = np.array(sol["x"].elements() + [np.nan]*2)
x_star = x_star.reshape(-1, 6)
x_star = pd.DataFrame(x_star, columns=["x","y","x_dot","y_dot","Fx","Fy"])
print(x_star)

fig, ax = plt.subplots()
ax.plot(x_star["x"], x_star["y"])
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
