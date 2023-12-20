from casadi import *
x = SX.sym('x',2)
f = x[0]**2+x[1]**2 # 目的関数
g = x[0]+x[1] # 制約条件
nlp = {'x':x,'f':f,'g':g} # 最適化問題の定義
opts = {'ipopt':{'max_iter':1000}} # 最適化ソルバーの設定
solver = nlpsol('solver','ipopt',nlp,opts) # 最適化問題を解くソルバーの定義
sol = solver(lbg=0,ubg=0,lbx=-inf,ubx=inf,x0=[0,0]) # 最適化問題を解く
print(sol['x']) # 解を表示
# print(sol['f']) # 目的関数の値を表示
# print(sol['g']) # 制約条件の値を表示
# print(sol['lam_g']) # ラグランジュ乗数を表示
# print(sol['lam_x']) # ラグランジュ乗数を表示
# print(x)