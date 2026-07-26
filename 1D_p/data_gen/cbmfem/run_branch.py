# -*- coding: utf-8 -*-
import numpy as np
from Sexample1 import faSh1
from Newton import Newton
import warnings
import sys

warnings.filterwarnings('ignore')

branch_id = int(sys.argv[1])
print(f"Branch {branch_id} start")

try:
    current_S = np.load('initial_S2.npy')
except:
    print("Failed to load initial data.")
    exit()

# --- Hyperparameters ---
p_current = 18.0
p_end = 0.0
dp = 0.001
max_allowed_diff = 0.2

branch_sol = current_S[:, branch_id:branch_id+1].copy()
branch_p_history = [p_current]
branch_s_history = [branch_sol.copy()]

print(f"Fixed dp={dp} | Branch {branch_id} Bifurcation start")

step = 0
p = p_current
active = True

while p > p_end and active:
    p_next = round(p - dp, 10)
    if p_next < p_end:
        p_next = p_end

    func_list = faSh1(branch_sol, p_next)
    F, dF = func_list[0], func_list[1]
    guess = branch_sol

    res = Newton(F, dF, guess, 10**(-9), 500)

    if res is None or not isinstance(res, np.ndarray):
        print(f"  Branch {branch_id} convergence failed at p={p_next:.6f} -> terminate")
        active = False
        break

    sol_updated = res[:-1, 0:1]
    diff = np.linalg.norm(sol_updated - guess, np.inf)

    if diff > max_allowed_diff:
        print(f"  Branch {branch_id} jump detected (diff={diff:.4f}) at p={p_next:.6f} -> terminate")
        active = False
        break

    branch_sol = sol_updated.copy()
    branch_p_history.append(p_next)
    branch_s_history.append(branch_sol.copy())

    p = p_next
    step += 1

    if step % 1000 == 0:
        print(f"  Branch {branch_id} | p = {p:.4f} | step = {step}")

# Save results
p_arr = np.array(branch_p_history)
s_arr = np.concatenate(branch_s_history, axis=1)
np.save(f'branch_{branch_id}_p_new.npy', p_arr)
np.save(f'branch_{branch_id}_S_new.npy', s_arr)
print(f"Branch {branch_id} done: {len(branch_p_history)} steps | S shape={s_arr.shape}")