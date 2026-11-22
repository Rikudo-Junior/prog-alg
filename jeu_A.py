import numpy as np
import matplotlib.pyplot as plt
import tp_dataset1 as tdb
import estimators as est  
import optimers as opt


#####Question 4
### a
rng = np.random.default_rng(42)
n, d, noise_std = 1000, 20, 5.0

X, y, w_cons = tdb.make_dataset_A(n, d, noise_std, rng)

### b
X_train, y_train = X[:500], y[:500]
X_val, y_val = X[500:], y[500:]

### c
lam, w_init, n_iter = 1, np.zeros(d), 10
plt.figure()
for pas in [1e-3, 1e-2, 1e-1,0.5,1]:
    w_opt, history = opt.grad_descent(X, y, w_init, lam, pas, n_iter )
    plt.semilogy(history, label = f"pas={pas}")
plt.xlabel("Itération k")
plt.ylabel("L(w^k)")
plt.title(f"Descente de gradient, lambda = {lam}")
plt.legend()
plt.tight_layout()
plt.savefig("4c_convergence_pas.png")





