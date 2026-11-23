import estimators as est
import matplotlib.pyplot as plt
import numpy as np
import optimers as opt
import tp_dataset1 as tdb

#####Question 4
### a
rng = np.random.default_rng(42)
n, d, noise_std = 1000, 20, 5.0

X, y, w_ref = tdb.make_dataset_A(n, d, noise_std, rng)

###b
X_train, y_train = X[:500], y[:500]
X_val, y_val = X[500:], y[500:]

### c
lam, w_init, n_iter = 0.0, np.zeros(d), 100
plt.figure()
for pas in [1e-3, 1e-2, 1e-1]:
    w_etoile = est.ridge_estimator(X_train, y_train, lam)
    lossen, _= opt.loss_grad(X_train, y_train, w_etoile, lam)
    w_opt, history = opt.grad_descent(X_train, y_train, w_init, lam, pas, n_iter )
    plt.semilogy(history-lossen, label = f"pas={pas}")
plt.xlabel("Itération k")
plt.ylabel("L(w^k)-L(w^*)")
plt.title(f"Descente de gradient, lambda = {lam}")
plt.legend()
plt.tight_layout()
plt.savefig("4c_convergence_pas.png")

###d 

pas0 = 0.05
plt.figure()
for lam0 in [0.0, 0.1, 1.0, 5.0]:
    w_etoile = est.ridge_estimator(X_train, y_train, lam0)
    lossen, _ = opt.loss_grad(X_train, y_train, w_etoile, lam0)
    w_opt0, history0 = opt.grad_descent(X_train, y_train, w_init, lam0, pas0, 200)
    plt.semilogy(history0-lossen + 1e-16, label = f"lambda={lam0}")
    error_to_ref = np.sum((w_opt0-w_ref)**2)
    print(print(f"pas={pas0}, lambda={lam0}, error_to_ref={error_to_ref}"))
plt.xlabel("Itération k")
plt.ylabel("L(w^k)-L(w^*)")
plt.title(f"Descente de gradient, pas = {pas0}")
plt.legend()
plt.tight_layout()
plt.savefig("4d_convergence_pas.png") #### On constate que la convergence devient faibe à mesure que lamda augmente


print(history.argmin(), history.min())
print(history0.argmin(), history0.min())

###e
w_opt0 = opt.grad_descent(X_train, y_train, w_init, 0.0, pas0, 100 )[0]
w_m = est.ridge_estimator(X_train, y_train, lam = 0.0 )
error = np.sum((w_opt0-w_m)**2)
width =0.2
plt.figure()
plt.bar(np.arange(d) - width, w_opt.flatten(), width=width, label="Gradient descent")
plt.bar(np.arange(d) + width, w_m.flatten(), width=width, label="Analytique")
plt.xlabel("Parameter")
plt.ylabel("Value")
plt.title(f"Comparaison poids Ridge vs GD (error={error:.2e})")
plt.legend()
plt.savefig("4e_comparaison.png")

###f
lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
w_mat = np.empty((len(lambdas),d))
mse_train, mse_val = np.empty(len(lambdas)), np.empty(len(lambdas))
for i, lam in enumerate(lambdas):
    w_o, _ = opt.grad_descent(X_train, y_train,w_init, lam, 0.1, 200)
    w_ov, _ = opt.grad_descent(X_val, y_val,w_init, lam, 0.1, 200)
    w_mat[i] = w_o
    mse_train[i], mse_val[i] = np.mean((y_train - X_train @ w_o)**2), np.mean((y_val - X_val @ w_ov)**2)

plt.figure()
for j in range(d):
    plt.semilogx(lambdas, w_mat[:,j], marker = "o", label = f"Beta{j}")

plt.xlabel("pas lambda")
plt.ylabel("Coefficient Beta_j")
plt.title("Chemins de régularisations")
plt.savefig("4f_chemins_reg.png")


plt.figure()
plt.semilogx(lambdas, mse_train, label = "MSE entrainement")
plt.semilogx(lambdas, mse_val, label = "MSE validation")
#plt.xscale("symlog", linthresh =1e-4)
plt.xlabel("pas lambda")
plt.ylabel("mse")
plt.title("graphe mse")
plt.legend()
plt.savefig("4f_msegraph.png")