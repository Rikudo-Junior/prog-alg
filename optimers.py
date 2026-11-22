import numpy as np
import tp_dataset1 as tdb
import estimators
from numpy.typing import NDArray


def loss_grad(X: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64], lam: float):
    n = X.shape[0]
    r = X @ w - y
    loss = 1 / n * np.linalg.norm(r) ** 2 + lam * np.linalg.norm(w) ** 2
    grad = 2 / n * (X.transpose() @ r) + 2 * lam * w
    return loss, grad

def grad_descent(X: NDArray[np.float64], y: NDArray[np.float64],w_init: NDArray[np.float64], lam: float, pas:float, n_iter: int ):
    w= w_init.copy()
    history = np.empty(n_iter)
    for i in range(n_iter):
        loss, grad = loss_grad(X, y, w, lam)
        history[i] = loss
        w -= pas * grad
    return w, history

def grad_descent_momentum(X: NDArray[np.float64], y: NDArray[np.float64],w_init: NDArray[np.float64], lam: float, pas:float, gamma: float, n_iter: int ):
    w=w_init.copy()
    m = np.zeros_like(w)
    history = np.empty(n_iter)
    for i in range(n_iter):
        loss, grad = loss_grad(X, y, w, lam)
        history[i] = loss
        m = gamma*m + (1-gamma)*grad
        w -= pas * m
    return w, history

def grad_descent_nesterov(X: NDArray[np.float64], y: NDArray[np.float64],w_init: NDArray[np.float64], lam: float, pas: float,n_iter: int, gamma_init: float = 0.0):
    w_0=w_init.copy()
    w_bar = w_init.copy()
    gamma_0=gamma_init
    history = np.empty(n_iter)
    for i in range(n_iter):
        gamma_1=(1+ np.sqrt(1 + 4*gamma_0**2))/2 
        eta = (gamma_0 - 1)/gamma_1
        gamma_0 = gamma_1
        loss, grad = loss_grad(X, y, w_bar, lam)
        history[i] = loss
        w_1 = w_bar - pas * grad
        w_bar = w_1 + eta * (w_1 - w_0)
        w_0 = w_1
    return w_1, history


if __name__ == "__main__":
     rng = np.random.default_rng(42)
     X_A, y_A, w_A = tdb.make_dataset_A(1000, 20, 5.0, rng)
     w = estimators.ridge_estimator(X_A, y_A, 0.0)
     loss, grad = loss_grad(X_A, y_A, w, 0.0)
     print(loss, grad)

