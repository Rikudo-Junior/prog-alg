import numpy as np
import tp_dataset1 as tdb
from numpy.typing import NDArray

def ridge_estimator(X: NDArray[np.float64], y: NDArray[np.float64], lam: float = 0.0 ) -> NDArray[np.float64]:
    """ Solution analytique de Ridge, de shape (d,)
    
    X: ndarray(n,d) matrice de design, 
    y: ndarray (n,)  variable cible,
    lam : float  paramètre de régularisation
    ------------------------

    """
    n, d = X.shape[0], X.shape[1]
    
    w = np.linalg.solve(X.transpose() @ X + lam * n * np.identity(d), X.transpose() @ y )
    return w


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_A, y_A, w_A = tdb.make_dataset_A(1000, 20, 5.0, rng)
    y_clean= X_A @ w_A
    w_clean = ridge_estimator(X_A, y_clean)
    assert np.allclose(w_clean, w_A)
    print("Test sans bruit ok")
    w_noisy = ridge_estimator(X_A, y_A)
    print("Écart avec bruit :",np.linalg.norm(w_noisy-w_A))