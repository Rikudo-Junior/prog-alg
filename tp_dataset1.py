import numpy as np



##### Jeu de données A
def make_dataset_A(n: int, d: int, noise_std: float, rng: np.random.Generator):
    X = rng.normal(0, 1, size = (n,d))
    w = rng.normal(0, 1, size = X.shape[1])
    eps = rng.normal(0, noise_std, size = X.shape[0])
    y = X @ w + eps

    return  X, y, w 

##### Jeu de données B
def make_dataset_B(n: int, d: int, noise_std: float, rng: np.random.Generator):
    return make_dataset_A(n, d, noise_std, rng)



##### Jeu de données C
def make_dataset_C(n: int, d:int, noise_std: float, rng: np.random.Generator):
    X = rng.normal(0,1, size = (n,d))
    z = rng.normal(0, 1, size = X.shape[0])
    x_1 = z + rng.normal(0, 0.1, size = X.shape[0])
    x_2 = z + rng.normal(0, 0.1, size = X.shape[0])
    X[:,0] = x_1
    X[:,1] = x_2
    w = rng.normal(0, 1, size = X.shape[1])
    eps = noise_std * rng.normal(0, 1, size = X.shape[0])
    y = X @ w + eps
    return X, y, w 
