import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve_assignment(G):
    """
    Решает задачу о назначениях с помощью алгоритма Венгера на матрице G
    """
    row_ind, col_ind = linear_sum_assignment(G)
    X = np.zeros_like(G)
    X[row_ind, col_ind] = 1
    return X


def compute_subgradient(X, D_list, b):
    """
    Вычисляет субградиент двойственной функции
    """
    K = len(D_list)
    grad = np.zeros(K)
    for k in range(K):
        grad[k] = np.sum(D_list[k] * X) - b[k]
    return grad


def compute_G(C, D_list, lambd):
    """
    Строит матрицу G из C, D^k и lambda^k
    """
    G = C.copy()
    for k in range(len(D_list)):
        G += D_list[k] * lambd[k]
    return G

def subgradient_method(C, D_list, b, lambda_0=None, N_max=100, eps=1e-4, step_rule='1/n', psi_const=1.0):
    """
    Реализация субградиентного метода для задачи о назначениях с дополнительными ограничениями
    """
    n = C.shape[0]
    K = len(D_list)
    lambda_k = np.ones(K) if lambda_0 is None else np.array(lambda_0)

    history = {
        'lambda': [],
        'grad': [],
        'cost': [],
        'violations': [],
        'X': [],
        'G': [],
        'time': []
    }

    for N in range(N_max):
        start_time = time.time()

        # Шаг 1: построить G
        G = compute_G(C, D_list, lambda_k)

        # Шаг 2: решить задачу о назначениях
        X = solve_assignment(G)

        # Шаг 3: субградиент
        grad = compute_subgradient(X, D_list, b)

        # Логгирование
        history['lambda'].append(lambda_k.copy())
        history['grad'].append(grad.copy())
        history['cost'].append(np.sum(C * X))
        history['violations'].append(np.linalg.norm(grad, ord=1))
        history['X'].append(X.copy())
        history['G'].append(G.copy())
        history['time'].append(time.time() - start_time)

        # Критерии останова
        if np.linalg.norm(grad, ord=1) <= eps:
            break

        # Шаг 4: шаг метода
        if step_rule == 'const':
            psi = psi_const
        elif step_rule == '1/n':
            psi = 1 / (N + 1)
        else:
            raise ValueError("Unknown step rule")

        # Обновление лямбда с проекцией на неотрицательные
        lambda_k = np.maximum(0, lambda_k + psi * grad)

    return X, lambda_k, history
