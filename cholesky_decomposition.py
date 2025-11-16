import numpy as np

def cholesky_decomposition(A):
    """
    Performs Cholesky decomposition on a symmetric, positive-definite
    matrix A such that A = L * L.T, where L is a lower-triangular matrix.

    Args:
        A (np.ndarray): A symmetric, positive-definite square matrix.

    Returns:
        np.ndarray: The lower-triangular matrix L.

    Raises:
        ValueError: If the matrix is not square, not symmetric, or
                    not positive-definite.
    """

    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric.")

    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                val = A[i, i] - s
                if val < 1e-9:
                    raise ValueError(f"Matrix is not positive-definite. Failed at diagonal {i}.")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    A1 = np.array([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ], dtype=float)

    print("--- Example 1 (Valid Matrix) ---")
    print("Original Matrix A:\n", A1)

    try:
        L_manual = cholesky_decomposition(A1)
        print("\nOur L:\n", L_manual)

        L_numpy = np.linalg.cholesky(A1)
        print("\nNumPy's L:\n", L_numpy)

        A_reconstructed = L_manual @ L_manual.T
        print("\nL @ L.T (Reconstructed A):\n", A_reconstructed)

        print("\nSuccess:", np.allclose(A1, A_reconstructed))

    except ValueError as e:
        print("\nFailed:", e)

    print("-" * 30)

    A2 = np.array([
        [1, 2],
        [2, 1]
    ], dtype=float)

    print("\n--- Example 2 (Not Positive-Definite) ---")
    print("Original Matrix A:\n", A2)

    try:
        L = cholesky_decomposition(A2)
        print("\nOur L:\n", L)
    except ValueError as e:
        print("\nFailed as expected:")
        print(e)

    print("-" * 30)
