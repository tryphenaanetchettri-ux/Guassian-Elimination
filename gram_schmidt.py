import numpy as np

def gram_schmidt(V):
    """
    Performs the Modified Gram-Schmidt orthogonalization process
    on a set of vectors.

    Args:
        V (np.ndarray): A 2D array where each column is a vector.
                        Shape: (M, N).

    Returns:
        np.ndarray: Orthonormal matrix Q (same shape as V).
    """

    V_copy = V.astype(float).copy()
    num_rows, num_cols = V_copy.shape

    Q = np.zeros_like(V_copy)

    for i in range(num_cols):
        v_i = V_copy[:, i]
        norm_v_i = np.linalg.norm(v_i)

        if norm_v_i < 1e-9:
            raise ValueError(f"Vectors are linearly dependent. Failed at index {i}.")

        q_i = v_i / norm_v_i
        Q[:, i] = q_i

        for j in range(i + 1, num_cols):
            v_j = V_copy[:, j]
            projection_coeff = np.dot(v_j, q_i)
            V_copy[:, j] = v_j - projection_coeff * q_i

    return Q

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)

    print("--- Example 1 (3x3) ---")
    print("Original Matrix V:\n", A)

    try:
        Q = gram_schmidt(A)
        print("\nOrthonormal Matrix Q:\n", Q)
        print("\nCheck (Q.T @ Q):\n", Q.T @ Q)
    except ValueError as e:
        print("\nFailed:", e)
    print("-" * 30)

    B = np.array([
        [1, 2],
        [0, 3],
        [2, 1]
    ], dtype=float)

    print("\n--- Example 2 (3x2) ---")
    print("Original Matrix V:\n", B)

    try:
        Q_B = gram_schmidt(B)
        print("\nOrthonormal Matrix Q:\n", Q_B)
        print("\nCheck (Q.T @ Q):\n", Q_B.T @ Q_B)
    except ValueError as e:
        print("\nFailed:", e)
    print("-" * 30)

    C = np.array([
        [1, 2],
        [1, 2]
    ], dtype=float)

    print("\n--- Example 3 (Linearly Dependent) ---")
    print("Original Matrix V:\n", C)

    try:
        Q_C = gram_schmidt(C)
        print("\nOrthonormal Matrix Q:\n", Q_C)
    except ValueError as e:
        print("\nFailed as expected:\n", e)
    print("-" * 30)
