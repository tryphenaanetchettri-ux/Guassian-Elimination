import numpy as np

def gaussian_elimination(matrix):
    """
    Performs Gaussian elimination on an augmented matrix to bring it to
    Row Echelon Form (REF).

    Args:
        matrix (np.ndarray): The augmented matrix (M x N).

    Returns:
        np.ndarray: The matrix in Row Echelon Form.
    """

    A = matrix.astype(float).copy()
    num_rows, num_cols = A.shape

    pivot_row = 0
    pivot_col = 0

    while pivot_row < num_rows and pivot_col < num_cols:

        max_row_idx = np.argmax(np.abs(A[pivot_row:, pivot_col])) + pivot_row

        if max_row_idx != pivot_row:
            A[[pivot_row, max_row_idx]] = A[[max_row_idx, pivot_row]]

        pivot_element = A[pivot_row, pivot_col]

        if np.isclose(pivot_element, 0):
            pivot_col += 1
            continue

        for i in range(pivot_row + 1, num_rows):
            factor = A[i, pivot_col] / pivot_element
            A[i, pivot_col:] = A[i, pivot_col:] - factor * A[pivot_row, pivot_col:]

        pivot_row += 1
        pivot_col += 1

    return A

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    A1 = np.array([
        [2,  1, -1,  8],
        [-3, -1,  2, -11],
        [-2,  1,  2, -3]
    ], dtype=float)

    print("--- Example 1 ---")
    print("Original Matrix:\n", A1)
    ref1 = gaussian_elimination(A1)
    print("\nRow Echelon Form:\n", ref1)
    print("-" * 20)

    A2 = np.array([
        [0, 2, 3, 4],
        [1, 1, -1, 0],
        [2, 4, -5, 2]
    ], dtype=float)

    print("--- Example 2 (Needs Pivoting) ---")
    print("Original Matrix:\n", A2)
    ref2 = gaussian_elimination(A2)
    print("\nRow Echelon Form:\n", ref2)
    print("-" * 20)

    A3 = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [1, 1, 1, 1]
    ], dtype=float)

    print("--- Example 3 (Dependent System) ---")
    print("Original Matrix:\n", A3)
    ref3 = gaussian_elimination(A3)
    print("\nRow Echelon Form:\n", ref3)
    print("-" * 20)
