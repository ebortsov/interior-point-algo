import numpy as np
import numpy.typing as npt
import argparse
import sys
from pathlib import Path


def maximize_with_interior_point(
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    alpha: float,
    eps: float = 10**-6,
) -> np.ndarray:
    # Ensure that x0 is indeed the solution
    if np.max(np.absolute((A @ x) - b)) > eps or np.min(x) < -eps:
        raise ValueError("x0 is not a solution")
    
    # Potential improvements: 
    # 1) Check that A has independent rows

    m, n = A.shape
    while True:
        # Step 1: construct the diagonal matrix D
        D = np.diag(x)

        # Step 2: calculate the projection matrix P onto the column space of A1
        A1 = A @ D
        C1 = D @ C
        A1_T = A1.transpose()
        P = np.identity(n) - A1_T @ np.linalg.inv(A1 @ A1_T) @ A1

        # Step 3: project the gradient onto the column space of A1
        C1_p = P @ C1

        # Step 4: find the smallest component of C1_p having the largest absolute value
        v = np.min(C1_p)

        if v > 0:
            print("The method is not applicable!")
            raise ValueError("The method is not applicable")
        
        # Step 5: calculate x1
        x1 = np.ones((n)) + alpha / abs(v) * C1_p

        # Step 6: calculate new trial solution
        x_new = D @ x1


        # Step 7: check the change of the solution
        if np.linalg.norm(x_new - x, ord=2) < eps:
            break

        x, x_new = x_new, x

    return x_new
    

def setup_io():
    # Handle redirecting io to files
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        help="path to input file (stdin if not specified)",
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        help="path to output file (stdout if not specified)",
    )

    args = parser.parse_args()
    if args.inputfile is not None:
        input_file = Path(args.inputfile)
        sys.stdin = input_file.open(mode="r", encoding="utf-8")
    if args.outputfile is not None:
        output_file = Path(args.outputfile)
        sys.stdout = output_file.open(mode="w", encoding="utf-8")


def main():
    setup_io()

    # n is the number of variables, m is the number of constraints
    n, m = list(map(int, input().split()))

    # A vector of coefficients of objective function
    C = np.array(list(map(float, input().split())))

    # A matrix of coefficients of constraint functions
    A = np.array(list(list(map(float, input().split())) for _ in range(m)))

    # A vector of rhs numbers of constraints
    b = np.array(list(map(float, input().split())))

    # # Initial solution
    x0 = np.array(list(map(float, input().split())))

    # # Approximation accuracy
    eps = float(input())

    # Controlling coefficients
    alpha1 = 0.5
    alpha2 = 0.9

    try:
        x1 = maximize_with_interior_point(C, A, b, x0, alpha1, eps)
        x2 = maximize_with_interior_point(C, A, b, x0, alpha2, eps)
        np.set_printoptions(precision=7, suppress=True)
        print(f"alpha = {alpha1:.2f}: {x1}")
        print(f"alpha = {alpha2:.2f}: {x2}")
    except Exception as e:
        sys.stderr.write(str(e) + '\n')


if __name__ == "__main__":
    main()
