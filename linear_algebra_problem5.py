import numpy as np
from scipy.linalg import null_space
import sympy as sp

print("PROBLEM 5: Linear Algebra Solution")
print("=" * 50)

# Define the vectors that span W
v1 = np.array([1, 0, 3, 0])
v2 = np.array([0, 1, 4, -1])
v3 = np.array([2, -3, -6, 3])
v4 = np.array([4, -1, 8, 2])

# Create matrix A with these vectors as columns
A = np.column_stack([v1, v2, v3, v4])
print("Matrix A (columns are the spanning vectors):")
print(A)
print()

# Part (a): Find a basis for W
print("PART (a): Finding a basis for W")
print("-" * 30)

# Use row reduction to find linearly independent columns
# We'll use sympy for exact rational arithmetic
A_sympy = sp.Matrix(A)
print("Matrix A:")
sp.pprint(A_sympy)
print()

# Get reduced row echelon form
rref_A, pivot_cols = A_sympy.rref()
print("Reduced Row Echelon Form of A:")
sp.pprint(rref_A)
print(f"Pivot columns: {pivot_cols}")
print()

# The basis for W consists of the original columns corresponding to pivot columns
basis_W = []
for col_idx in pivot_cols:
    basis_W.append(A[:, col_idx])

print("Basis for W:")
for i, vec in enumerate(basis_W):
    print(f"v{i+1} = {vec}")
print(f"Dimension of W: {len(basis_W)}")
print()

# Part (b): Find a basis for W⊥ (W perpendicular)
print("PART (b): Finding a basis for W⊥")
print("-" * 30)

# W⊥ is the null space of A^T
# Find null space of A^T
A_T = A.T
null_space_basis = null_space(A_T)

print("A^T (transpose of A):")
print(A_T)
print()

print("Basis for W⊥ (null space of A^T):")
if null_space_basis.size > 0:
    for i in range(null_space_basis.shape[1]):
        vec = null_space_basis[:, i]
        print(f"u{i+1} = {vec}")
    print(f"Dimension of W⊥: {null_space_basis.shape[1]}")
else:
    print("W⊥ = {0} (only the zero vector)")
    print("Dimension of W⊥: 0")
print()

# Verify that dim(W) + dim(W⊥) = 4 (dimension of R^4)
dim_W = len(basis_W)
dim_W_perp = null_space_basis.shape[1] if null_space_basis.size > 0 else 0
print(f"Verification: dim(W) + dim(W⊥) = {dim_W} + {dim_W_perp} = {dim_W + dim_W_perp}")
print()

# Part (c): Find all c such that [c, 1, 1, 0]^T lies in W
print("PART (c): Finding all c such that [c, 1, 1, 0]^T lies in W")
print("-" * 50)

# A vector [c, 1, 1, 0]^T lies in W if and only if it can be written as
# a linear combination of the columns of A
# This means we need to solve: A * x = [c, 1, 1, 0]^T for some x

# Set up the augmented matrix [A | b] where b = [c, 1, 1, 0]^T
c = sp.Symbol('c')
b = sp.Matrix([c, 1, 1, 0])

# Create augmented matrix
augmented = A_sympy.row_join(b)
print("Augmented matrix [A | b] where b = [c, 1, 1, 0]^T:")
sp.pprint(augmented)
print()

# Get RREF of augmented matrix
rref_augmented, pivot_cols_aug = augmented.rref()
print("RREF of augmented matrix:")
sp.pprint(rref_augmented)
print()

# For the system to be consistent, we need to check if there are any
# contradictory equations (rows of the form [0 0 0 0 | non-zero])
print("Analysis for consistency:")
print("For [c, 1, 1, 0]^T to lie in W, the system Ax = b must be consistent.")
print()

# Let's solve this more systematically
# We know that [c, 1, 1, 0]^T ∈ W if and only if [c, 1, 1, 0]^T ⊥ every vector in W⊥
print("Alternative approach using W⊥:")
print("A vector lies in W if and only if it's orthogonal to every vector in W⊥")
print()

if null_space_basis.size > 0:
    print("Conditions for [c, 1, 1, 0]^T to be in W:")
    target_vector = sp.Matrix([c, 1, 1, 0])
    
    conditions = []
    for i in range(null_space_basis.shape[1]):
        null_vec = sp.Matrix(null_space_basis[:, i])
        # Convert to rational for exact computation
        null_vec_rational = sp.Matrix([sp.Rational(float(x)).limit_denominator() for x in null_vec])
        
        dot_product = target_vector.dot(null_vec_rational)
        print(f"Orthogonality condition {i+1}: {target_vector.T} · {null_vec_rational.T} = 0")
        print(f"This gives us: {dot_product} = 0")
        
        # Solve for c
        solution_c = sp.solve(dot_product, c)
        conditions.append(solution_c)
        print(f"Therefore: c = {solution_c}")
        print()
    
    # Find intersection of all conditions
    if len(conditions) > 1:
        common_solutions = set(conditions[0])
        for condition in conditions[1:]:
            common_solutions = common_solutions.intersection(set(condition))
        print(f"All conditions must be satisfied simultaneously.")
        print(f"Final answer: c ∈ {common_solutions if common_solutions else 'No solution'}")
    else:
        print(f"Final answer: c = {conditions[0]}")
else:
    print("Since W⊥ = {0}, every vector in R^4 lies in W.")
    print("Therefore, [c, 1, 1, 0]^T lies in W for all values of c.")

print("\n" + "=" * 50)
print("SOLUTION SUMMARY")
print("=" * 50)
print(f"(a) Basis for W has {len(basis_W)} vectors")
print(f"(b) Basis for W⊥ has {dim_W_perp} vectors") 
print("(c) Values of c will be determined from the orthogonality conditions above")