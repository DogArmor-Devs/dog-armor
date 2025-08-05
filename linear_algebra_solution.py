import sympy as sp
from sympy import Matrix, pprint, Rational

print("PROBLEM 5: Linear Algebra Solution")
print("=" * 60)

# Define the vectors that span W using exact rational arithmetic
v1 = Matrix([1, 0, 3, 0])
v2 = Matrix([0, 1, 4, -1])
v3 = Matrix([2, -3, -6, 3])
v4 = Matrix([4, -1, 8, 2])

# Create matrix A with these vectors as columns
A = Matrix.hstack(v1, v2, v3, v4)
print("Given vectors that span W:")
print(f"v1 = {v1.T}")
print(f"v2 = {v2.T}")
print(f"v3 = {v3.T}")
print(f"v4 = {v4.T}")
print()

print("Matrix A (columns are the spanning vectors):")
pprint(A)
print()

# Part (a): Find a basis for W
print("PART (a): Finding a basis for W")
print("-" * 40)

# Get reduced row echelon form to find linearly independent columns
rref_A, pivot_cols = A.rref()
print("Reduced Row Echelon Form of A:")
pprint(rref_A)
print(f"Pivot columns: {pivot_cols}")
print()

# Extract basis vectors
basis_W = []
original_vectors = [v1, v2, v3, v4]
for col_idx in pivot_cols:
    basis_W.append(original_vectors[col_idx])

print("Basis for W:")
for i, vec in enumerate(basis_W):
    print(f"w{i+1} = {vec.T}")
print(f"Dimension of W: {len(basis_W)}")
print()

# Part (b): Find a basis for W⊥ (W perpendicular)
print("PART (b): Finding a basis for W⊥")
print("-" * 40)

# W⊥ is the null space of A^T
A_T = A.T
print("A^T (transpose of A):")
pprint(A_T)
print()

# Find null space of A^T
null_space_basis = A_T.nullspace()
print("Basis for W⊥ (null space of A^T):")
if null_space_basis:
    for i, vec in enumerate(null_space_basis):
        print(f"u{i+1} = {vec.T}")
    print(f"Dimension of W⊥: {len(null_space_basis)}")
else:
    print("W⊥ = {0} (only the zero vector)")
    print("Dimension of W⊥: 0")
print()

# Verify dimension theorem
dim_W = len(basis_W)
dim_W_perp = len(null_space_basis) if null_space_basis else 0
print(f"Verification: dim(W) + dim(W⊥) = {dim_W} + {dim_W_perp} = {dim_W + dim_W_perp}")
print("This should equal 4 (dimension of R^4) ✓")
print()

# Part (c): Find all c such that [c, 1, 1, 0]^T lies in W
print("PART (c): Finding all c such that [c, 1, 1, 0]^T lies in W")
print("-" * 60)

c = sp.Symbol('c')
target_vector = Matrix([c, 1, 1, 0])

print("Method: A vector lies in W if and only if it's orthogonal to every vector in W⊥")
print()

if null_space_basis:
    print("For [c, 1, 1, 0]^T to be in W, it must be orthogonal to all vectors in W⊥:")
    
    conditions = []
    for i, null_vec in enumerate(null_space_basis):
        dot_product = target_vector.dot(null_vec)
        print(f"Condition {i+1}: [c, 1, 1, 0] · {null_vec.T} = 0")
        print(f"This gives us: {dot_product} = 0")
        
        # Solve for c
        solutions = sp.solve(dot_product, c)
        conditions.extend(solutions)
        print(f"Therefore: c = {solutions}")
        print()
    
    if len(conditions) == 1:
        print(f"FINAL ANSWER: c = {conditions[0]}")
    elif len(conditions) > 1:
        # Find common solutions
        common_solutions = set(conditions)
        print(f"FINAL ANSWER: c must satisfy all conditions simultaneously")
        print(f"c ∈ {common_solutions}")
    else:
        print("No solution exists")
else:
    print("Since W⊥ = {0}, every vector in R^4 lies in W.")
    print("FINAL ANSWER: c can be any real number")

print()

# Alternative verification using augmented matrix
print("Verification using augmented matrix method:")
print("-" * 45)
b = Matrix([c, 1, 1, 0])
augmented = A.row_join(b)
print("Augmented matrix [A | b]:")
pprint(augmented)
print()

rref_aug, pivot_cols_aug = augmented.rref()
print("RREF of augmented matrix:")
pprint(rref_aug)
print()

# Check for inconsistency
print("Analysis: Looking for rows of the form [0 0 0 0 | non-zero]")
n_rows, n_cols = rref_aug.shape
inconsistent_rows = []
for i in range(n_rows):
    row = rref_aug.row(i)
    if all(row[j] == 0 for j in range(n_cols-1)) and row[n_cols-1] != 0:
        inconsistent_rows.append(i)
        print(f"Row {i+1}: {row} indicates inconsistency")

if inconsistent_rows:
    print("The system is inconsistent for the given value of c")
else:
    print("The system can be consistent for appropriate values of c")

print("\n" + "=" * 60)
print("COMPLETE SOLUTION SUMMARY")
print("=" * 60)
print(f"(a) Basis for W: {len(basis_W)} vectors")
for i, vec in enumerate(basis_W):
    print(f"    w{i+1} = {vec.T}")
print()
print(f"(b) Basis for W⊥: {len(null_space_basis)} vector(s)")
if null_space_basis:
    for i, vec in enumerate(null_space_basis):
        print(f"    u{i+1} = {vec.T}")
print()
print("(c) Values of c such that [c, 1, 1, 0]^T ∈ W:")
if null_space_basis and conditions:
    print(f"    c = {conditions[0]}")
    # Simplify the fraction if possible
    if isinstance(conditions[0], sp.Rational):
        print(f"    c = {float(conditions[0]):.10f} (decimal approximation)")
else:
    print("    c can be any real number")