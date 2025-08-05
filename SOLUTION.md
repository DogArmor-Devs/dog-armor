# Problem 5 Solution

## Given
W = Span{[1, 0, 3, 0]ᵀ, [0, 1, 4, -1]ᵀ, [2, -3, -6, 3]ᵀ, [4, -1, 8, 2]ᵀ}

## Part (a): Find a basis for W

**Method:** Use row reduction to find linearly independent columns of matrix A.

Matrix A (columns are the spanning vectors):
```
⎡1  0   2   4 ⎤
⎢             ⎥
⎢0  1   -3  -1⎥
⎢             ⎥
⎢3  4   -6  8 ⎥
⎢             ⎥
⎣0  -1  3   2 ⎦
```

Reduced Row Echelon Form:
```
⎡1  0  2   0⎤
⎢           ⎥
⎢0  1  -3  0⎥
⎢           ⎥
⎢0  0  0   1⎥
⎢           ⎥
⎣0  0  0   0⎦
```

Pivot columns: (0, 1, 3)

**Answer:** A basis for W consists of:
- w₁ = [1, 0, 3, 0]ᵀ
- w₂ = [0, 1, 4, -1]ᵀ  
- w₃ = [4, -1, 8, 2]ᵀ

Dimension of W = 3

## Part (b): Find a basis for W⊥

**Method:** W⊥ is the null space of Aᵀ.

Aᵀ (transpose of A):
```
⎡1  0   3   0 ⎤
⎢             ⎥
⎢0  1   4   -1⎥
⎢             ⎥
⎢2  -3  -6  3 ⎥
⎢             ⎥
⎣4  -1  8   2 ⎦
```

Finding null space of Aᵀ:

**Answer:** A basis for W⊥ consists of:
- u₁ = [-3, -4, 1, 0]ᵀ

Dimension of W⊥ = 1

**Verification:** dim(W) + dim(W⊥) = 3 + 1 = 4 = dim(ℝ⁴) ✓

## Part (c): Find all c such that [c, 1, 1, 0]ᵀ lies in W

**Method:** A vector lies in W if and only if it's orthogonal to every vector in W⊥.

For [c, 1, 1, 0]ᵀ to be in W, it must be orthogonal to u₁ = [-3, -4, 1, 0]ᵀ:

[c, 1, 1, 0] · [-3, -4, 1, 0] = 0

-3c - 4(1) + 1(1) + 0(0) = 0

-3c - 4 + 1 = 0

-3c - 3 = 0

-3c = 3

**Answer:** c = -1

## Final Summary

**(a)** Basis for W: {[1, 0, 3, 0]ᵀ, [0, 1, 4, -1]ᵀ, [4, -1, 8, 2]ᵀ}

**(b)** Basis for W⊥: {[-3, -4, 1, 0]ᵀ}

**(c)** c = -1