# Alternating Direction Method of Multipliers (ADMM)
## Sparce Logistic Regression (ADMM)

<br>

**Objective Function:**

The objective function is designed to minimize the logistic regression loss (cross-entropy) while incorporating an L1 regularization term for sparsity. It aims to find a set of coefficients `x` that can predict the target variable `b` from the input data `A` while penalizing large coefficients to encourage sparsity. The logistic regression loss `l` is computed for each data point and then averaged.

$minimize \; (1/2)||\sigma(A x) - b||_2^{2} + \lambda ||x||_1$

where $\sigma$ is the sigmoid fuction:

$\sigma(z) = \frac{exp(z)}{1+exp(z)} = \frac{1}{1+exp(-z)}$

<br>

**Centralized Case**:
<br>Loss minimization<br>
$ minimize \sum{l_i(A_i x_i - b_i) + r(z)}$,
subject to $x_i − z = 0, i = 1, . . . , N$

<br>

The ADMM algorithm alternates between three steps:

1. **Update `x`:** represents the model coefficients.This step minimizes the augmented Lagrangian with respect to `x`, where the augmented Lagrangian combines the logistic regression loss, the L2 regularization term, and a penalty term associated with the difference between `x` and `z`. It aims to update `x` to minimize the loss while encouraging sparsity.

2. **Update `z`:** is an auxiliary variable used in the update.`z` is updated using the `S` operator with a shrinkage threshold `λ/ρ`, which encourages sparsity. This step enforces the sparsity constraint on the model coefficients.

3. **Update `u`:** `u` is the dual variable and is updated based on the difference between `x` and `z`. It is used to enforce the equality constraint between `x` and `z`.
<br>

$x^{k+1} = argmin  (l||A x - b||_2^{2} +(\rho/2)||x - z^k + u^{k}||_2^{2} )$

$z^{k+1} = S_{λ/ρ}(x^{k+1} + u^k)$

$u^{k+1} = u^{k} + x^{k+1} - z^{k+1}$

<br>

where $l$ is the *logistic regression* cost fuction or more known as *cross-entropy*:

$ l = (-1/N)\sum{b \log(\sigma(A x)) + (1-b) \log(1 - \sigma(A x))}$

or in a more compact form:

$ l = (1/N) \sum{\log(1+exp(-b x^{T}A))} $



<br>


**Consensus form of the model**:

<br>


The formulation can also be expressed in a consensus form, where it's assumed that the data is distributed across multiple blocks. Each block is indexed by `i`, and the objective is to find a common solution `z` while taking into account the local loss functions `l_i` associated with each block.

In this consensus form, the algorithm iteratively updates `x_i` for each block `i` (similar to the centralized case), and then computes an average of the updated `x_i` values to obtain `z`. The dual variables `u_i` for each block are also updated.

This distributed version allows you to parallelize the computation across multiple blocks of data, making it suitable for situations where data is distributed across different locations or nodes.
<br>
<br>
$ minimize \sum{l_i(A_i x_i - b_i) + r(z)}$, subject to $x_i − z = 0, i = 1, . . . , N$

where $l_i$ refers to the loss function for the ith block of
data.

<br>

Consensus ADMM algorithm for Sparse Logistic Regression problem:

$x_i^{k+1} = argmin  ( l_i||A_i x_i - b_i||_2^{2} +(\rho/2)||x_i - z^k + u_i^{k}||_2^{2} )$

$z^{k+1} = S_{λ/ρN}(\overline{x}^{k+1} + \overline{u}^k)$

$u_i^{k+1} = u_i^{k} + x_i^{k+1} - z^{k+1}$

<br>

This is identical to distributed lasso, except for $x_i$ update, which here involves an $l_2$ regularized
logistic regression problem.


