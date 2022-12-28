# Solving Stochastic QUBOs Using QAOA #
The code here tries to solve a QUBO containing uncertainty in the linear term:
$$
\text{max } x^T Q x + c^T x,
$$
where $c$ could have any value $c_1, c_2, \ldots, c_n$ with probability $p_1, p_2, \ldots, p_n$.
We do this by splitting the problem into different scenarios $s_1, s_2, \ldots, s_n$:
$$
    \text{max } x^T Q x + c_i^T x.
$$
For each scenario $s_i$, we solve the corresponding QUBO with QAOA on variable qubits corresponding to $x$, where we control the corresponding cost unitary $U(C_i, \gamma)$ is controlled by a scenario qubit $s_i$.
We initialize the scenario qubits into the *W state*:
$$
    W = \frac{1}{\sqrt{2^n}} \sum_{i=1}^n |0\rangle^{\otimes i-1} |1\rangle |0\rangle^{\otimes n-i-1},
$$
where a 1 indicates which scenario we are in.

