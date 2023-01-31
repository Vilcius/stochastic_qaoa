#!/usr/bin/env python
# coding: utf-8
# file: squbo_qaoa.py
# author: Anthony Wilkie

# TODO: Figure out way to optimize lambdas
# TODO: look for better way to implement basis encoding (W state)

# %% Imports
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import squbo_qaoa-methods as mthd


# %% Create QUBO Problem Matrix
linear = np.array(
        [[1.0, -3.0],
         [1.0, 0.0]#,
         # [1.0, 5.0],
         # [2.0, 5.0]#,
         ])
quadratic = np.array(
        [[0.0, 2.0],
         [0.0, 0.0]])

print(np.shape(linear))
Q = []
for i in range(len(linear)):
    q = np.copy(quadratic)
    for ii in range(len(linear[i])):
        q[ii][ii] = q[ii][ii] + linear[i][ii]
    Q.append(q)

for i in range(len(linear)):
    print('The QUBO matrices in each scenario')
    print(f'Q_{i} =\n{Q[i]}\n')

# combined
print('The combined QUBO matrix')
print(f'Q_c =\n{sum(Q)}\n')

mthd.print_obj_vals(np.shape(linear)[1], np.shape(linear)[0])


# %% Create Hamiltonian from Matrix
C = [mthd.get_qml_ising(linear[i], quadratic) for i in range(len(linear))]
for i in range(len(linear)):
    print('The Hamiltonians in each scenario')
    print(f'H_{i} =\n{C[i]}\n')

# combined
print('The combined Hamiltonian')
print(f'H_c =\n{sum(C)}\n')


# %% Wires
n_scenarios, n_wires = np.shape(linear)

encoding = 'basis'
# encoding = 'amplitude'

shots = 1000
var, scen = mthd.create_wires(n_wires, n_scenarios, encoding)
sdev = qml.device("default.qubit", wires=var+scen, shots=shots)


# %% Circuit
@qml.qnode(sdev)
def circuit(gammas, betas, encoding='basis', stage=1, xi = None, counts=False, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for x in var:
        qml.Hadamard(wires=x)
        
    # initialize the scenario qubits
    probs = np.array([1/n_scenarios for _ in range(n_scenarios)])
    normalized = probs / np.linalg.norm(probs)

    if encoding == 'basis':
        # use Basis Encoding
        mthd.basis_encoding(n_scenarios, scen, normalized)
    elif encoding == 'amplitude':
        # use Amplitude Embedding
        qml.AmplitudeEmbedding(normalized, scen, pad_with=0.0)

    
    # p instances of unitary operators
    for i in range(n_layers):
        mthd.U_C(C,gammas[i], var, scen, n_scenarios, encoding)
        mthd.U_B(betas[i], var)

    # return samples instead of expectation value
    if counts:
        # measurement phase
        return qml.counts(all_outcomes=True)

    if stage == 1:
        # Currently use the sum of the Cost Hamiltonians.
        H = sum(C)
        return qml.expval(H.map_wires(dict(enumerate(var))))
    elif stage == 2:
        return [qml.expval(qml.PauliZ(wires=xi) @ (qml.PauliZ(wires=f's{u}') - qml.PauliZ(wires=f's{v}'))) for u in range(n_scenarios-1) for v in range(u+1, n_scenarios)]


# %% Optimization
# np.random.seed(1248)

def sqaoa(n_layers=1, encoding='basis' ):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)
    lambs = [1 for i in range(n_wires)]

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]

        # Stage 1
        c_expval = circuit(gammas, betas, encoding=encoding, n_layers=n_layers)

        # Stage 2
        xs_expval = []
        for xi in var:
            xs_expval.append(circuit(gammas, betas, encoding=encoding, stage=2, xi=xi, n_layers=n_layers))

        neg_obj = c_expval + sum([lambs[i] * xs_expval[i]**2 for i in range(n_wires)])
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 50
    for i in range(steps):
        # TODO: implement step_and_cost, if needed
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # count measured bitstrings 1000 times
    n_counts=shots
    counts = circuit(params[0], params[1], encoding=encoding, counts=True, n_layers=n_layers)

    # print optimal parameters and most frequently countd bitstring
    most_freq_bit_string = max(counts, key=counts.get)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print(f"Most frequently countd bit string is: {most_freq_bit_string} with probability {counts[most_freq_bit_string]/n_counts:.4f}")

    return params, counts

# %% Run the Thing
p = 1
params, bitstrings = sqaoa(n_layers=p, encoding=encoding)
parsed = parse_scenraio(bitstrings, encoding=encoding)
graph(parsed, p=p, encoding=encoding, beamer=True)
print_obj_vals(n_wires, n_scenarios)


# %% graphs
# params, bitstrings = sqaoa(n_layers=1)
# bitstrings = parse_scenraio(bitstrings)
print(bitstrings)
parsed = parse_scenraio(bitstrings, encoding=encoding)
print(parsed)
graph(parsed, p=p, encoding=encoding, beamer=True)


# %% Draw circuit
print(qml.draw_mpl(circuit, style='default')(params[0],params[1],encoding))
