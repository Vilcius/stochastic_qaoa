#!/usr/bin/env python
# coding: utf-8
# file: squbo_qaoa.py
# author: Anthony Wilkie

# TODO: figure out why QAOA no longer give correct solutions
# TODO: implement the constraint that E(s,x_i) = E(s,x_j)
# TODO: look for better way to implement basis encoding (W state)

# %% Imports
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt


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

def print_obj_vals(n_wires, n_scenarios):
    for i in range(n_wires):
        for ii in range(n_wires):
            x = np.array([i,ii])
            print(f'x = {x},', end=' ')
            obj_c = 0
            for s in range(n_scenarios):
                obj_s = np.dot(x,np.dot(quadratic,x)) + np.dot(x,linear[s])
                obj_c = obj_c + obj_s
                print(f'obj_{s} = {obj_s},', end=' ')
            print(f'obj_c = {obj_c}')

print_obj_vals(np.shape(linear)[1], np.shape(linear)[0])


# %% Create Hamiltonian from Matrix
def get_hvals(linear, quadratic): 
    # Gets the Ising coefficients, NOT the 2^n by 2^n Ising Hamiltonian.
    # Reparameterizes the QUBO problem into {+1, -1}
    # Reparameterization done using x=(Z+1)/2
    # Minimize z.J.z + z.h + offset
    nvars = len(linear)
    jmat = np.zeros(shape=(nvars,nvars))
    hvec = np.zeros(nvars)
    
    for i in range(nvars):
        # the coefficients for the linear terms
        hvec[i] = hvec[i] + (1/2 * linear[i]
                             + 1/4 * sum([quadratic[k][i] for k in range(i)])
                             + 1/4 * sum([quadratic[i][l] for l in range(i+1,nvars)]))

        for j in range(i+1, nvars):
            # the coefficients for the quadratic terms 
            jmat[i][j] = jmat[i][j] + quadratic[i][j]/4
    
    # Gives the correct offset value to the CF
    offset = (np.sum(quadratic)/4 + np.sum(linear)/2)
    return jmat, hvec, offset

def get_qml_ising(linear, quadratic):
    # Function to obtain qml.Hamiltonian from Ising coefficients
    nq = len(linear)
    jj, hh, oo = get_hvals(linear, quadratic)
    h_coeff = []
    h_obs = []

    for i in range(nq):
        h_coeff.append(hh[i])
        h_obs.append(qml.PauliZ(i))
        for j in range(i+1,nq):
            h_coeff.append(jj[i,j])
            h_obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    h_coeff.append(oo)
    h_coeff = np.array(h_coeff)
    h_obs.append(qml.Identity(0))
    hamiltonian = qml.Hamiltonian(h_coeff, h_obs)
    return hamiltonian

C = [get_qml_ising(linear[i], quadratic) for i in range(len(linear))]
for i in range(len(linear)):
    print('The Hamiltonians in each scenario')
    print(f'H_{i} =\n{C[i]}\n')

# combined
print('The combined Hamiltonian')
print(f'H_c =\n{sum(C)}\n')


# %% bitstring
def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

def to_ctrl_val(i, n):
    i = f'{i:0{n}b}'
    ctrl_val = []
    for s in i[::-1]:
        ctrl_val.append(int(s))
    return ctrl_val

for i in range(len(linear)):
    print(to_ctrl_val(2**i, len(linear)))
    print(to_ctrl_val(i,int(np.ceil(np.log2(len(linear))))))


# %% Wires
n_scenarios, n_wires = np.shape(linear)

def create_wires(encoding='basis'):
    r"""
    Create the variable and scenario wires based on the type of encoding used

    Args:
        encoding (data type): TODO

    Returns:
        var (list): the variable wires
        scen (list): the scenario wires
    """
    var = [f'x{i}' for i in range(n_wires)]

    if encoding == 'basis':
        scen = [f's{i}' for i in range(n_scenarios)]
    elif encoding == 'amplitude':
        scen = [f's{i}' for i in range(int(np.ceil(np.log2(n_scenarios))))]

    return var, scen

encoding = 'basis'
# encoding = 'amplitude'

shots = 1000
var, scen = create_wires(encoding)
sdev = qml.device("default.qubit", wires=var+scen, shots=shots)

print(sdev.wires)
print(dict(enumerate(var)))


# %% Initalize Scenario Qubits
def basis_encoding(probs):
    r"""
    Initializes the scenario qubits using Basis Encoding, in the form of the W state.

    Args:
        probabilities (np.array): The probability of being in each scenario
    """

    w_state = np.zeros(2**n_scenarios)
    for i in range(n_scenarios):
        w_state[2**i] = w_state[2**i] +  probs[i]
    qml.MottonenStatePreparation(w_state,wires=scen)


# %% Gates
# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in var:
        qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
def U_C(C, gamma, scen, encoding='basis'):
    for s in range(n_scenarios):
        Cs = C[s].map_wires(dict(enumerate(var)))
        UCs = qml.CommutingEvolution(Cs,gamma)

        if encoding == 'basis':
            qml.ctrl(UCs,
                     control=scen,
                     control_values=to_ctrl_val(2**s,n_scenarios)
                     )
        elif encoding == 'amplitude':
            qml.ctrl(UCs,
                     control=scen,
                     control_values=to_ctrl_val(s,int(np.ceil(np.log2(n_scenarios))))
                     )


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
        basis_encoding(normalized)
    elif encoding == 'amplitude':
        # use Amplitude Embedding
        qml.AmplitudeEmbedding(normalized, scen, pad_with=0.0)

    
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(C,gammas[i],scen, encoding)
        U_B(betas[i])

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


# %% Parse scenraio bitstring
def parse_scenraio(bitstrings, encoding='basis'):
    parsed = [{} for _ in range(n_scenarios)]
    for i in bitstrings.keys():
        var_bits = i[:n_wires]
        scen_bits = i[n_wires:]
        print(i, var_bits, scen_bits)
        if encoding == 'basis':
            if scen_bits.count('1') == 1:
                print(scen_bits)
                idx = scen_bits.index('1')
                print(idx)
                parsed[idx][var_bits] = bitstrings[i]/shots
        elif encoding == 'amplitude':
            idx = bitstring_to_int(scen_bits)
            parsed[idx][var_bits] = bitstrings[i]/shots
    return parsed


# %% Plot
import matplotlib.pyplot as plt
barcolors = ['#286d8c', '#a99b63', '#936846', '#4d7888']

def graph(bitstrings, beamer, p, encoding):

    if beamer:
        plt.figure(figsize=(16, 8))
        x = np.arange(2**n_wires)  # the label locations
        width = 0.8  # the width of the bars

        fig, ax = plt.subplots()
        s = [ax.bar(x - width/n_scenarios + i/n_scenarios * width, bitstrings[i].values(),
                    align='edge',
                    width=width/n_scenarios,
                    label=f'scenario {i}',
                    edgecolor = "#041017",
                    color=barcolors[i])
             for i in range(n_scenarios)]
        ax.set_xticks(x, bitstrings[0].keys())

        plt.rc('font', size=16)
        plt.rc('axes', edgecolor='#98c9d3', labelcolor='#98c9d3', titlecolor='#98c9d3', facecolor='#041017')
        plt.rc('figure', facecolor='#041017')
        plt.rc('savefig', facecolor='#041017')
        plt.rc('xtick',color='#98c9d3')
        plt.rc('ytick',color='#98c9d3')
        plt.rc('legend',labelcolor='#98c9d3', edgecolor='#98c9d3',facecolor=(0,0,0,0))
        plt.title(f"Stochastic QUBO with {encoding} encoding using {p}-QAOA")
        plt.xlabel("Solutions")
        plt.ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        plt.savefig(f'stochastic_qubo_{encoding}_p={p}_beamer.pdf',
                   transparent=True)
    else:
        plt.rcdefaults()
        plt.figure(figsize=(16, 8))
        x = np.arange(2**n_wires)  # the label locations
        width = 0.8  # the width of the bars

        fig, ax = plt.subplots()
        s = [ax.bar(x - width/n_scenarios + i/n_scenarios * width, bitstrings[i].values(),
                    align='edge',
                    width=width/n_scenarios,
                    label=f'scenario {i}',
                    edgecolor = "#041017",
                    color=barcolors[i])
             for i in range(n_scenarios)]
        ax.set_xticks(x, bitstrings[0].keys())

        plt.rc('font', size=16)
        plt.title(f"Stochastic QUBO with {encoding} encoding using {p}-QAOA")
        plt.rc('font', size=16)
        plt.xlabel("Solutions")
        plt.rc('font', size=16)
        plt.ylabel("Frequency")

        ax.legend()
        fig.tight_layout()
        plt.savefig(f'stochastic_qubo_{encoding}_p={p}.pdf',
                   transparent=True)

    plt.show()


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
