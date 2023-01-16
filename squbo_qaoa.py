#!/usr/bin/env python
# coding: utf-8
# file: squbo_qaoa.py
# author: Anthony Wilkie

# TODO: figure out why QAOA no longer give correct solutions

# %% Imports
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt


# %% Create QUBO Problem Matrix
linear = np.array(
        [[1.0, -3.0],
         [1.0, 5.0],
         [1.0, 0.0]])
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

for i in range(3):
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

C = [get_qml_ising(linear[i], quadratic) for i in range(3)]
for i in range(3):
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

for i in range(3):
    print(to_ctrl_val(2**i, 3))
    print(to_ctrl_val(i,int(np.ceil(np.log2(3)))))


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

var, scen = create_wires('amplitude')

sdev = qml.device("default.qubit", wires=var+scen, shots=1)
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
def circuit(gammas, betas, encoding='basis', sample=False, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for x in var:
        qml.Hadamard(wires=x)
        
    # initialize the scenario qubits
    probs = np.array([1/n_scenarios,1/n_scenarios,1/n_scenarios])
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
    if sample:
        # measurement phase
        return qml.sample()
    
    # during the optimization phase we are evaluating the objective using expval
    # use Hamiltonian corresponding to scenario
    # DOES NOT WORK FOR MORE THAN 2 SCENARIOS
    #s = qml.measure(scen)
    #if s == 0:
    #    H = H_0
    #else:
    #    H = H_1
    
    # Currently use the sum of the Cost Hamiltonians.
    H = sum(C)
    return qml.expval(H.map_wires(dict(enumerate(var))))


# %% Optimization
# np.random.seed(1248)

def sqaoa(n_layers=1, encoding='basis' ):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = circuit(gammas, betas, encoding=encoding, n_layers=n_layers)
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 50
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 1000
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], encoding=encoding, sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:03b} with probability {:.4f}".format(most_freq_bit_string,counts[most_freq_bit_string]/n_samples))

    return params, bit_strings


# %% Parse scenraio bitstring
def parse_scenraio(bitstrings, encoding='basis'):
    parsed = [[] for _ in range(n_scenarios)]
    for i in bitstrings:
        var_bits = bitstring_to_int(f'{i:0{n_wires+len(scen)}b}'[:n_wires])
        scen_bits = f'{i:0{n_wires+len(scen)}b}'[n_wires:]
        if encoding == 'basis':
            idx = scen_bits.index('1')
        elif encoding == 'amplitude':
            idx = bitstring_to_int(scen_bits)
        parsed[idx].append(var_bits)
    return parsed


# %% Plot
import matplotlib.pyplot as plt
barcolors = ['#286d8c', '#a99b63', '#936846', '#4d7888']

def graph(bitstrings, beamer):

    if beamer:
        xticks = range(0, 2**(n_wires))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires}b"), xticks))
        bins = np.arange(0, 2**(n_wires)+1) - 0.5

        plt.figure(figsize=(16, 8))
        plt.rc('font', size=16)
        plt.rc('axes', edgecolor='#98c9d3', labelcolor='#98c9d3', titlecolor='#98c9d3', facecolor='#041017')
        plt.rc('figure', facecolor='#041017')
        plt.rc('savefig', facecolor='#041017')
        plt.rc('xtick',color='#98c9d3')
        plt.rc('ytick',color='#98c9d3')
        plt.rc('legend',labelcolor='#98c9d3', edgecolor='#98c9d3',facecolor=(0,0,0,0))
        plt.title("s-QAOA")
        plt.xlabel("Bitstrings")
        plt.ylabel("Frequency")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings,
                 bins=bins,
                 density=True,
                 color=barcolors[:n_scenarios],
                 edgecolor = "#041017",
                 label=[f'scenario {i}' for i in range(n_scenarios)])
        plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1_beamer.pdf',
                   # transparent=True)
    else:
        xticks = range(0, 2**(n_wires))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires}b"), xticks))
        bins = np.arange(0, 2**(n_wires)+1) - 0.5

        plt.figure(figsize=(16, 8))
        plt.rc('font', size=16)
        plt.title("s-QAOA")
        plt.xlabel("Bitstrings")
        plt.ylabel("Frequency")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings,
                 bins=bins,
                 density=True,
                 color=barcolors[:n_scenarios],
                 edgecolor = "#041017",
                 label=[f'scenario {i}' for i in range(n_scenarios)])

        plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1.pdf',
        #            transparent=True)

    plt.show()


# %% Run the Thing
n_scenarios, n_wires = np.shape(linear)
encoding = 'basis'
# encoding = 'amplitude'

var, scen = create_wires(encoding)
sdev = qml.device("default.qubit", wires=var+scen, shots=1)

params, bitstrings = sqaoa(n_layers=1, encoding=encoding)
parsed = parse_scenraio(bitstrings, encoding=encoding)
graph(parsed, beamer=True)
print_obj_vals(n_wires, n_scenarios)


# %% graphs
# params, bitstrings = sqaoa(n_layers=1)
# bitstrings = parse_scenraio(bitstrings)
# graph(bitstrings, beamer=True)

