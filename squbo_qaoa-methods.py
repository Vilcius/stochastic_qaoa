#!/usr/bin/env python
# coding: utf-8
# file: squbo_qaoa-methods.py
# author: Anthony Wilkie


# %% Imports
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt


# %% Print Objective Values
def print_obj_vals(n_wires, n_scenarios, quadratic, linear):
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


# %% Create Wires based on Encoding
def create_wires(n_wires, n_scenarios, encoding='basis'):
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


# %% Initalize Scenario Qubits
def basis_encoding(n_scenarios, scen, probs):
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
def U_B(beta, var):
    for wire in var:
        qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
def U_C(C, gamma, var, scen, n_scenarios, encoding='basis'):
    for s in range(n_scenarios):
        Cs = C[s].map_wires(dict(enumerate(var)))
        UCs = qml.CommutingEvolution(Cs,gamma)

        if encoding == 'basis':
            qml.ctrl(UCs,
                     control=scen[s]
                     )
        elif encoding == 'amplitude':
            qml.ctrl(UCs,
                     control=scen,
                     control_values=to_ctrl_val(s,int(np.ceil(np.log2(n_scenarios))))
                     )


# %% Parse scenraio bitstring
def parse_scenraio(bitstrings, n_wires, n_scenarios, encoding='basis'):
    parsed = [{} for _ in range(n_scenarios)]
    for i in bitstrings.keys():
        var_bits = i[:n_wires]
        scen_bits = i[n_wires:]
        if encoding == 'basis':
            if scen_bits.count('1') == 1:
                idx = scen_bits.index('1')
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




