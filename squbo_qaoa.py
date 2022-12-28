# %% Imports
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)


# %% Variables
n_wires=2
n_scenarios=3

var = [f'x{i}' for i in range(n_wires)]
scen = [f's{i}' for i in range(n_scenarios)]

# used to remap the wires in the Hamiltonians defined above
wiremap = {}
for i in range(n_wires):
    wiremap[i] = var[i]

sdev = qml.device("default.qubit", wires=var+scen, shots=1)
print(sdev.wires)


# %% create Hamiltonians
x = [1/2 * (qml.Identity(i) - qml.PauliZ(i)) for i in range(2)]
c = [-3, 1, 0]

# create the objective function/cost Hamiltonian based off of the c value
def obj(x0, x1, c):
    if type(x0) == qml.Hamiltonian:
        return 2*x0@x1 + x0 + c*x1
    else:
        return 2*x0*x1 + x0 + c*x1

C = [obj(x[0], x[1], s) for s in c]

for i in range(3):
    print(f'H_{i} =\n{C[i]}\n')

# combined
print(f'H_c =\n{sum(C)}\n')

def print_obj_vals(ss):
    for i in range(2):
        for ii in range(2):
            print(f'x = ({i},{ii}),', end=' ')
            obj_c = 0
            for s in range(ss):
                obj_s = obj(i, ii, c[s])
                obj_c += obj_s
                print(f'obj_{s} = {obj_s},', end=' ')
            print(f'obj_c = {obj_c}')

print_obj_vals(3)


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


# %% Gates
# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in var:
        qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
def U_C(C,gamma,scen):
    for s in range(n_scenarios):
        Cs = C[s].map_wires(wiremap)
        UCs = qml.CommutingEvolution(Cs,gamma)
        qml.ctrl(UCs,
                 control=scen,
                 control_values=to_ctrl_val(2**s,n_scenarios))


# %% Circuit
@qml.qnode(sdev)
def circuit(gammas, betas, sample=False, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for x in var:
        qml.Hadamard(wires=x)
        
    # initialize scenario qubit
    #qml.Hadamard(wires='s0')
    
    # will be useful when having multiple scenarios
    w_state = np.zeros(2**n_scenarios)
    probs = [1/n_scenarios,1/n_scenarios,1/n_scenarios]
    for i in range(n_scenarios):
        w_state[2**i] = np.sqrt(probs[i])
    qml.MottonenStatePreparation(w_state,wires=scen)
    
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(C,gammas[i],scen)
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
    return qml.expval(H.map_wires(wiremap))


# %% Optimization
def sqaoa(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = -circuit(gammas, betas,n_layers=n_layers)
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
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:03b} with probability {:.4f}".format(most_freq_bit_string,counts[most_freq_bit_string]/n_samples))

    return params, bit_strings


# %% Plot
import matplotlib.pyplot as plt

def graph(bitstrings1, beamer):

    if beamer:
        xticks = range(0, 2**(n_wires+n_scenarios))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires+n_scenarios}b"), xticks))
        bins = np.arange(0, 2**(n_wires+n_scenarios)+1) - 0.5

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
        plt.hist([bitstrings1],
                 bins=bins,
                 density=True,
                 color=['#286d8c'],
                 edgecolor = "#041017",
                 label=['s-QAOA'])
        plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1_beamer.pdf',
                   # transparent=True)
    else:
        xticks = range(0, 2**(n_wires+n_scenarios))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires+n_scenarios}b"), xticks))


        plt.figure(figsize=(16, 8))
        plt.rc('font', size=16)
        plt.title("s-QAOA")
        plt.xlabel("Bitstrings")
        plt.ylabel("Frequency")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist([bitstrings1],
                 bins=bins,
                 density=True,
                 color=['#286d8c'],
                 edgecolor = "#041017",
                 label=['s-QAOA'])

        plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1.pdf',
        #            transparent=True)

    plt.show()


# %% Run the Thing
params, bitstrings = sqaoa(n_layers=1)
graph(bitstrings, beamer=True)
print_obj_vals(n_scenarios)


# %% graphs
graph(bitstrings, beamer=True)
