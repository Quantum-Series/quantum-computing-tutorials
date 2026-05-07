# QAOA vs VQE on Max-Cut (Qiskit 2 + IBM Quantum, Aer + hardware)


This notebook is a **teaching-focused project** for comparing **QAOA** and **VQE** on the same combinatorial optimization problem: **Max-Cut**.

## Design goals

- Uses **Qiskit 2.x-style primitives (V2)** and the **IBM Quantum Platform** primitives when running on real hardware.
- Runs end-to-end on:
  - **Aer simulator** (local)
  - **IBM Quantum hardware** (optional section; one-click switch)
- Keeps the instance size small enough to be **hardware-feasible** and **visually interpretable**.

## What students will do

1. Define a small graph and compute the **exact classical optimum** (brute force).
2. Build the **Max-Cut cost operator** and connect it to:
   - QAOA (sampling-based objective)
   - VQE (energy minimization of a related Hamiltonian)
3. Run both on **Aer**, then (optionally) execute **one evaluation run on hardware**.
4. Compare solution quality, distributions, and circuit resources.

> **Note on hardware runtime:** full variational optimization on real hardware can be slow due to queue time and noise. This notebook defaults to **optimize on Aer** and **evaluate on hardware** with the found parameters.


```python
# Install Qiskit v2.x stack
#%pip install qiskit qiskit-ibm-runtime qiskit-optimization qiskit-algorithms qiskit-aer networkx matplotlib pylatexenc

```

---
## 1. Configuration

Set execution mode and parameters.


```python
# Execution mode
RUN_ON = "aer"  # "aer" for local simulator, "hardware" for IBM Quantum

# IBM Quantum API Token
# If you have not saved your account to disk yet, paste your token below.
# You can find it at: https://quantum.ibm.com/account
API_TOKEN = None  # Replace with string: "your_token_here"

# Optimization settings
MAX_OPT_ITERS = 100  # Maximum optimizer iterations
SHOTS = 1024         # Number of shots for sampling
```

---
## 2. Imports


```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

# Qiskit core
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Aer simulator and local primitives
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2

# IBM Quantum runtime primitives
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2

# Optimizers
from qiskit_algorithms.optimizers import COBYLA, SPSA

print("✓ All imports successful")
```

    ✓ All imports successful


---
## 3. Define the Max-Cut Problem Instance

We use a **5-node weighted graph** that is small enough for:
- Classical brute-force solution
- Hardware execution
- Visual interpretation


```python
# Define weighted edges: (node_u, node_v, weight)
edges = [
    (0, 1, 1.0),
    (0, 2, 1.5),
    (1, 2, 2.0),
    (1, 3, 1.0),
    (2, 3, 1.5),
    (2, 4, 1.0),
    (3, 4, 2.0)
]

# Build graph
G = nx.Graph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

n = G.number_of_nodes()
print(f"Graph: {n} nodes, {G.number_of_edges()} edges")

# Visualize
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6, 5))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=600, font_size=14, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)
plt.title("Max-Cut Problem Instance", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

    Graph: 5 nodes, 7 edges


    /tmp/ipykernel_121545/2193459816.py:28: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      plt.tight_layout()



    
![png](output_7_2.png)
    


---
## 4. Classical Brute-Force Solution

Compute the **exact optimum** by exhaustive search over all $2^n$ bitstrings.


```python
def compute_cut(bitstring, graph):
    """Compute the cut value for a given bitstring.
    
    Args:
        bitstring: tuple of 0s and 1s (partition assignment)
        graph: NetworkX graph with 'weight' edge attribute
    
    Returns:
        Cut value (sum of weights crossing the partition)
    """
    cut = 0.0
    for u, v, data in graph.edges(data=True):
        if bitstring[u] != bitstring[v]:
            cut += data['weight']
    return cut

# Brute-force search
best_cut = 0
best_bitstring = None

for bitstring in product([0, 1], repeat=n):
    cut = compute_cut(bitstring, G)
    if cut > best_cut:
        best_cut = cut
        best_bitstring = bitstring

print(f"Classical optimum cut value: {best_cut}")
print(f"Classical optimum bitstring: {''.join(map(str, best_bitstring))}")
```

    Classical optimum cut value: 7.5
    Classical optimum bitstring: 00110


---
## 5. Construct the Max-Cut Cost Hamiltonian

The Max-Cut objective is:

$$
C = \sum_{(i,j) \in E} w_{ij} \frac{1 - Z_i Z_j}{2}
$$

We construct $H = -C$ so that **minimizing energy maximizes the cut**.

Expanding:
$$
C = \sum_{(i,j)} w_{ij} \left(\frac{1}{2} - \frac{1}{2} Z_i Z_j\right)
$$

Thus:
$$
H = -C = \sum_{(i,j)} \frac{w_{ij}}{2} Z_i Z_j - \text{const}
$$

We drop the constant term for optimization.


```python
# Build the cost Hamiltonian H = -C
pauli_list = []

for u, v, data in G.edges(data=True):
    w = data['weight']
    # Create ZZ term for edge (u, v)
    pauli_str = ['I'] * n
    pauli_str[u] = 'Z'
    pauli_str[v] = 'Z'
    pauli_list.append((''.join(pauli_str), w / 2.0))

H = SparsePauliOp.from_list(pauli_list)
H = H.simplify()

print("Cost Hamiltonian H (to minimize):")
print(H)
print(f"\nNumber of terms: {len(H)}")
```

    Cost Hamiltonian H (to minimize):
    SparsePauliOp(['ZZIII', 'ZIZII', 'IZZII', 'IZIZI', 'IIZZI', 'IIZIZ', 'IIIZZ'],
                  coeffs=[0.5 +0.j, 0.75+0.j, 1.  +0.j, 0.5 +0.j, 0.75+0.j, 0.5 +0.j, 1.  +0.j])
    
    Number of terms: 7


---
## 6. Setup Backend and Primitives

Configure the execution backend and primitives based on `RUN_ON` setting.


```python
if RUN_ON == "aer":
    # Local Aer simulator
    backend = AerSimulator()
    backend.set_options(shots=SHOTS)
    
    sampler = BackendSamplerV2(backend=backend)
    estimator = BackendEstimatorV2(backend=backend)
    
    print(f"✓ Using AerSimulator with {SHOTS} shots")
    
else:
    # IBM Quantum hardware
    try:
        # Try loading saved account first
        service = QiskitRuntimeService()
    except Exception:
        # If no saved account, use the provided token
        if API_TOKEN:
            service = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN)
        else:
            raise ValueError("Please provide a valid API_TOKEN in the configuration cell or save your account to disk.")

    print("Searching for the least busy operational quantum computer...")
    # Find the least busy real backend (not a simulator)
    backend = service.least_busy(operational=True, simulator=False)
    
    sampler = SamplerV2(backend=backend)
    estimator = EstimatorV2(backend=backend)
    
    print(f"✓ Using IBM Quantum backend: {backend.name}")
    print(f"  Status: {backend.status().status_msg}")
    print(f"  Pending jobs: {backend.status().pending_jobs}")

# Generate transpiler pass manager
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
print(f"✓ Transpiler pass manager ready (optimization_level=3)")
```

    ✓ Using AerSimulator with 1024 shots
    ✓ Transpiler pass manager ready (optimization_level=3)


---
## 7. QAOA Implementation

### 7.1 QAOA Circuit Construction

QAOA alternates between:
- **Cost layer**: $e^{-i\gamma H_C}$ (encodes the problem)
- **Mixer layer**: $e^{-i\beta H_M}$ (explores the solution space)

We use a **1-layer QAOA** (p=1) with parameters $(\gamma, \beta)$.


```python
def qaoa_circuit(gamma, beta, graph):
    """Build a 1-layer QAOA circuit for Max-Cut.
    
    Args:
        gamma: Cost layer parameter
        beta: Mixer layer parameter
        graph: NetworkX graph
    
    Returns:
        QuantumCircuit with measurements
    """
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)
    
    # Initial state: uniform superposition
    qc.h(range(n))
    
    # Cost layer: e^{-i gamma H_C}
    # For Max-Cut: H_C = sum_edges w * (1 - ZZ)/2
    # Implements as RZZ gates
    for u, v, data in graph.edges(data=True):
        w = data['weight']
        qc.rzz(gamma * w, u, v)
    
    # Mixer layer: e^{-i beta sum(X_i)}
    for i in range(n):
        qc.rx(2 * beta, i)
    
    # Measurements
    qc.measure_all()
    
    return qc

# Test circuit
test_qc = qaoa_circuit(0.5, 0.5, G)
print(f"QAOA circuit: {test_qc.num_qubits} qubits, {test_qc.depth()} depth")
print(test_qc)
```

    QAOA circuit: 5 qubits, 10 depth
            ┌───┐                     ┌───────┐                               »
       q_0: ┤ H ├─■─────────■─────────┤ Rx(1) ├───────────────────────────────»
            ├───┤ │ZZ(0.5)  │         └───────┘           ┌───────┐           »
       q_1: ┤ H ├─■─────────┼───────────■───────■─────────┤ Rx(1) ├───────────»
            ├───┤           │ZZ(0.75)   │ZZ(1)  │         └───────┘           »
       q_2: ┤ H ├───────────■───────────■───────┼─────────■──────────■────────»
            ├───┤                               │ZZ(0.5)  │ZZ(0.75)  │        »
       q_3: ┤ H ├───────────────────────────────■─────────■──────────┼────────»
            ├───┤                                                    │ZZ(0.5) »
       q_4: ┤ H ├────────────────────────────────────────────────────■────────»
            └───┘                                                             »
    meas: 5/══════════════════════════════════════════════════════════════════»
                                                                              »
    «                           ░ ┌─┐            
    «   q_0: ───────────────────░─┤M├────────────
    «                           ░ └╥┘┌─┐         
    «   q_1: ───────────────────░──╫─┤M├─────────
    «        ┌───────┐          ░  ║ └╥┘┌─┐      
    «   q_2: ┤ Rx(1) ├──────────░──╫──╫─┤M├──────
    «        └───────┘┌───────┐ ░  ║  ║ └╥┘┌─┐   
    «   q_3: ──■──────┤ Rx(1) ├─░──╫──╫──╫─┤M├───
    «          │ZZ(1) ├───────┤ ░  ║  ║  ║ └╥┘┌─┐
    «   q_4: ──■──────┤ Rx(1) ├─░──╫──╫──╫──╫─┤M├
    «                 └───────┘ ░  ║  ║  ║  ║ └╥┘
    «meas: 5/══════════════════════╩══╩══╩══╩══╩═
    «                              0  1  2  3  4 


### 7.2 QAOA Objective Function

QAOA uses a **sampling-based objective**: compute the expected cut value from measurement outcomes.


```python
def qaoa_objective(params, graph, sampler_primitive, pm_transpiler, shots_count):
    """Compute expected cut value from QAOA sampling.
    
    Args:
        params: [gamma, beta]
        graph: NetworkX graph
        sampler_primitive: Sampler primitive
        pm_transpiler: Transpiler pass manager
        shots_count: Number of shots
    
    Returns:
        Negative expected cut (for minimization)
    """
    gamma, beta = params
    
    # Build and transpile circuit
    qc = qaoa_circuit(gamma, beta, graph)
    isa_qc = pm_transpiler.run(qc)
    
    # Run sampler
    if RUN_ON == "aer":
        job = sampler_primitive.run([isa_qc], shots=shots_count)
    else:
        job = sampler_primitive.run([isa_qc])
    
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    # Compute expected cut
    exp_cut = 0.0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        bits = tuple(int(b) for b in bitstring)
        cut_val = compute_cut(bits, graph)
        exp_cut += cut_val * count / total
    
    # Return negative (optimizer minimizes)
    return -exp_cut
```

### 7.3 QAOA Optimization


```python
print("Starting QAOA optimization...")

# Initial parameters
theta0_qaoa = np.array([0.5, 0.5])

# Optimizer
opt_qaoa = COBYLA(maxiter=MAX_OPT_ITERS)

# Run optimization
res_qaoa = opt_qaoa.minimize(
    fun=lambda params: qaoa_objective(params, G, sampler, pm, SHOTS),
    x0=theta0_qaoa
)

gamma_opt, beta_opt = res_qaoa.x
qaoa_exp_cut = -res_qaoa.fun

print(f"\n✓ QAOA optimization complete")
print(f"  Optimal γ = {gamma_opt:.4f}")
print(f"  Optimal β = {beta_opt:.4f}")
print(f"  Expected cut = {qaoa_exp_cut:.4f}")
print(f"  Function evaluations: {res_qaoa.nfev}")
```

    Starting QAOA optimization...
    
    ✓ QAOA optimization complete
      Optimal γ = 1.6290
      Optimal β = 0.4040
      Expected cut = 5.8589
      Function evaluations: 31


### 7.4 QAOA Results Analysis


```python
# Sample from optimal QAOA circuit
qc_qaoa_opt = qaoa_circuit(gamma_opt, beta_opt, G)
isa_qaoa = pm.run(qc_qaoa_opt)

if RUN_ON == "aer":
    job_qaoa = sampler.run([isa_qaoa], shots=SHOTS)
else:
    job_qaoa = sampler.run([isa_qaoa])

result_qaoa = job_qaoa.result()
counts_qaoa = result_qaoa[0].data.meas.get_counts()

# Find best sampled bitstring
best_sampled_cut_qaoa = 0
best_sampled_bitstring_qaoa = None

for bitstring, count in counts_qaoa.items():
    bits = tuple(int(b) for b in bitstring)
    cut_val = compute_cut(bits, G)
    if cut_val > best_sampled_cut_qaoa:
        best_sampled_cut_qaoa = cut_val
        best_sampled_bitstring_qaoa = bitstring

print(f"Best sampled bitstring: {best_sampled_bitstring_qaoa}")
print(f"Best sampled cut: {best_sampled_cut_qaoa}")
print(f"Approximation ratio: {best_sampled_cut_qaoa / best_cut:.4f}")

# Plot distribution
sorted_counts = sorted(counts_qaoa.items(), key=lambda x: -x[1])
top_10 = sorted_counts[:min(10, len(sorted_counts))]
labels = [x[0] for x in top_10]
values = [x[1] for x in top_10]

plt.figure(figsize=(10, 5))
plt.bar(labels, values, color='steelblue', alpha=0.8)
plt.xlabel('Bitstring', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.title('QAOA Sample Distribution (Top 10)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

    Best sampled bitstring: 00110
    Best sampled cut: 7.5
    Approximation ratio: 1.0000



    
![png](output_21_1.png)
    


---
## 8. VQE Implementation

### 8.1 VQE Ansatz Construction

VQE uses a **hardware-efficient ansatz** with parameterized rotation gates and entangling layers.


```python
def vqe_ansatz(params, n_qubits):
    """Build a hardware-efficient ansatz for VQE.
    
    Args:
        params: Parameter values (or Parameter objects)
        n_qubits: Number of qubits
    
    Returns:
        QuantumCircuit (without measurements)
    """
    qc = QuantumCircuit(n_qubits)
    
    # Initial layer: uniform superposition
    qc.h(range(n_qubits))
    
    # Parameterized rotation layer
    for i in range(n_qubits):
        qc.ry(params[i], i)
    
    # Entangling layer
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    return qc

num_params_vqe = n
print(f"VQE ansatz: {num_params_vqe} parameters")

# Test ansatz
test_params = [Parameter(f'θ{i}') for i in range(num_params_vqe)]
test_ansatz = vqe_ansatz(test_params, n)
print(test_ansatz)
```

    VQE ansatz: 5 parameters
         ┌───┐┌────────┐                    
    q_0: ┤ H ├┤ Ry(θ0) ├──■─────────────────
         ├───┤├────────┤┌─┴─┐               
    q_1: ┤ H ├┤ Ry(θ1) ├┤ X ├──■────────────
         ├───┤├────────┤└───┘┌─┴─┐          
    q_2: ┤ H ├┤ Ry(θ2) ├─────┤ X ├──■───────
         ├───┤├────────┤     └───┘┌─┴─┐     
    q_3: ┤ H ├┤ Ry(θ3) ├──────────┤ X ├──■──
         ├───┤├────────┤          └───┘┌─┴─┐
    q_4: ┤ H ├┤ Ry(θ4) ├───────────────┤ X ├
         └───┘└────────┘               └───┘


### 8.2 Transpile VQE Ansatz and Observable


```python
# Build parameterized ansatz
param_list = [Parameter(f'θ{i}') for i in range(num_params_vqe)]
ansatz_param = vqe_ansatz(param_list, n)

# Transpile ansatz to ISA
isa_ansatz = pm.run(ansatz_param)

# Apply layout to observable
isa_H = H.apply_layout(isa_ansatz.layout)

print(f"✓ VQE ansatz transpiled")
print(f"  Depth: {isa_ansatz.depth()}")
print(f"  Size: {isa_ansatz.size()}")
print(f"\n✓ Observable layout applied")
```

    ✓ VQE ansatz transpiled
      Depth: 6
      Size: 14
    
    ✓ Observable layout applied


### 8.3 VQE Objective Function

VQE uses an **Estimator-based objective**: compute $\langle H \rangle$ directly.


```python
def vqe_objective(params):
    """Compute expectation value <H> using Estimator.
    
    Args:
        params: Parameter values
    
    Returns:
        Energy (expectation value)
    """
    # EstimatorV2 PUB: (circuit, observable, parameter_values)
    job = estimator.run([(isa_ansatz, isa_H, list(params))])
    result = job.result()
    
    # result[0].data.evs is a scalar
    energy = float(result[0].data.evs)
    
    return energy
```

### 8.4 VQE Optimization


```python
print("Starting VQE optimization...")

# Initial parameters (random)
np.random.seed(42)
theta0_vqe = np.random.rand(num_params_vqe) * 2 * np.pi

# Optimizer (use SPSA for hardware, COBYLA for Aer)
if RUN_ON == "aer":
    opt_vqe = COBYLA(maxiter=MAX_OPT_ITERS)
else:
    opt_vqe = SPSA(maxiter=MAX_OPT_ITERS)

# Run optimization
res_vqe = opt_vqe.minimize(vqe_objective, x0=theta0_vqe)

theta_opt_vqe = res_vqe.x
vqe_energy = res_vqe.fun

print(f"\n✓ VQE optimization complete")
print(f"  Optimal energy: {vqe_energy:.4f}")
print(f"  Optimal parameters: {theta_opt_vqe}")
print(f"  Function evaluations: {res_vqe.nfev}")
```

    Starting VQE optimization...
    
    ✓ VQE optimization complete
      Optimal energy: -2.4585
      Optimal parameters: [4.27326172 7.84810928 4.77073772 1.52737972 4.09936593]
      Function evaluations: 100


### 8.5 VQE Results Analysis

Sample from the optimal VQE state to extract bitstrings.


```python
# Build optimal VQE circuit with measurements
qc_vqe_opt = vqe_ansatz(theta_opt_vqe, n)
qc_vqe_opt.measure_all()
isa_vqe = pm.run(qc_vqe_opt)

# Sample
if RUN_ON == "aer":
    job_vqe = sampler.run([isa_vqe], shots=SHOTS)
else:
    job_vqe = sampler.run([isa_vqe])

result_vqe = job_vqe.result()
counts_vqe = result_vqe[0].data.meas.get_counts()

# Find best sampled bitstring
best_sampled_cut_vqe = 0
best_sampled_bitstring_vqe = None

for bitstring, count in counts_vqe.items():
    bits = tuple(int(b) for b in bitstring)
    cut_val = compute_cut(bits, G)
    if cut_val > best_sampled_cut_vqe:
        best_sampled_cut_vqe = cut_val
        best_sampled_bitstring_vqe = bitstring

# Compute expected cut from samples
vqe_exp_cut = 0.0
total_vqe = sum(counts_vqe.values())

for bitstring, count in counts_vqe.items():
    bits = tuple(int(b) for b in bitstring)
    cut_val = compute_cut(bits, G)
    vqe_exp_cut += cut_val * count / total_vqe

print(f"Best sampled bitstring: {best_sampled_bitstring_vqe}")
print(f"Best sampled cut: {best_sampled_cut_vqe}")
print(f"Expected cut (from samples): {vqe_exp_cut:.4f}")
print(f"Approximation ratio: {best_sampled_cut_vqe / best_cut:.4f}")

# Plot distribution
sorted_counts_vqe = sorted(counts_vqe.items(), key=lambda x: -x[1])
top_10_vqe = sorted_counts_vqe[:min(10, len(sorted_counts_vqe))]
labels_vqe = [x[0] for x in top_10_vqe]
values_vqe = [x[1] for x in top_10_vqe]

plt.figure(figsize=(10, 5))
plt.bar(labels_vqe, values_vqe, color='coral', alpha=0.8)
plt.xlabel('Bitstring', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.title('VQE Sample Distribution (Top 10)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

    Best sampled bitstring: 11001
    Best sampled cut: 7.5
    Expected cut (from samples): 7.4507
    Approximation ratio: 1.0000



    
![png](output_31_1.png)
    


---
## 9. Comparison and Analysis

### 9.1 Solution Quality Comparison


```python
print("=" * 70)
print("QAOA vs VQE: SOLUTION QUALITY COMPARISON")
print("=" * 70)
print(f"\nClassical optimum cut: {best_cut}")
print(f"Classical optimum bitstring: {''.join(map(str, best_bitstring))}")
print()
print("-" * 70)
print("QAOA Results:")
print("-" * 70)
print(f"  Expected cut (optimization): {qaoa_exp_cut:.4f}")
print(f"  Best sampled cut: {best_sampled_cut_qaoa}")
print(f"  Best sampled bitstring: {best_sampled_bitstring_qaoa}")
print(f"  Approximation ratio: {best_sampled_cut_qaoa / best_cut:.4f}")
print(f"  Optimal parameters: γ={gamma_opt:.4f}, β={beta_opt:.4f}")
print()
print("-" * 70)
print("VQE Results:")
print("-" * 70)
print(f"  Optimal energy: {vqe_energy:.4f}")
print(f"  Expected cut (from samples): {vqe_exp_cut:.4f}")
print(f"  Best sampled cut: {best_sampled_cut_vqe}")
print(f"  Best sampled bitstring: {best_sampled_bitstring_vqe}")
print(f"  Approximation ratio: {best_sampled_cut_vqe / best_cut:.4f}")
print()
print("=" * 70)
```

    ======================================================================
    QAOA vs VQE: SOLUTION QUALITY COMPARISON
    ======================================================================
    
    Classical optimum cut: 7.5
    Classical optimum bitstring: 00110
    
    ----------------------------------------------------------------------
    QAOA Results:
    ----------------------------------------------------------------------
      Expected cut (optimization): 5.8589
      Best sampled cut: 7.5
      Best sampled bitstring: 00110
      Approximation ratio: 1.0000
      Optimal parameters: γ=1.6290, β=0.4040
    
    ----------------------------------------------------------------------
    VQE Results:
    ----------------------------------------------------------------------
      Optimal energy: -2.4585
      Expected cut (from samples): 7.4507
      Best sampled cut: 7.5
      Best sampled bitstring: 11001
      Approximation ratio: 1.0000
    
    ======================================================================


### 9.2 Circuit Resource Comparison


```python
print("\n" + "=" * 70)
print("CIRCUIT RESOURCE COMPARISON")
print("=" * 70)
print(f"\nQAOA Circuit (transpiled):")
print(f"  Depth: {isa_qaoa.depth()}")
print(f"  Size (gate count): {isa_qaoa.size()}")
print(f"  Number of parameters: 2")
print()
print(f"VQE Circuit (transpiled):")
print(f"  Depth: {isa_vqe.depth()}")
print(f"  Size (gate count): {isa_vqe.size()}")
print(f"  Number of parameters: {num_params_vqe}")
print("=" * 70)
```

    
    ======================================================================
    CIRCUIT RESOURCE COMPARISON
    ======================================================================
    
    QAOA Circuit (transpiled):
      Depth: 10
      Size (gate count): 22
      Number of parameters: 2
    
    VQE Circuit (transpiled):
      Depth: 6
      Size (gate count): 14
      Number of parameters: 5
    ======================================================================


### 9.3 Side-by-Side Distribution Comparison


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# QAOA distribution
sorted_counts_qaoa = sorted(counts_qaoa.items(), key=lambda x: -x[1])
top_10_qaoa = sorted_counts_qaoa[:min(10, len(sorted_counts_qaoa))]
labels_qaoa = [x[0] for x in top_10_qaoa]
values_qaoa = [x[1] for x in top_10_qaoa]

axes[0].bar(labels_qaoa, values_qaoa, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Bitstring', fontsize=11)
axes[0].set_ylabel('Counts', fontsize=11)
axes[0].set_title('QAOA Distribution', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# VQE distribution
axes[1].bar(labels_vqe, values_vqe, color='coral', alpha=0.8)
axes[1].set_xlabel('Bitstring', fontsize=11)
axes[1].set_ylabel('Counts', fontsize=11)
axes[1].set_title('VQE Distribution', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](output_37_0.png)
    


---
## 10. Key Takeaways

### Algorithm Comparison

**QAOA:**
- Problem-specific circuit structure (encodes Max-Cut directly)
- Fewer parameters (2 for p=1)
- Sampling-based optimization
- Often finds good solutions with shallow circuits

**VQE:**
- General-purpose ansatz (hardware-efficient)
- More parameters (scales with qubits)
- Energy-based optimization (uses Estimator)
- More flexible but requires careful ansatz design

### Practical Considerations

1. **Hardware execution**: Both algorithms are NISQ-friendly but sensitive to noise
2. **Optimization landscape**: QAOA often has smoother landscapes for combinatorial problems
3. **Scalability**: Circuit depth and parameter count grow differently
4. **Hybrid approach**: Can combine QAOA structure with VQE optimization techniques

### Next Steps

- Try different graph instances
- Increase QAOA layers (p > 1)
- Experiment with different VQE ansätze
- Run on real hardware and analyze noise effects
- Implement error mitigation techniques

---
## 11. (Optional) Hardware Evaluation

If you optimized on Aer, you can evaluate the found parameters on real hardware here.


```python
# Uncomment and run this cell to evaluate on hardware

if RUN_ON == "aer":
    print("Setting up hardware evaluation...")
    
    try:
        service = QiskitRuntimeService()
    except Exception:
        if API_TOKEN:
            service = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN)
        else:
            raise ValueError("Please provide a valid API_TOKEN.")

    print("Searching for the least busy operational quantum computer...")
    hw_backend = service.least_busy(operational=True, simulator=False)
    print(f"✓ Found backend: {hw_backend.name}")

    # CORRECTED LINE: Use 'mode' instead of 'backend' for QiskitRuntimeService SamplerV2
    hw_sampler = SamplerV2(mode=hw_backend)
    
    hw_pm = generate_preset_pass_manager(optimization_level=3, backend=hw_backend)
    
    # Evaluate QAOA on hardware
    qc_qaoa_hw = qaoa_circuit(gamma_opt, beta_opt, G)
    isa_qaoa_hw = hw_pm.run(qc_qaoa_hw)
    job_qaoa_hw = hw_sampler.run([isa_qaoa_hw])
    
    print(f"QAOA hardware job submitted: {job_qaoa_hw.job_id()}")
    print("Waiting for results...")
    
    result_qaoa_hw = job_qaoa_hw.result()
    counts_qaoa_hw = result_qaoa_hw[0].data.meas.get_counts()
    
    # Analyze hardware results
    best_hw_cut = 0
    best_hw_bitstring = None
    for bitstring, count in counts_qaoa_hw.items():
        bits = tuple(int(b) for b in bitstring)
        cut_val = compute_cut(bits, G)
        if cut_val > best_hw_cut:
            best_hw_cut = cut_val
            best_hw_bitstring = bitstring
    
    print(f"\\nHardware results:")
    print(f"  Best cut: {best_hw_cut}")
    print(f"  Best bitstring: {best_hw_bitstring}")
    print(f"  Approximation ratio: {best_hw_cut / best_cut:.4f}")
```

    Setting up hardware evaluation...


    qiskit_runtime_service.__init__:WARNING:2026-01-06 01:59:34,224: Instance was not set at service instantiation. Free and trial plan instances will be prioritized. Based on the following filters: (tags: None, region: us-east, eu-de), and available plans: (open), the available account instances are: open-instance. If you need a specific instance set it explicitly either by using a saved account with a saved default instance or passing it in directly to QiskitRuntimeService().


    Searching for the least busy operational quantum computer...


    qiskit_runtime_service.backends:WARNING:2026-01-06 01:59:34,555: Loading instance: open-instance, plan: open
    qiskit_runtime_service.backends:WARNING:2026-01-06 01:59:35,942: Using instance: open-instance, plan: open


    ✓ Found backend: ibm_fez


    /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/qiskit_ibm_runtime/qiskit_runtime_service.py:1183: UserWarning: This instance has met its usage limit. Workloads will not run until time is made available. Check https://quantum.cloud.ibm.com/instances/crn%3Av1%3Abluemix%3Apublic%3Aquantum-computing%3Aus-east%3Aa%2F1d503490ff104d7fadfc5e7849670e58%3Acf2b9206-1d8c-4d88-a07c-576231e364e9%3A%3A for more details.
      warnings.warn(


    QAOA hardware job submitted: d5e6p2767pic7381jfvg
    Waiting for results...


---
**End of Notebook**

This notebook demonstrated a complete comparison of QAOA and VQE for Max-Cut using Qiskit 2.x primitives. Both algorithms successfully found high-quality solutions to the combinatorial optimization problem.
