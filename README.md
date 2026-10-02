# HeteroGuard

**Runtime assurance for heterogeneous scientific computing.**

HeteroGuard is a lightweight design pattern and emerging Python toolkit for scientific workflows that move computation across heterogeneous execution backends — especially CPU and GPU — while verifying results at the boundaries instead of treating an accelerated backend as automatically trustworthy.

The core idea is simple:

> **Offload when acceleration adds value. Verify at the handoff. Fall back safely when it does not. Refuse scientifically invalid states.**

HeteroGuard is not a new CPU/GPU scheduling paradigm. Heterogeneous computing, hybrid CPU/GPU execution, task runtimes, and differential testing all have substantial prior art. HeteroGuard focuses on a narrower operational problem that appears frequently in modern scientific software:

- only part of a workflow is GPU-accelerated;
- work moves repeatedly between CPU and GPU implementations;
- accelerated libraries may have incomplete feature coverage;
- new hardware may depend on JIT-compiled kernels or rapidly evolving software stacks;
- numerical agreement can be subtly affected by precision, implementation, convergence, or state-selection differences;
- and a scientifically plausible result can still be wrong even when the hardware calculation itself is numerically correct.

HeteroGuard makes those boundaries explicit and attaches inexpensive runtime assurance to them.

---

## Why HeteroGuard?

A common scientific workflow looks like this:

```text
CPU preparation
      |
      v
GPU-accelerated solve
      |
      v
CPU-only analysis
      |
      v
GPU-accelerated kernel
      |
      v
CPU post-processing
```

In practice, the transition points are often treated as plumbing.

HeteroGuard treats them as **assurance boundaries**.

```text
        +------------------+
        |   CPU backend    |
        +---------+--------+
                  |
             Backend Probe
                  |
                  v
        +------------------+
        |   GPU backend    |
        +---------+--------+
                  |
            Compute Witness
                  |
                  v
        +------------------+
        | Invariant Guards |
        +---------+--------+
                  |
          accept / fallback
```

The goal is not to duplicate every calculation. The goal is to perform the **cheapest independent check that meaningfully increases confidence**.

---

# Core mechanisms

HeteroGuard currently organizes the pattern into four mechanisms.

## 1. Backend Probe

Before trusting an accelerated implementation for production work, compare it against a trusted backend on representative inputs.

For example:

```python
rng = np.random.default_rng(1)
a = rng.standard_normal((2, n, n)) * 1e-2

for name, dm, hermi in (
    ("symmetric", a + a.transpose(0, 2, 1), 1),
    ("non-symmetric", a, 0),
):
    j_cpu, k_cpu = cpu_get_jk(dm, hermi)
    j_gpu, k_gpu = gpu_get_jk(dm, hermi)

    error = max(
        np.abs(j_cpu - j_gpu).max(),
        np.abs(k_cpu - k_gpu).max(),
    )

    if error > tolerance:
        reject_gpu_path()
```

A backend does not become trusted merely because it imported successfully or returned a value.

The probe establishes that the accelerated implementation agrees with the reference implementation for the relevant call type.

A useful consequence is **selective trust**: if one category of calls verifies and another does not, HeteroGuard can dispatch only the verified subset to the accelerator.

---

## 2. Compute Witness

A GPU calculation may be expensive to reproduce in full on the CPU. Instead, HeteroGuard re-evaluates a cheaper quantity derived from the GPU result using an independent backend.

Example:

```python
gpu_result = run_gpu_scf()

cpu_object.mo_coeff = gpu_result.mo_coeff
cpu_object.mo_occ = gpu_result.mo_occ
cpu_object.mo_energy = gpu_result.mo_energy

e_cpu = cpu_object.energy_tot(cpu_object.make_rdm1())
delta = abs(e_cpu - gpu_result.e_tot)

if delta > tolerance:
    reject_gpu_result()
```

This is a **witness**, not a full redundant execution.

The accelerated backend performs the expensive computation. The trusted backend independently evaluates a property that should agree if the result is internally consistent.

This makes runtime verification practical even when repeating the complete solve would eliminate the performance advantage of acceleration.

---

## 3. Guarded Fallback

Acceleration should be opportunistic, not mandatory.

If the GPU implementation:

- is unavailable,
- does not support a required operation,
- fails a backend probe,
- disagrees with its witness,
- does not converge,
- or reaches a scientifically suspicious state,

the workflow can continue through a trusted CPU path.

Conceptually:

```python
result = try_gpu()

if result is None:
    result = run_cpu()

if not result.converged:
    result = run_cpu_newton()
```

The important detail is that fallback should not blindly continue from a corrupted or diverged intermediate state.

A guard may decide that the fallback solver should restart from a known seed instead:

```python
diverged = reference_energy is not None and e_diis > reference_energy + 0.5

start = original_seed if diverged else last_density
run_cpu_newton(start)
```

HeteroGuard therefore treats fallback as part of the correctness model, not just error handling.

---

## 4. Scientific Invariant Guards

Cross-backend numerical agreement is necessary but not sufficient.

Two implementations can agree perfectly while both operate on the wrong scientific state.

HeteroGuard therefore includes **domain invariants** that refuse to continue when a result violates a scientific relationship that should hold.

Examples from the originating quantum-chemistry workflow include:

### Balanced model-space guard

Both spin states must use the same active-space dimensions:

```python
if spaces[0][:2] != spaces[1][:2]:
    raise RuntimeError(
        "active spaces differ between spins; "
        "the comparison would not be balanced"
    )
```

### Energy-continuity guard

A large discontinuity across nearby geometries can indicate that the solver has converged to a different electronic state:

```python
delta_kcal = (current_energy - source_energy) * HARTREE_TO_KCAL

if delta_kcal > 15:
    warn("large upward jump: possible different excited solution")
```

### Reference-energy guard

A reduced-space variational result should not lie above its own reference in a way that signals a malformed starting state:

```python
if casci_energy > scf_energy + tolerance:
    refuse("starting orbitals are inconsistent with the reduced-space solve")
```

### Checkpoint reproducibility guard

A persisted state must reproduce the energy stored with it:

```python
delta = abs(recomputed_energy - checkpoint_energy)

if delta > 1e-6:
    refuse("checkpoint does not reproduce its stored energy")
```

### Same-state guard

Independent starts should converge to the same minimum within a scientifically meaningful tolerance, or the workflow should report that different local solutions were reached.

---

# The HeteroGuard execution model

HeteroGuard can be summarized as:

```text
                +----------------------+
                |  Candidate backend   |
                +----------+-----------+
                           |
                    1. PROBE
                           |
             +-------------+-------------+
             |                           |
           pass                         fail
             |                           |
             v                           v
       accelerated                  trusted
         dispatch                   fallback
             |
             v
       accelerated
          result
             |
        2. WITNESS
             |
      +------+------+
      |             |
    agree        disagree
      |             |
      v             v
  3. GUARDS      fallback
      |
 +----+----+
 |         |
pass     refuse
 |
 v
accept
```

The accelerator is therefore **conditionally trusted**, not globally trusted.

---

# What HeteroGuard is — and is not

## HeteroGuard is

- runtime verification at heterogeneous compute boundaries;
- selective dispatch based on verified backend capability;
- cheap independent witnessing of accelerated results;
- guarded CPU fallback;
- scientific invariant enforcement;
- provenance-friendly recording of verification decisions;
- useful for partially accelerated scientific Python workflows.

## HeteroGuard is not

- a replacement for CUDA, ROCm, SYCL, OpenMP, or heterogeneous runtimes;
- a new CPU/GPU scheduler;
- a claim that heterogeneous computing itself is novel;
- formal verification;
- a substitute for unit tests or continuous integration;
- proof that two agreeing numerical implementations are scientifically correct.

HeteroGuard complements those layers.

---

# Origin

The pattern emerged from a mixed CPU/GPU quantum-chemistry workflow using:

- PySCF 2.14
- GPU4PySCF 1.8.1
- NumPy / CuPy
- NVIDIA RTX 5090 (Blackwell)

The workflow repeatedly moved between GPU-supported and CPU-only stages.

Instead of treating each device transition as an implementation detail, the pipeline began verifying the transition itself.

Examples included:

- validating GPU density-fitted Coulomb/exchange builds against CPU builds on random symmetric and non-symmetric densities;
- selectively keeping unsupported or unverified call types on the CPU;
- running SCF on the GPU and independently re-evaluating its energy on the CPU;
- falling back to CPU solvers on import failure or non-convergence;
- refusing inconsistent active spaces, discontinuous electronic states, malformed checkpoints, and physically inconsistent reduced-space solutions.

---

# Measured record from the originating workflow

In the initial application, every checked GPU result agreed with the CPU at approximately **1e-10 or better**.

The device checks therefore did **not** expose a GPU numerical error in those runs.

Their demonstrated value was:

- assurance on a new GPU architecture;
- verification across CPU/GPU implementations;
- confidence while GPU kernels were JIT-compiled for newer hardware;
- explicit handling of unavailable GPU libraries;
- safe fallback when GPU SCF did not converge.

Importantly, the checks that **did catch real workflow errors** were the scientific invariant guards.

Examples included:

- detecting different active spaces between spin states;
- identifying an excited-state solution through an anomalous energy jump;
- rejecting a CASCI result inconsistent with its SCF reference;
- refusing checkpoints that did not reproduce their stored energy.

That distinction matters.

HeteroGuard should not be marketed as "we found GPUs making mistakes."

The stronger and more accurate claim is:

> **HeteroGuard makes heterogeneous scientific execution auditable at runtime and adds domain-aware refusal mechanisms for silent scientific errors.**

---

# Why this matters now

Scientific software is becoming increasingly heterogeneous.

A single application may combine:

- Python orchestration;
- native CPU libraries;
- CUDA or ROCm kernels;
- JIT-compiled device code;
- vendor-specific accelerators;
- cloud execution;
- approximate solvers;
- machine-learning models;
- and eventually quantum or other specialized backends.

Acceleration coverage is rarely complete.

That means real applications increasingly behave like pipelines of **handoffs** between different execution environments.

Each handoff introduces questions:

- Did both implementations interpret the input the same way?
- Is the accelerated operation valid for this input class?
- Did precision change materially?
- Did a solver converge to the same state?
- Is the output internally consistent?
- Can the result be cheaply witnessed elsewhere?
- Is fallback safe?
- Does the answer satisfy domain invariants?

HeteroGuard makes those questions first-class.

---

# Potential applications

Although the initial implementation comes from quantum chemistry, the mechanism is intentionally general.

Potential applications include:

## Computational chemistry

- SCF / DFT
- CASSCF / CASCI
- correlated methods
- molecular dynamics
- GPU-integral evaluation
- accelerated tensor contractions
- electronic-structure pipelines

## Drug discovery

- molecular property calculations
- docking/scoring pipelines
- molecular dynamics
- hybrid physics/ML workflows
- accelerated screening
- model-to-physics cross-checks

## Materials science

- electronic structure
- atomistic simulation
- structure optimization
- accelerated surrogate models

## Numerical simulation

- sparse and dense linear algebra
- PDE solvers
- iterative methods
- GPU kernels with CPU references

## AI-assisted science

A future HeteroGuard backend need not be another numerical device.

The same architecture can support:

```text
ML surrogate -> physics witness
approximate solver -> high-fidelity witness
local execution -> cloud witness
classical backend -> quantum backend
implementation A -> implementation B
```

The key abstraction is independent, inexpensive evidence at an execution boundary.

---

# Planned Python API

The project is currently being extracted from application code into reusable components.

A possible interface is:

```python
from heteroguard import Probe, Witness, Guard, FallbackPolicy
```

### Backend probe

```python
probe = Probe(
    candidate=gpu_get_jk,
    reference=cpu_get_jk,
    comparator=max_abs_error,
    tolerance=1e-8,
)

status = probe.run(test_cases)
```

### Compute witness

```python
witness = Witness(
    evaluator=cpu_energy,
    comparator=absolute_difference,
    tolerance=1e-6,
)

verified = witness.check(
    candidate_value=gpu_energy,
    state=gpu_orbitals,
)
```

### Invariant guard

```python
guard = Guard(
    name="balanced-active-space",
    predicate=lambda a, b: a[:2] == b[:2],
)

guard.require(space_singlet, space_triplet)
```

### Guarded execution

```python
result = heteroguard.run(
    preferred=gpu_solver,
    fallback=cpu_solver,
    probe=backend_probe,
    witness=energy_witness,
    guards=[
        active_space_guard,
        continuity_guard,
        checkpoint_guard,
    ],
)
```

This API is intentionally provisional.

---

# Provenance

A scientific assurance layer should leave evidence behind.

A future HeteroGuard run record may contain:

```json
{
  "backend": "gpu",
  "reference_backend": "cpu",
  "operation": "scf",
  "probe": {
    "status": "pass",
    "max_error": 8.1e-11,
    "tolerance": 1e-8
  },
  "witness": {
    "status": "pass",
    "gpu_energy": -1234.56789012,
    "cpu_energy": -1234.56789012,
    "difference": 2.3e-10,
    "tolerance": 1e-6
  },
  "guards": {
    "checkpoint": "pass",
    "continuity": "pass",
    "model_balance": "pass"
  },
  "decision": "accepted"
}
```

The intent is to turn "the GPU path ran" into a more useful scientific statement:

> The accelerated path ran, was cross-checked, satisfied the configured scientific invariants, and the evidence was recorded.

---

# Design principles

HeteroGuard follows a few simple principles.

### Accelerate opportunistically

Use the GPU where it provides real value.

Do not force a workflow onto an accelerator when CPU execution is simpler, better supported, or more reliable.

### Verify cheaply

Prefer a low-cost independent witness over full redundant computation.

### Trust narrowly

Verification applies to a backend, operation, input class, software stack, and tolerance — not to "the GPU" in the abstract.

### Fail closed on scientific invariants

When a result violates a domain rule that makes the scientific comparison invalid, refuse to produce a polished answer.

### Preserve fallback correctness

Fallback is not safe if it starts from a corrupted or diverged state.

### Record evidence

Verification should be observable and reproducible.

---

# Relationship to existing work

HeteroGuard builds on established ideas rather than claiming to replace them.

Relevant areas include:

- heterogeneous CPU/GPU computing;
- differential testing;
- N-version programming;
- redundant execution;
- algorithm-based fault tolerance;
- runtime validation;
- resilience in high-performance computing;
- task-based heterogeneous runtimes.

Systems such as MAGMA, StarPU, PaRSEC, and Legion demonstrate sophisticated heterogeneous scheduling and execution.

HeteroGuard's focus is different:

> **What evidence should accompany a scientific result when work crosses backend boundaries?**

Its contribution is intended to be a small, practical runtime-assurance layer that can sit above existing scientific libraries and heterogeneous runtimes.

---

# Initial roadmap

## Phase 0 — extraction

- [ ] Extract backend-probe abstraction
- [ ] Extract compute-witness abstraction
- [ ] Extract invariant-guard abstraction
- [ ] Add explicit fallback policy
- [ ] Define structured verification records

## Phase 1 — generic examples

- [ ] NumPy / CuPy example
- [ ] CPU/GPU matrix-operation verification
- [ ] selective dispatch example
- [ ] injected numerical-error example
- [ ] fallback demonstration

## Phase 2 — scientific example

- [ ] PySCF / GPU4PySCF example
- [ ] GPU SCF + CPU energy witness
- [ ] density-fitted J/K backend probe
- [ ] scientific invariant examples

## Phase 3 — provenance

- [ ] JSON run records
- [ ] backend metadata
- [ ] tolerance metadata
- [ ] environment capture
- [ ] deterministic test inputs

## Phase 4 — broader backends

- [ ] multiple GPU implementations
- [ ] approximate / high-fidelity solver pairs
- [ ] ML surrogate / physical-model witnesses
- [ ] distributed or cloud execution
- [ ] experimental classical / quantum handoffs

---

# Status

**Early-stage / experimental.**

The design has been exercised inside a working scientific-computing pipeline, but the reusable HeteroGuard package is being extracted from that application.

The first goal is deliberately small:

> make runtime cross-backend verification and domain-aware refusal easy enough that scientific developers will actually leave it enabled.

---

# Repository structure

The intended layout is:

```text
heteroguard/
├── README.md
├── pyproject.toml
├── src/
│   └── heteroguard/
│       ├── __init__.py
│       ├── probe.py
│       ├── witness.py
│       ├── guard.py
│       ├── fallback.py
│       └── provenance.py
├── examples/
│   ├── numpy_cupy/
│   └── pyscf_gpu4pyscf/
├── tests/
└── docs/
```

---

# Naming

**HeteroGuard** = **heterogeneous-compute guard**.

The name reflects the project's scope:

- *heterogeneous* execution across different computational backends;
- *guarded* transitions, acceptance, and fallback;
- scientific rather than merely infrastructure-level correctness.

---

# Citation / research use

HeteroGuard does not yet have a formal release or archival citation.

If the project develops into a stable scientific software package, releases, benchmark artifacts, and archival records should be versioned separately from experimental application results.

---

# Contributing

The project is at an early design stage.

Useful contributions will eventually include:

- additional backend adapters;
- numerical comparators;
- domain-specific guards;
- failure-injection tests;
- reproducibility tooling;
- examples from scientific applications where partial GPU acceleration creates backend handoff boundaries.

---

# License

License to be selected before the first packaged release.

---

## One-line summary

**HeteroGuard verifies heterogeneous scientific computation at runtime: probe before trust, witness accelerated results, enforce scientific invariants, and fall back safely.**
