"""
imports_IBM_NM.py
=================
Shared utilities for the IBM quantum noise-modelling notebooks.

Sections
--------
Imports & colour palettes
Device / backend helpers
Measurement & data utilities
Circuit-conversion helpers  (cirq ↔ qiskit)
Filter-function & FTTPS circuits
RB circuits
Bloch-vector math & gate-error utilities
Markovian noise simulation  (Gmat, sim_exact, gen_T2DD_circs, gen_fpw_circuits)
Stochastic-noise simulation  (noisy_simulation class)
Noise-characterization class  (noise_characterization)
Backend / analysis helpers
"""
# General Imports
import numpy as np
import pandas as pd
import qutip as qt
import cirq
import itertools
from tqdm import tqdm
import copy
import scipy.optimize as spopt
import pickle as pk
import matplotlib.pyplot as plt
import scipy as sc
import scipy.signal as si
from sklearn.metrics import mean_squared_error as mse
import mezze.tfq as mez
import os, sys
import time
# sys.stdout, sys.stderr = os.devnull, os.devnull
from mezze.tfq import *
# sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

from cirq.contrib.qasm_import import circuit_from_qasm
# Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb
# Import Qiskit classes 
import qiskit_ibm_provider
import qiskit as qk
from qiskit import assemble, transpile, IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.transpiler.passes import RemoveBarriers
from datetime import datetime
from qiskit import transpile
import matplotlib as mpl

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*10
colors_blais = ['#33658A','#86BBD8']
plot_colors = ['#9A0EEA', '#BF77F6', '#030AA7', '#0165FC', '#39AD48', '#F97306', '#F7022A']
colors_greg = plot_colors   # backward-compatible alias

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def get_backend_name(device):
    if device in ['guadalupe','kolkata','mumbai']:
        return 'ibmq_'+device
    else:
    # elif device in ['auckland','algiers','cairo','hanoi','lagos','nazca','cusco','sherbrooke','perth']:
        return 'ibm_'+device


def get_instance(device):
    if device in ['cairo','algiers','auckland','cairo','hanoi','lagos','nazca','cusco','sherbrooke','perth','guadalupe','kolkata','mumbai','torino']:
        return 'ibm-q-ornl/ornl/phy147'
    else:
    # elif device in ['auckland','cairo','hanoi','lagos','nazca','cusco','sherbrooke','perth','guadalupe','kolkata','mumbai']:
        return 'ibm-q-research-2/johns-hopkins-un-3/main'


def get_jobs_from_ids(job_ids, provider, backend_name, limit=50):
    if type(job_ids)==dict:
        dict_keys = list(job_ids.keys())
        job_ids   = list(job_ids.values())
    else:
        dict_keys = None
    if type(job_ids)==str:
        job_ids = [job_ids]
    assert type(backend_name)==str

    jobs_bknd = provider.jobs(backend_name=backend_name, limit=limit)
    job_bknd_ids = {job.job_id():job for job in jobs_bknd}

    jobs = {} if dict_keys else []
    for job_id,job in job_bknd_ids.items():
        if job_id in job_ids:
            if not dict_keys:
                jobs += [job]
            else:
                arg = job_ids.index(job_id)
                jobs[dict_keys[arg]] = job
            print("[%d] job found:"%(len(jobs)),job)
        if len(jobs)==len(job_ids):
            break

    if len(jobs)==1:
        return jobs[0]
    return jobs


def complete_count_keys(count):
    if '00' not in count.keys():
        count['00'] = 0 
    if '01' not in count.keys():
        count['01'] = 0 
    if '10' not in count.keys():
        count['10'] = 0 
    if '11' not in count.keys():
        count['11'] = 0 
    return count


def split_1stneighbors(device, qubits):
    if device in ['quito','lima','belem']: # "T" shaped / 5 qubits
        Q1 = [0,2,3]
        Q2 = [1,4]
    elif device in ['guadalupe']: # Fat-person shaped / 16 qubits
        Q1 = [0,2,4,6,10,13,15,11,9,5]
        Q2 = [1,3,8,14,12,7]
    elif device in ['manila']: # Linear shaped / 5 qubits
        Q1 = [0,2,4]
        Q2 = [1,3]
    elif device in ['lagos','nairobi','jakarta','perth']: # "I" shaped / 7 qubits
        Q1 = [0,2,3,4,6]
        Q2 = [1,5]
    elif device in ['mumbai','hanoi','algiers','auckland']: # Infinity shaped / 27 qubits
        Q1 = [0,2,4,6,10,13,15,17,21,24,26,22,20,16,11,9,5]
        Q2 = [1,3,8,14,19,25,23,18,12,7]
    else:
        print("Device not mapped.")
        Q1=Q2=[]
    print("All qubits accounted for:", all(np.sort(Q1+Q2) == qubits))
    return Q1,Q2

def counts2ps(counts, shots, keys=['0']):
    # try:
    #     ps = np.array([val[key]/shots for val in counts ])
    # except:
    #     try:
    #         ps = counts[key]/shots
    #     except:
    #         ps = 0
    ps = np.zeros(len(counts))
    for i,count in enumerate(counts):
        for k in keys:
            if k in count.keys():
                ps[i] += count[k]/shots
    return ps


def round_to_1(x):
    if x==0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x)))))

def sci_not(x):
    if x==0:
        return 0,1
    power = np.floor(np.log10(round_to_1(x)))
    return np.around(x/(10**power),1), power 


# FTTPS
def get_FTTS_circuits(N, pulse=cirq.rx(np.pi), pulse_flip=False):
    freqs = np.linspace(0, np.pi, N // 2 + 1)[:-1]

    q = cirq.GridQubit.rect(1, 1)
    mod_funs = []
    for i, ww in enumerate(freqs):
        mod_fun = np.sign(np.cos(ww * np.arange(1, N + 1)))
        mod_fun[mod_fun == 0] = 1
        mod_fun = (mod_fun + 1) / 2
        mod_funs.append(mod_fun)

    circuits = []
    even = False
    for mod_fun in mod_funs:
        circuit = cirq.Circuit()
        circuit.append(cirq.rx(np.pi / 2).on(q[0]))

        flip = np.concatenate(([0], np.abs(mod_fun[1:] - mod_fun[:-1])))
        for f in flip:
            if f == 1:
                if pulse_flip==True and even==True:
                    circuit.append(cirq.Z.on(q[0]))
                    circuit.append(pulse.on(q[0]))
                    circuit.append(cirq.Z.on(q[0]))
                else:
                    circuit.append(pulse.on(q[0]))
                even = not even
            else:
                circuit.append(cirq.I(q[0]))

        if np.mod(np.sum(flip), 2) == 1:
            circuit.append(cirq.rx(np.pi / 2).on(q[0]))
        else:
            circuit.append(cirq.rx(-np.pi / 2).on(q[0]))
        circuits.append(circuit)
    return circuits


def zeros_FTTPS(m,k, circ_type='cos'):
    zeros_list = []
    for l in range(2*k):
        zero = int((2*l+1)*2**(m-1)/k) if circ_type=='cos' else int(l*2**m/k)
        zeros_list += [zero]
    return zeros_list

def FTTPS_even_circuits(m, num_FTTPS, N, circ_type='cos', pulse_flip=False, measure=True):
    kk = range(num_FTTPS)
    q = cirq.GridQubit.rect(1, 1)
    circuits = []
    for k in kk:
        pulse_locations = zeros_FTTPS(m,k,circ_type=circ_type)
        circuit = cirq.Circuit()
        circuit.append(cirq.rx(np.pi/2).on(q[0]))
        flip = False
        for i in range(N):
            if i in pulse_locations:
                if pulse_flip and flip: circuit.append(cirq.Z.on(q[0]))
                circuit.append(cirq.X.on(q[0]))
                if pulse_flip and flip: circuit.append(cirq.Z.on(q[0]))
                flip = not flip
            else:
                circuit.append(cirq.I.on(q[0]))
        circuit.append(cirq.rx(-np.pi/2).on(q[0]))
        if measure:
            circuit.append(cirq.measure(q[0]))
        circuits.append(circuit)
    return circuits



def get_FTTS_FFs(N=128,worN=8192):
    freqs = np.linspace(0, np.pi, N // 2 + 1)[:-1]

    mod_funs = []
    for i, ww in enumerate(freqs):
        mod_fun = np.sign(np.cos(ww * np.arange(1, N + 1)))
        mod_fun[mod_fun == 0] = 1
        mod_funs.append(mod_fun)

    # This is the filter function matrix for predictions
    Phi = np.array([np.abs(np.fft.fft(mf, n=worN)) ** 2 for mf in mod_funs])/worN/2

    # For reconstruction
    PhiRecon = np.array([np.abs(np.fft.fft(mf, N)) ** 2 for mf in mod_funs])/N/2 
    PhiRecon = PhiRecon[:, :N // 2]
    PhiRecon[1:, :] = PhiRecon[1:, :] * 2
    
    num_gates = np.array([np.sum(np.abs((mf[1:] + 1) / 2 - (mf[:-1] + 1) / 2)) for mf in mod_funs])[:, np.newaxis]

    return Phi, PhiRecon, num_gates


# RB circuits
def generate_RBcircs(rb_type, nseeds, nums, exp, save=True):
    q0 = cirq.GridQubit(0,0)

    # From Cliff XY
    if rb_type == 'cq':

        cliffords = cirq.experiments.qubit_characterizations._single_qubit_cliffords()
        cliffords = cliffords.c1_in_xy
        clifford_mats = np.array([cirq.experiments.qubit_characterizations._gate_seq_to_mats(gates) for gates in cliffords])

        rb_circs = []
        for seed in range(nseeds):
            rb_circs += [[cirq.experiments.qubit_characterizations._random_single_q_clifford(q0,i,cliffords,clifford_mats) for i in nums]]

    # From Qiskit
    elif rb_type == 'qk':
        rb_opts = {}
        rb_opts['length_vector'] = nums
        rb_opts['nseeds'] = nseeds
        rb_opts['rb_pattern'] = [[0]]

        rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)
    
    else:
        print("Not recognized RB type")
    if save:
        pk.dump(rb_circs, open('rb_circs-noise_est-%s-%d.p'%(rb_type,exp),'wb'))
    return rb_circs



# Experiment Batch
def build_circ_batch(rb_circs, rb_type, nums, num_FTTPS=64):
    circuit_batch = {}
    
    # Load RB
    nseeds = len(rb_circs)
    for seed in range(nseeds):
        for idx,circ in enumerate( rb_circs[seed] ):
            rb_length = nums[idx]
            if rb_type == 'cq':
                for q in circ.all_qubits():
                    circ.append(cirq.measure(q))
                qk_circ = cirq2qiskit_XZ(circ)
            elif rb_type == 'qk':
                qk_circ = qiskit2XZ(circ)
            circuit_batch['rb-%s-%s' % (seed,rb_length)] = qk_circ
    
    #Load FTTPS
    N = 2*num_FTTPS
    FTTPS_circuits = get_FTTS_circuits(N, pulse=cirq.X, pulse_flip=False)
    for fttps_idx, circ in enumerate(FTTPS_circuits):
        for q in circ.all_qubits():
            circ.append(cirq.measure(q))
        qk_circ = cirq2qiskit(circ)
        circuit_batch['fttps-%s' % (fttps_idx)] = qk_circ
    
    # Load SPAM
    qr = qk.QuantumRegister(1)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    for i, circ in enumerate(meas_calibs):
        circuit_batch['meas-%d'%i] = circ
    
    return circuit_batch

qk2cirq_dict = {
    'id':cirq.I,
    'x': cirq.X,
    'z': cirq.Z,
    'sx': cirq.X**.5,
    'sxdg': cirq.X**(-.5),
    'y': cirq.Y,
    's': cirq.S,
    'sdg': cirq.inverse(cirq.S),
    'h': cirq.H
   }


cirq_decompose_XZ = {
    cirq.S: [cirq.rz(np.pi/2)],
    cirq.S**-1: [cirq.rz(-np.pi/2)],
    cirq.Y: [cirq.rz(np.pi/2), cirq.X**.5, cirq.rz(2*np.pi), cirq.X**.5, cirq.rz(7*np.pi/2)],
    cirq.H: [cirq.rz(np.pi/2), cirq.X**.5, cirq.rz(np.pi/2)]
    }

# Decompose gates that contain X or sX gates
def decompose_X(circ):
    q = cirq.GridQubit(0, 0)
    dec_circ = cirq.Circuit()

    for gate in circ:
        gate_type = gate.operations[0].gate
        if gate_type in cirq_decompose_XZ.keys():
            for dec_gate in cirq_decompose_XZ[gate_type]:
                dec_circ.append( dec_gate(q) )
        else:
            dec_circ.append( gate.operations[0] )
    return dec_circ


# Convert cirq circuit to qiskit
def cirq2qiskit(circ):
    str_circ_qasm = str(cirq.circuits.QasmOutput(circ, circ.all_qubits()))
    qk_circ = qk.QuantumCircuit.from_qasm_str(str_circ_qasm)
    return qk_circ

# Convert qiskit circuit to cirq
def qiskit2cirq(qk_circ):
    qk_circ = RemoveBarriers()(qk_circ)
    str_qk_qasm = qk.QuantumCircuit.qasm(qk_circ)
    cq_circ = circuit_from_qasm(str_qk_qasm)
    return cq_circ


# Useful dictionary
def cirq2qiskit_XZ(circ, verbose=False):
    if verbose: print("old circ:", circ)
    qk_circ = qk.QuantumCircuit(1,1)
    for moment in circ:
        gate = moment.operations[0].gate
        if gate==cirq.I:
            qk_circ.i(0)
            continue
        try:
            gate_exp = gate.exponent
            gate_basis = (moment**(1/gate_exp)).operations[0].gate if gate_exp!=0 else cirq.I
        except:
            gate_basis = cirq.I
            
        if gate==cirq.MeasurementGate(1, '(0, 0)', ()):
            qk_circ.measure(0,0)
        elif gate_basis==cirq.I:
            qk_circ.i(0)
        elif gate==cirq.X: 
            qk_circ.x(0)
        elif gate==cirq.X**0.5:
            qk_circ.sx(0)
        elif gate==cirq.Y:
            qk_circ.z(0)
            qk_circ.x(0)
        elif gate==cirq.Y**0.5:
            qk_circ.s(0)
            qk_circ.sx(0)
            qk_circ.sdg(0)
        elif gate==cirq.X**-0.5:
            qk_circ.z(0)
            qk_circ.sx(0)
            qk_circ.z(0)
        elif gate==cirq.Y**-0.5:
            qk_circ.sdg(0)
            qk_circ.sx(0)
            qk_circ.s(0)
        else:
            print("Got an unexpected gate:", gate)
            qk_circ.append(cirq2qiskit(cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0))))[0])
    return qk_circ



def get_pink_arma(alpha, power=None, wl=.001*np.pi, wh=.999*np.pi):
    """
    Implementes the approach from
    
         S. Plaszczynski, Fluctuation and Noise Letters7, R1 (2007)
     
    alpha: float between in (0,2] that determines the noise exponent
    power: float to normalize total power
    wl: float normalized frequency cut off for white band at start
    wh: float normalized frequency cut off for 1/f^2 band at end
    
    returns bb, aa np.array of ARMA coefficients (in si.filter form)
    
    """
    Nf = np.ceil(2.5*(np.log10(wh)-np.log10(wl)))
    delp = (np.log10(wh)-np.log10(wl))/Nf
    logps = np.log10(wl)+.5*(1-alpha/2.)*delp + np.arange(Nf)*delp
    logzs = logps+alpha/2.*delp
    ps = 10**(logps)
    zs = 10**(logzs)

    pstx = (1-ps)/(1+ps)
    zstx = (1-zs)/(1+zs)
    bb,aa = si.zpk2tf(zstx,pstx,k=1e-4)
    if power is not None:
        w_pa,h_pa = si.freqz(bb,aa,worN=2048*8, whole=True)
        acv = np.fft.ifft(np.abs(h_pa)**2)
        bb = bb/np.sqrt(acv[0])*np.sqrt(power)
    
    return bb, aa


base_gates = [cirq.X, cirq.Y, cirq.X**.5, cirq.Y**.5, cirq.X**-.5, cirq.Y**-.5 ]

def trace_dist(gate1,gate2):
    return np.abs(.5*np.trace( np.dot( gate1.conj().transpose(), gate2 ) ))**2

def trace_dist_cirq(gate1,gate2):
    gate1 = cirq.unitary(gate1)
    gate2 = cirq.unitary(gate2)
    return np.abs(.5*np.trace( np.dot( gate1.conj().transpose(), gate2 ) ))**2

def belongs2set(gate, gateset):
    for g in gateset:
        if np.isclose(trace_dist_cirq(gate, g),1):
            return True
    return False

def notIZgate(op):
    gate = op.gate
    if gate==cirq.I:
        return False
    elif belongs2set(gate, base_gates):
        return True
    try:
        gate._circuit_diagram_info_(0.) == 'F^-1'
    except:
        pass
    else:
        return True
    exp = gate.exponent
    if exp==0:
        return False
    base = gate**(1/exp)
    if belongs2set(base, base_gates):
        return True
    return False

def notZgate(op):
    gate = op.gate
    if gate==cirq.I:
        return True
    elif belongs2set(gate, base_gates):
        return True
    try:
        gate._circuit_diagram_info_(0.) == 'F^-1'
    except:
        pass
    else:
        return True
    exp = gate.exponent
    if exp==0:
        return False
    base = gate**(1/exp)
    if belongs2set(base, base_gates):
        return True
    return False

def notIgate(op):
    gate = op.gate
    if gate==cirq.I:
        return False
    return True


pauli_gates = {'I':cirq.I,
               'X':cirq.X,
               'Y':cirq.Y,
               'Z':cirq.Z}

def print_decompose_pauli(unitary,digits=2):
    return ' + '.join(['%f %s'%(np.around(trace_dist(unitary,cirq.unitary(v)),digits),k) for k,v in pauli_gates.items() ])


class MyGate(cirq.Gate):
    def __init__(self, gate, name='MyGate'):
        super(MyGate, self)
        self.gate = np.matrix(gate)
        self.name = name
    def _num_qubits_(self):
        return 1
    def _unitary_(self):
        return self.gate
    def _circuit_diagram_info_(self, args):
        return self.name

cosc = lambda x: 2*x*np.sinc(x/2)**2

Id = np.array([[1,0],[0,1]])
pauli_x = np.array([[0,1],[1,0]])
pauli_y = np.array([[0,-1j],[1j,0]])
pauli_z = np.array([[1,0],[0,-1]])

def bloch2dm(v):
    rho = (Id + v[0]*pauli_x + v[1]*pauli_y + v[2]*pauli_z)/2
    return rho

def dm2bloch(rho):
    v = [np.trace(rho@pauli_x), np.trace(rho@pauli_y), np.trace(rho@pauli_z)]
    return np.real(v)

def transpile(circ, backend, qubit=0, basis_gates=['id','x','rz','sx']):
    t_circ = qk.compiler.transpile(circ, backend=backend, basis_gates=basis_gates, 
                                         initial_layout=[qubit], optimization_level=0)
    return t_circ


# ── Markovian noise simulation ────────────────────────────────────────────────
import scipy.linalg as _scipy_linalg

_v0    = np.array([0., 0., 1.])      # |0> Bloch vector
_rho_0 = np.array([[1, 0], [0, 0]])  # |0> density matrix


def _bloch_rz(v, theta):
    """Apply an Rz(theta) rotation to Bloch vector v."""
    return np.array([np.cos(theta)*v[0] - np.sin(theta)*v[1],
                     np.sin(theta)*v[0] + np.cos(theta)*v[1],
                     v[2]])


def Gmat(theta, alpha, mu, eta, beta, eps, dt):
    """
    Generator matrix for the Bloch-vector Lindblad Master Equation.

    Parameters
    ----------
    theta : float  Gate rotation angle (pi=X, pi/2=SX, 0=I).
    alpha : float  Transverse decay rate  gamma/2 + lmbda  (MHz).
    mu    : float  Modified transverse rate  alpha + nu  (MHz).
    eta   : float  Longitudinal decay rate  gamma + nu  (MHz).
    beta  : float  Qubit detuning (MHz).
    eps   : float  Control / amplitude error.
    dt    : float  Gate duration (µs).

    Returns
    -------
    3×3 numpy array
    """
    return np.array([
        [-alpha,              -beta,               0.               ],
        [ beta,               -mu,                -theta*(1+eps)/dt ],
        [ 0.,                  theta*(1+eps)/dt,  -eta              ]
    ])


def sim_exact(circ, spam, gamma, q, beta, lmbda, eps, nu,
              dt=0.035555, backend=None, verbose=0):
    """
    Exact LME simulation of a Qiskit circuit via Bloch-vector matrix exponentials.

    Parameters
    ----------
    circ    : qiskit.QuantumCircuit
    spam    : float  SPAM error probability.
    gamma   : float  Amplitude damping rate 1/T1 (MHz).
    q       : float  Ground-state thermal population.
    beta    : float  Qubit detuning (MHz).
    lmbda   : float  Pure dephasing rate (MHz).
    eps     : float  Control / amplitude error.
    nu      : float  Additional gate dephasing rate (MHz).
    dt      : float  Identity-gate duration in µs (default 0.035555).
    backend : qiskit Backend or None.
    verbose : int    Verbosity level.

    Returns
    -------
    float  Survival probability p(|0⟩).
    """
    import qiskit.compiler as _qkc
    t_circ = _qkc.transpile(
        circ, backend=backend,
        basis_gates=['id', 'x', 'rz', 'sx'],
        initial_layout=[0], optimization_level=0
    )

    alpha = gamma/2 + lmbda
    mu    = alpha + nu
    eta   = gamma + nu
    c     = np.array([0., 0., gamma*(2*q - 1)])

    G_pi   = Gmat(np.pi,   alpha, mu,    eta,   beta/1.5, eps, dt)
    G_pi2  = Gmat(np.pi/2, alpha, mu,    eta,   beta/1.5, eps, dt)
    G_0    = Gmat(0.,       alpha, alpha, gamma, beta,     eps, dt)

    eGt_pi,  Ginv_pi  = _scipy_linalg.expm(G_pi  * dt), _scipy_linalg.inv(G_pi)
    eGt_pi2, Ginv_pi2 = _scipy_linalg.expm(G_pi2 * dt), _scipy_linalg.inv(G_pi2)
    eGt_0,   Ginv_0   = _scipy_linalg.expm(G_0   * dt), _scipy_linalg.inv(G_0)

    vt = [_v0.copy()]
    for Cinst in t_circ:
        instr = Cinst[0]
        name  = instr.name
        if   name == 'rz':
            new_v = _bloch_rz(vt[-1], float(instr.params[0]))
        elif name == 'x':
            new_v = eGt_pi  @ vt[-1] + (eGt_pi  - np.eye(3)) @ Ginv_pi  @ c
        elif name == 'sx':
            new_v = eGt_pi2 @ vt[-1] + (eGt_pi2 - np.eye(3)) @ Ginv_pi2 @ c
        elif name == 'id':
            new_v = eGt_0   @ vt[-1] + (eGt_0   - np.eye(3)) @ Ginv_0   @ c
        elif name in ('barrier', 'measure'):
            if verbose: print(name)
            continue
        else:
            if verbose: print(f'sim_exact: unknown gate {instr}')
            return 0.
        vt.append(np.real(np.around(new_v, 6)).tolist())

    rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z) / 2
    if spam:
        rho_t = (1 - spam)*rho_t + spam * pauli_x @ rho_t @ pauli_x
    return float(np.real(np.trace(rho_t @ _rho_0)))


def gen_T2DD_circs(n_pulses, nc):
    """
    Generate T2 / Hahn-echo / dynamical-decoupling Ramsey circuits (Cirq).

    Parameters
    ----------
    n_pulses : int   0 = free induction (T2*), 1 = Hahn echo, >1 = multi-pulse DD.
    nc       : noise_characterization  Object that supplies timing parameters.

    Returns
    -------
    list of cirq.Circuit
    """
    delays = np.around(
        np.linspace(nc.T2_dt, nc.T2_periods * nc.T2_time * 1e-6, nc.num_T2), 8
    )
    circs = []
    for d in delays:
        n_id = (int(d / nc.I_duration / n_pulses / 2)
                if n_pulses > 0 else int(d / nc.I_duration))
        q_bit = cirq.GridQubit(0, 0)
        circ  = cirq.Circuit()
        circ.append((cirq.X**0.5).on(q_bit))
        if n_pulses == 0:
            circ.append([cirq.I.on(q_bit)] * n_id)
        else:
            for n in range(n_pulses):
                circ.append([cirq.I.on(q_bit)] * n_id)
                if n % 2 == 1:
                    circ.append(cirq.X.on(q_bit))
                else:
                    circ.append([cirq.Z.on(q_bit), cirq.X.on(q_bit), cirq.Z.on(q_bit)])
                circ.append([cirq.I.on(q_bit)] * n_id)
        circ.append((cirq.X**(0.5 if n_pulses % 2 == 1 else -0.5)).on(q_bit))
        circ.append(cirq.measure(q_bit))
        circs.append(circ)
    return circs


def gen_fpw_circuits(num_beta, step_beta):
    """
    Generate Fixed Pulse Width (FPW / P-filter) Qiskit circuits.

    Each circuit applies ``i`` repetitions of (X·Z·X·Z).
    The first circuit (i=0, trivially empty) is dropped.

    Parameters
    ----------
    num_beta  : int  Total number of circuits requested (use nc.num_FPW + 1).
    step_beta : int  Step size in (X·Z·X·Z) repetitions.

    Returns
    -------
    list of qiskit.QuantumCircuit
    """
    circs = []
    for i in np.arange(0, num_beta * step_beta, step_beta):
        circ = qk.QuantumCircuit(1, 1)
        for _ in range(i):
            circ.x(0); circ.z(0); circ.x(0); circ.z(0)
        circ.measure(0, 0)
        circs.append(circ)
    return circs[1:]   # drop the trivial i=0 circuit

class noisy_simulation():
    def __init__(self, backend=None, qubit=0):
        self.backend = backend
        self.qubit = qubit
        self.l_us = self.backend.properties().gate_length('id',self.qubit)*1e6 if backend else 0.035555
        self.noisy_circ = cirq.Circuit()
        self.qubits = cirq.GridQubit(0, 0)

    def bloch_update_id(self, v0=[0,0,1], dt=1, gamma=0, q=1, lmbda=0, Gamma=0, eps=0, beta=0):
        alpha = gamma/2+Gamma+lmbda
        eta = gamma + Gamma
        
        M = np.array([[np.exp(-alpha*dt)*np.cos(beta*dt), -np.exp(-alpha*dt)*np.sin(beta*dt), 0],
                      [np.exp(-alpha*dt)*np.sin(beta*dt), np.exp(-alpha*dt)*np.cos(beta*dt), 0],
                      [0, 0, np.exp(-eta*dt)]])
        c = gamma*(2*q-1)*dt*np.array([0,0,1])
        
        return M @ v0 + c

#     def bloch_update_x(self, v0=[0,0,1], dt=1, gamma=0, q=1, lmbda=0, Gamma=0, theta=0, eps=0, beta=0):
#         alpha = gamma/2+Gamma+lmbda
#         eta = gamma + Gamma
#         gamma_q = gamma*(2*q-1)
#         theta_eps = theta*(1+eps)
#         xi_plus = np.exp(-(alpha+eta)*dt/2)*np.cos(theta_eps) + (1-np.exp(-(alpha-eta)*dt/2))*np.sinc(theta_eps)
#         xi_minus = np.exp(-(alpha+eta)*dt/2)*np.cos(theta_eps) - (1-np.exp(-(alpha-eta)*dt/2))*np.sinc(theta_eps)
        
#         M = np.array([[np.exp(-alpha*dt)*np.cos(beta*dt), -np.sin(beta*dt)*np.sinc(theta_eps), np.sin(beta*dt)*cosc(theta_eps)],
#                       [np.sin(beta*dt)*np.sinc(theta_eps), xi_minus*np.cos(beta*dt), -np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps)],
#                       [np.sin(beta*dt)*cosc(theta_eps), np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps), xi_plus]])
#         c = gamma_q*dt*np.array([0,-cosc(theta_eps),np.sinc(theta_eps)])
        
#         return M @ v0 + c

    def bloch_update_x(self, v0=[0,0,1], dt=1, gamma=0, q=1, lmbda=0, Gamma=0, theta=0, eps=0, beta=0):
        alpha = gamma/2+Gamma+lmbda
        eta = gamma + Gamma
        gamma_q = gamma*(2*q-1)
        theta_eps = theta*(1+eps)
        xi_plus = np.exp(-(alpha+eta)*dt/2)*np.cos(theta_eps) + (1-np.exp(-(alpha-eta)*dt/2))*np.sinc(theta_eps)
        xi_minus = np.exp(-(alpha+eta)*dt/2)*np.cos(theta_eps) - (1-np.exp(-(alpha-eta)*dt/2))*np.sinc(theta_eps)
        
        M = np.array([[np.exp(-alpha*dt), -beta*dt*np.sinc(theta_eps), beta*dt*cosc(theta_eps)],
                      [beta*dt*np.sinc(theta_eps), xi_minus, -np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps)],
                      [beta*dt*cosc(theta_eps), np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps), xi_plus]])
        c = gamma_q*dt*np.array([0,-cosc(theta_eps),np.sinc(theta_eps)])
        
        return M @ v0 + c

    def bloch_update_x_2beta(self, v0=[0,0,1], dt=1, gamma=0, q=1, lmbda=0, Gamma=0, theta=0, eps=0, beta=0):
        alpha = gamma/2+Gamma+lmbda
        eta = gamma + Gamma
        theta_eps = theta*(1+eps)
        xi_yy = np.exp(-(alpha+eta)*dt/2)*(np.cos(theta_eps) + (eta-alpha)*dt/2*np.sinc(theta_eps))
        xi_zz = np.exp(-(alpha+eta)*dt/2)*(np.cos(theta_eps) - (eta-alpha)*dt/2*np.sinc(theta_eps))
        c_theta = 2*(1-np.cos(theta_eps))/theta_eps**2

        M = np.array([[np.exp(-alpha*dt)*np.cos(beta*dt*np.sqrt(c_theta)), -np.exp(-(alpha+eta)*dt/2)*np.sin(beta*dt)*np.sinc(theta_eps), np.exp(-alpha*dt)*np.sin(beta*dt)*cosc(theta_eps)],
                      [np.exp(-(alpha+eta)*dt/2)*np.sin(beta*dt)*np.sinc(theta_eps), xi_yy, -np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps)*np.cos(beta*dt/theta_eps)],
                      [np.exp(-alpha*dt)*np.sin(beta*dt)*cosc(theta_eps), np.exp(-(alpha+eta)*dt/2)*np.sin(theta_eps)*np.cos(beta*dt/theta_eps), xi_zz+(beta*dt)**2/2*c_theta]])
        
        c = gamma*(2*q-1)*dt*np.array([beta*dt/theta_eps,
                                       -cosc(theta_eps),
                                       np.sinc(theta_eps)+(beta*dt/theta_eps)**2])
        
        return M @ v0 + c

    def bloch_update_rz(self, v0=[0,0,1], theta=0.):
        rz_v0 = np.array([np.cos(theta)*v0[0]-np.sin(theta)*v0[1],
                          np.sin(theta)*v0[0]+np.cos(theta)*v0[1],
                          v0[2]])
        return rz_v0

    # Lindblad Master Equation simulator
    def lme_sim(self, circ, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        vt = [np.real(v0)]
        for Cinst in t_circ:
            instruction = Cinst[0]
            name = instruction.name
            if name=='rz':
                theta = float(instruction.params[0])
                new_vt = self.bloch_update_rz(vt[-1], theta=theta)
            elif name=='x':
                new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        theta=np.pi, eps=eps, beta=beta/dt)
            elif name=='sx':
                new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        theta=np.pi/2, eps=eps, beta=beta/dt)
            elif name=='id':
                new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        eps=eps, beta=beta/dt)
            elif name=='barrier' or 'measure':
                continue
            else:
                print("unknown gate/instruction:")
                print(instruction)
                return 0
            vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
        rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
        
        if s:
            rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x 
        
        if type(measure_state)==int:
            ps = np.real(rho_t[measure_state, measure_state]) 
        else:
            ps = np.real(np.trace(rho_t @ measure_state))

        return ps, vt

    # Lindblad Master Equation simulator
    def lme_sim_2beta(self, circ, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        vt = [np.real(v0)]
        for Cinst in t_circ:
            instruction = Cinst[0]
            name = instruction.name
            if name=='rz':
                theta = float(instruction.params[0])
                new_vt = self.bloch_update_rz(vt[-1], theta=theta)
            elif name=='x':
                new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        theta=np.pi, eps=eps, beta=beta/dt)
            elif name=='sx':
                new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        theta=np.pi/2, eps=eps, beta=beta/dt)
            elif name=='id':
                new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                        eps=eps, beta=beta/dt)
            elif name=='barrier' or 'measure':
                continue
            else:
                print("unknown gate/instruction:")
                print(instruction)
                return 0
            vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
        rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
        
        if s:
            rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x 
        
        if type(measure_state)==int:
            ps = np.real(rho_t[measure_state, measure_state]) 
        else:
            ps = np.real(np.trace(rho_t @ measure_state))

        return ps, vt
    

    # Lindblad Master Equation simulator with correlated noise
    def lme_corr_sim(self, circ, S=NullSchWARMAFier(), num_MC=1, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        str_circ = str(t_circ)
#         len_circ = str_circ.count('X') + str_circ.count('I')
        len_circ = len(t_circ)
        noise_trajs = np.reshape(S.gen_noise_instances(cirq.Circuit([cirq.I.on(cirq.GridQubit(1,1))]*len_circ), num_MC=num_MC), (num_MC,len_circ) )
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        
        ps_list = []
        for noise_traj in noise_trajs:
            vt = [np.real(v0)]
            i=0
            for Cinst in t_circ:
                instruction = Cinst[0]
                name = instruction.name
                if name=='rz':
                    theta = float(instruction.params[0])
                    new_vt = self.bloch_update_rz(vt[-1], theta=theta)
                elif name=='x':
                    new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi, eps=eps, beta=(beta)/dt)
                elif name=='sx':
                    new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi/2, eps=eps, beta=(beta)/dt)
                elif name=='id':
                    new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            eps=eps, beta=(beta)/dt)
                elif name=='barrier' or 'measure':
                    continue
                else:
                    print("unknown gate/instruction:")
                    print(instruction)
                    return 0
                if name in ['x','sx','id','rz']:
                    new_vt = self.bloch_update_rz(new_vt, theta=noise_traj[i])
                    i+=1
                vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
            rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
            
            if s: rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x
            
            ps = np.real(rho_t[measure_state, measure_state]) if type(measure_state)==int else np.real(np.trace(rho_t @ measure_state))

            ps_list += [ps]

        return np.mean(ps_list), np.std(ps_list), ps_list

    # Lindblad Master Equation simulator with correlated noise
    def lme_corr_fpw_sim(self, circ, S=NullSchWARMAFier(), num_MC=1, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        str_circ = str(t_circ)
        len_circ = str_circ.count('X') + str_circ.count('I')
        noise_trajs = np.reshape(S.gen_noise_instances(cirq.Circuit([cirq.I.on(cirq.GridQubit(1,1))]*len_circ), num_MC=num_MC), (num_MC,len_circ) )
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        
        ps_list = []
        for noise_traj in noise_trajs:
            vt = [np.real(v0)]
            i=0
            for Cinst in t_circ:
                instruction = Cinst[0]
                name = instruction.name
                if name=='rz':
                    theta = float(instruction.params[0])
                    new_vt = self.bloch_update_rz(vt[-1], theta=theta)
                elif name=='x':
                    new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi, eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='sx':
                    new_vt = self.bloch_update_x(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi/2, eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='id':
                    new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='barrier' or 'measure':
                    continue
                else:
                    print("unknown gate/instruction:")
                    print(instruction)
                    return 0
                if name in ['x','sx','id']: i+=1
                vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
            rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
            
            if s: rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x
            
            ps = np.real(rho_t[measure_state, measure_state]) if type(measure_state)==int else np.real(np.trace(rho_t @ measure_state))

            ps_list += [ps]

        return np.mean(ps_list), np.std(ps_list), ps_list


    # Lindblad Master Equation simulator with correlated noise
    def lme_corr_sim_2beta(self, circ, S=NullSchWARMAFier(), num_MC=1, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        str_circ = str(t_circ)
        len_circ = str_circ.count('X') + str_circ.count('I')
        noise_trajs = np.reshape(S.gen_noise_instances(cirq.Circuit([cirq.I.on(cirq.GridQubit(1,1))]*len_circ), num_MC=num_MC), (num_MC,len_circ) )
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        
        ps_list = []
        for noise_traj in noise_trajs:
            vt = [np.real(v0)]
            i=0
            for Cinst in t_circ:
                instruction = Cinst[0]
                name = instruction.name
                if name=='rz':
                    theta = float(instruction.params[0])
                    new_vt = self.bloch_update_rz(vt[-1], theta=theta)
                elif name=='x':
                    new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi, eps=eps, beta=(beta)/dt)
                elif name=='sx':
                    new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi/2, eps=eps, beta=(beta)/dt)
                elif name=='id':
                    new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            eps=eps, beta=(beta)/dt)
                elif name=='barrier' or 'measure':
                    continue
                else:
                    print("unknown gate/instruction:")
                    print(instruction)
                    return 0
                if name in ['x','sx','id']:
                    new_vt = self.bloch_update_rz(new_vt, theta=noise_traj[i])
                    i+=1
                vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
            rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
            
            if s: rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x
            
            ps = np.real(rho_t[measure_state, measure_state]) if type(measure_state)==int else np.real(np.trace(rho_t @ measure_state))

            ps_list += [ps]

        return np.mean(ps_list), np.std(ps_list), ps_list

    # Lindblad Master Equation simulator with correlated noise
    def lme_corr_fpw_sim_2beta(self, circ, S=NullSchWARMAFier(), num_MC=1, T1=None, q=1, T2=None, beta=0., eps=0., s=0., p=0., rho_0=np.array([[1,0],[0,0]]), measure_state=0): 
        t_circ = qk.compiler.transpile(circ, backend=self.backend, basis_gates=['id','x','rz','sx'], 
                                             initial_layout=[self.qubit], optimization_level=0)
        
        str_circ = str(t_circ)
        len_circ = str_circ.count('X') + str_circ.count('I')
        noise_trajs = np.reshape(S.gen_noise_instances(cirq.Circuit([cirq.I.on(cirq.GridQubit(1,1))]*len_circ), num_MC=num_MC), (num_MC,len_circ) )
        
        dt = self.l_us
        gamma = 1-np.exp(-dt/T1) if T1 else 0
        lmbda = 1-np.exp(-dt/T2) if T2 else 0
        
        v0 = [np.trace(rho_0@pauli_x), np.trace(rho_0@pauli_y), np.trace(rho_0@pauli_z)]
        
        ps_list = []
        for noise_traj in noise_trajs:
            vt = [np.real(v0)]
            i=0
            for Cinst in t_circ:
                instruction = Cinst[0]
                name = instruction.name
                if name=='rz':
                    theta = float(instruction.params[0])
                    new_vt = self.bloch_update_rz(vt[-1], theta=theta)
                elif name=='x':
                    new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi, eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='sx':
                    new_vt = self.bloch_update_x_2beta(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            theta=np.pi/2, eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='id':
                    new_vt = self.bloch_update_id(vt[-1], dt=dt, gamma=gamma/dt, q=q, lmbda=lmbda/dt, Gamma=p/dt, 
                                            eps=eps, beta=(beta+noise_traj[i])/dt)
                elif name=='barrier' or 'measure':
                    continue
                else:
                    print("unknown gate/instruction:")
                    print(instruction)
                    return 0
                if name in ['x','sx','id']: i+=1
                vt.append( list(np.real(np.around(new_vt, 6)) ) )
        
            rho_t = (Id + vt[-1][0]*pauli_x + vt[-1][1]*pauli_y + vt[-1][2]*pauli_z)/2
            
            if s: rho_t = (1-s)*rho_t + s*pauli_x@rho_t@pauli_x
            
            ps = np.real(rho_t[measure_state, measure_state]) if type(measure_state)==int else np.real(np.trace(rho_t @ measure_state))

            ps_list += [ps]

        return np.mean(ps_list), np.std(ps_list), ps_list


    def append_GAD(self, circ=None, q=1, gamma=None):
        if not circ:
            circ = self.noisy_circ
        circ.append(cirq.generalized_amplitude_damp(q,gamma).on(self.qubits))
        if not circ:
            self.noisy_circ = circ
        return circ

    def append_PD(self, circ=None, lmbda=None):
        if not circ:
            circ = self.noisy_circ
        self.noisy_circ.append(cirq.phase_damp(lmbda).on(self.qubits))
        if not circ:
            self.noisy_circ = circ
        return circ

    def append_DEP(self, circ=None, p=None):
        if not circ:
            circ = self.noisy_circ
        self.noisy_circ.append(cirq.depolarize(p).on(self.qubits))
        if not circ:
            self.noisy_circ = circ
        return circ

    def append_noises(self, op, beta=0., circ=None, q=1,gamma=None,lmbda=None,p=None):
        if not circ:
            circ = self.noisy_circ
        if gamma and notZgate(op):
            # self.append_GAD(q,gamma)
            circ.append(cirq.generalized_amplitude_damp(q,gamma).on(self.qubits))
        if lmbda and notZgate(op):
            # self.append_PD(lmbda)
            circ.append(cirq.phase_damp(lmbda).on(self.qubits))
        if p and notIZgate(op):
            # self.append_DEP(p)
            circ.append(cirq.depolarize(p).on(self.qubits))
        if beta and notZgate(op):
            circ.append(cirq.rz(2*beta).on(self.qubits))
        if not circ:
            self.noisy_circ = circ
        return circ

    # With trotter decomposition
    def noisyfy_circ(self, circ, spam=0, S=None, beta=0., epsilon=0., T1=None, q=None, T2=None, p=0., trotter=1):    
        # Schwarmafy circuit
        if S:
            sim = TensorFlowSchWARMASim(circ,S)
            circ = sim.schwarmafier.gen_noisy_circuit(circ)
    
        # Decompose schwarmafied circuit into X gates
        circ = decompose_X(circ)
        
        lmbda = 1-np.exp(-self.l_us/T2) if T2 else 0
        gamma = 1-np.exp(-self.l_us/T1) if T1 else 0

        self.noisy_circ = cirq.Circuit()
        for moment in circ:
            op = moment.operations[0]
            gate = op.gate
            try:
                gate_exp = 0 if gate==cirq.I else gate.exponent 
            except:
                self.noisy_circ.append(gate.on(self.qubits))
                self.append_noises(op, beta=beta, q=q, gamma=gamma, lmbda=lmbda, p=p)
                continue
            gate_basis = cirq.I if gate_exp==0 else (moment**(1/gate_exp)).operations[0].gate

            # I and X gates with detuning and/or coherent errors
            if gate_basis in [cirq.X,cirq.rx(np.pi)]:
                Hamiltonian = np.pi*gate_exp*(1.+epsilon)*qt.sigmax().full()/2 + beta*qt.sigmaz().full()
                U_n = sc.linalg.expm( -1j * Hamiltonian/trotter )
                # X gates allow trotterization for finite pulse width effects
                for _ in range(trotter):
                    self.noisy_circ.append(MyGate(U_n, "X'").on(self.qubits))
                    self.append_noises(op, beta=0, q=q, gamma=gamma/trotter, lmbda=lmbda/trotter, p=p/trotter) # set beta=0 because it's already included in the Hamiltonian
            else:
                # if gate_basis == cirq.I and beta:
                # GAD and PD commute with detuning noise so no trotterization is needed
                if gate_basis is not cirq.I: self.noisy_circ.append(gate.on(self.qubits))
                self.append_noises(op, beta=beta, q=q, gamma=gamma, lmbda=lmbda, p=p)

            # if gate_basis == cirq.I and beta:
            #     # GAD and PD commute with detuning noise so no trotterization is needed
            #     self.noisy_circ.append(cirq.rz(2*beta).on(self.qubits))
            #     self.append_noises(op, q=q, gamma=gamma, lmbda=lmbda, p=p)
            # elif gate_basis in [cirq.X,cirq.rx(np.pi)] and not beta:
            #     # print(trotter)
            #     for _ in range(trotter):
            #         self.noisy_circ.append(cirq.rx(np.pi*gate_exp*(1.+epsilon)/trotter).on(self.qubits))
            #         self.append_noises(op, q=q, gamma=gamma/trotter, lmbda=lmbda/trotter, p=p/trotter)
            # elif gate_basis in [cirq.X,cirq.rx(np.pi)] and beta:
            #     # print(trotter)
            #     Hamiltonian = np.pi*gate_exp*(1.+epsilon)*qt.sigmax().full()/2 + beta*qt.sigmaz().full()
            #     U_n = sc.linalg.expm( -1j * Hamiltonian/trotter )
            #     for _ in range(trotter):
            #         self.noisy_circ.append(MyGate(U_n,'bX').on(self.qubits))
            #         self.append_noises(op, q=q, gamma=gamma/trotter, lmbda=lmbda/trotter, p=p/trotter)
            # else:
            #     self.noisy_circ.append(gate.on(self.qubits))
            #     self.append_noises(op, q=q, gamma=gamma, lmbda=lmbda, p=p)
        if spam:
            self.noisy_circ.append(cirq.bit_flip(spam).on(self.qubits))
        return self.noisy_circ


    def cirq_dmsim(self, circ, S=None, beta=0., epsilon=0., spam=0., p=0., T1=None, q=1, T2=None, shots=1, trotter=1, initial_state=np.array([[1,0],[0,0]]), measure_state=0):
        # Qubits
        self.qubits = list(circ.all_qubits())[0]
        # DM Simulator
        dm_sim = cirq.DensityMatrixSimulator()
        # Simulation
        counts = []
        if S is None:
            self.noisy_circ = self.noisyfy_circ(circ, spam, None, beta, epsilon, T1, q, T2, p, trotter)
        for _ in range(shots):
            if S is not None:
                self.noisy_circ = self.noisyfy_circ(circ,spam,S,beta,epsilon,T1,q,T2,p,trotter)

            self.rho = dm_sim.simulate(self.noisy_circ,initial_state=initial_state).final_density_matrix
            # counts += [ np.real(rho[0,0])*(1-2*spam)+spam ]
            if type(measure_state)==int:
                counts += [ np.real(self.rho[measure_state, measure_state]) ]
            else:
                counts += [ np.real(np.trace(self.rho @ measure_state)) ]
        ps = np.mean(counts)
        std = np.std(counts)
        return ps, std


class noise_characterization():
    def __init__(self, backend=None, qubit=0, m_FTTPS=6, num_T1=10, num_T2=10, num_DEP=10, worN=8192):
        self.backend = backend
        self.qubit = qubit
        self.I_duration = 0.035555*1e-6
        if backend:
            self.max_experiments = self.backend.configuration().max_experiments
            self.I_duration = self.backend.properties().gate_length('id',[self.qubit])
        self.l_us = self.I_duration*1e6
        
        self.m_FTTPS = m_FTTPS
        self.num_FTTPS = 2**self.m_FTTPS
        self.measure_FTTPS = True
        self.num_T1 = num_T1
        self.num_T2 = num_T2
        self.num_DEP = num_DEP
        if backend:
            self.T1_time = np.around(self.backend.properties().t1(self.qubit)*1e6,2)
            self.T2_time = np.around(self.backend.properties().t2(self.qubit)*1e6,2)
        self.T1_periods = 2
        self.T1_dt = 10*self.I_duration
        self.t2_echo = True
        self.T2_periods = 2
        self.T2_dt = 10*self.I_duration
        self.worN = worN
        
        if backend:
            self.ibm_xerr = self.backend.properties().gate_error('x', [self.qubit])
            self.step_DEP = int(np.round(0.01/self.ibm_xerr)/self.num_DEP)
        self.even = True
        self.flip = False
        self.cs_type = 'cos'
        self.rb_lens = 2**np.arange(3,10)
        self.nseeds = 1

    ### Characterization circuits
    # FTTPS
    def generate_FTTPS_circuits(self):
        print("FTTPS num:",self.num_FTTPS)
        self.N_FTTPS = 2*self.num_FTTPS

        if self.even:
            self.FTTPS_circs = FTTPS_even_circuits(self.m_FTTPS, self.num_FTTPS, self.N_FTTPS, self.cs_type, pulse_flip=self.flip, measure=self.measure_FTTPS)
        else:
            self.FTTPS_circs = get_FTTS_circuits(self.N_FTTPS, pulse=cirq.X, pulse_flip=self.flip)

        Phi, PhiRecon, num_gates = get_FTTS_FFs(self.N_FTTPS,self.worN)
        self.Phi = Phi.astype(np.float32)
        self.num_gates = num_gates.astype(np.float32)

        return self.FTTPS_circs

    # T1
    def generate_T1_circuits(self):
        print("T1 (time, num): (%d us, %d)" % (self.T1_time, self.num_T1) ) # T1 time in micro seconds
        self.delays_t1 = np.around(np.linspace(self.T1_dt, self.T1_periods * self.T1_time * 1e-6, self.num_T1),8)

        self.T1_circs = []

        for t1_idx,d in enumerate(self.delays_t1):
            num_Is = int(d/self.I_duration)

            q = cirq.GridQubit(0,0)
            circ = cirq.Circuit()
            circ.append(cirq.X.on(q))
            
            for _ in range(num_Is):
                circ.append(cirq.I.on(q))
            
            circ.append(cirq.X.on(q))
            circ.append(cirq.measure(q))

            self.T1_circs += [circ]

        return self.T1_circs

    # T2
    def generate_T2_circuits(self):
        print("T2 (time, num): (%d us, %d)" % (self.T2_time, self.num_T2) )  # T2 time in microseconds
        self.delays_t2 = np.around(np.linspace(self.T2_dt, self.T2_periods * self.T2_time * 1e-6, self.num_T2),8)

        self.T2_circs = []

        for t2_idx,d in enumerate(self.delays_t2):
            num_Is = int(d/self.I_duration)

            q = cirq.GridQubit(0,0)
            circ = cirq.Circuit()
            circ.append((cirq.X**.5).on(q))
            
            for _ in range(num_Is//2):
                circ.append(cirq.I.on(q))
            
            if self.t2_echo:
                circ.append(cirq.Z.on(q))
                circ.append(cirq.X.on(q))
                circ.append(cirq.Z.on(q))
            
            for _ in range(num_Is//2):
                circ.append(cirq.I.on(q))
            
            circ.append((cirq.X**(.5 if self.t2_echo else -0.5)).on(q))
            circ.append(cirq.measure(q))
            
            self.T2_circs += [circ]

        return self.T2_circs

    # SPAM
    def generate_SPAM_circuits(self):
        qr = qk.QuantumRegister(1)
        self.meas_calibs, self.state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
        return self.meas_calibs



    # Testing circuits
    def generate_RB_circs(self):
        print("RB (max length, nseeds): (%d, %d)" % (self.rb_lens[-1], self.nseeds))
        rb_opts = {'length_vector': self.rb_lens, 'nseeds': self.nseeds, 'rb_pattern': [[0]]}
        self.rb_circs_qk, _ = rb.randomized_benchmarking_seq(**rb_opts)

        self.RB_circs = []
        for i,l in enumerate(self.rb_lens):
            for j in range(self.nseeds):
                rcirc = qiskit2cirq(RemoveBarriers()(self.rb_circs_qk[j][i]))
                qubits = cirq.GridQubit(0,0)
                circ = cirq.Circuit()
                for moment in rcirc[:-1]:
                    circ.append(moment.operations[0].gate.on(qubits))
                self.RB_circs += [circ]
                
        self.num_RB = len(self.RB_circs)

        return self.RB_circs

    def generate_random_circs(self, gate_set=[cirq.X], circuit_lengths=[8], num_random=None):
        self.circuit_lengths = circuit_lengths
        self.num_random = num_random if num_random else int((self.max_experiments-self.num_T1-2*self.num_FTTPS-2)/len(self.circuit_lengths))
        self.batches = range(self.num_random)
        print("circuit lengths", self.circuit_lengths)
        print("batches %d...%d" %(self.batches[0], self.batches[-1]))

        random_circs = {}

        for rc_len in self.circuit_lengths:
            random_circs[rc_len] = []
            for batch in self.batches:
                q = cirq.GridQubit(0,0)
                circ = cirq.Circuit()

                gates = np.random.choice(gate_set, rc_len)
                for gate in gates:
                    circ.append(gate.on(q))
                
                random_circs[rc_len] += [circ]
        
        return random_circs

    ## Generate identity circuits
    def append_cumulative_inverse(self, circ, measure=True):
        q = list(circ.all_qubits())[0]

        F = circ.unitary()
        Finv = np.matrix(F).getH()
        final_gate = MyGate(Finv,'F^-1')
        circ.append(final_gate.on(q))
        if measure:
            circ.append(cirq.measure(q))
        return circ

    def generate_characterization_batch(self, test=False):
        self.circuit_batch = {}

        # Append SPAM circuits
        self.SPAM_circs = self.generate_SPAM_circuits()
        for i, circ in enumerate(self.SPAM_circs):
            self.circuit_batch['spam-%d'%i] = circ

        # Append T1 circuits
        self.T1_circs = self.generate_T1_circuits()
        for t1_idx,circ in enumerate(self.T1_circs):
            qk_circ = cirq2qiskit(circ)
            self.circuit_batch['t1-%s'%t1_idx] = qk_circ

        # Append T2 circuits
        self.T2_circs = self.generate_T2_circuits()
        for t2_idx,circ in enumerate(self.T2_circs):
            qk_circ = cirq2qiskit(circ)
            self.circuit_batch['t2-%s'%t2_idx] = qk_circ

        # Append FTTPS circuits
        self.FTTPS_circs = self.generate_FTTPS_circuits()
        for fttps_idx, circ in enumerate(self.FTTPS_circs):
#             for q in circ.all_qubits():
#                 circ.append(cirq.measure(q))
            qk_circ = cirq2qiskit(circ)
            self.circuit_batch['fttps-%s' % (fttps_idx)] = qk_circ

        self.all_circuits = {'T1': self.T1_circs,
                             'T2': self.T2_circs,
                             'FTTPS': self.FTTPS_circs}

        # Exit if not testing
        if not test:
            return self.circuit_batch

        # Append testing random circuits 
        # cliffords_noIZ = [cirq.X, cirq.Y, cirq.H]
        # cliffords_noZ = [cirq.X, cirq.Y, cirq.H, cirq.I]
        # cliffords = [cirq.X, cirq.Y, cirq.H, cirq.I, cirq.Z]
        # self.random_circs = self.generate_random_circs(gate_set=cliffords, circuit_lengths=self.circuit_lengths, num_random=self.num_random)
        
        # for rc_len in self.circuit_lengths:
        #     for batch in self.batches:
        #         circ = self.random_circs[rc_len][batch]
        #         circ_I = self.append_cumulative_inverse(circ, True)
        #         self.random_circs[rc_len][batch] = circ_I
        #         qk_circ = cirq2qiskit(circ_I)
        #         circuit_batch['random-%d-%d'%(rc_len,batch)] = qk_circ
        self.RB_circs = self.generate_RB_circs()
        for i,l in enumerate(self.rb_lens):
            for j in range(self.nseeds):
                self.circuit_batch['rb-%d'%(self.nseeds*i+j)] = self.rb_circs_qk[j][i]

        self.all_circuits['RB'] = self.RB_circs

        return self.circuit_batch


    def analyze_SPAM_exp(self, results):
        meas_fitter = CompleteMeasFitter(results, self.state_labels, circlabel='mcal')
        meas_filter = meas_fitter.filter

        self.spam = 1 - np.trace(meas_filter.cal_matrix)/2
        return self.spam


    def analyze_T_exp(self, ps_T, delays):
        # ps_T1 = 1-counts2ps(results.get_counts(), self.shots)

        t_fun = lambda t,T_est,a,b: a*np.exp(-t/T_est)+b
        params_t = sc.optimize.curve_fit(t_fun,  delays,  ps_T,  p0=(self.T1_time*1e-6, 1, 0))[0]
        # delays_fine = np.linspace(self.delays[0],self.delays[-1],len(self.delays)*10)
        # ps_T1_fit = t1_fun(delays_fine, *params_t1)
        T_est = params_t[0]
        # self.T1_meas = T1_est*1e6
        return T_est*1e6

    # def analyze_FTTPS_exp(results):


def scan_backend(token, provider, device, avg=False):
    try:
        backend = provider.get_backend('ibmq_'+device)
        print("(q)",end=' ')
    except:
        backend = provider.get_backend('ibm_'+device)
        print("(no-q)",end=' ')
    print("Device:",device)
    n_qubits = backend.configuration().n_qubits
    qubits = range(n_qubits)
    T1s = []
    T2s = []
    for qubit in qubits:
        # 0 if no information is provided
        try:
            T1 = backend.properties().t1(qubit)*1e6
        except:
            T1 = 0 
        try:
            T2 = backend.properties().t2(qubit)*1e6
        except:
            T2 = 0
        print("q = %d: T1 = %dus, T2 = %dus" % (qubit, T1, T2))
        T1s += [T1]
        T2s += [T2]
    if avg:
        print("Avg T1 = %dus"%int(np.mean(T1s)))
        print("Avg T2 = %dus"%int(np.mean(T2s)))


def device_avg_Ts(backend):
    n_qubits = backend.configuration().n_qubits
    qubits = range(n_qubits)
    Ts = [[backend.properties().t1(qubit)*1e6, backend.properties().t2(qubit)*1e6] for qubit in qubits]
    T1_avg = int(np.mean(Ts, axis=0)[0])
    T2_avg = int(np.mean(Ts, axis=0)[1])
    print("Avg T1 = %dus"%T1_avg)
    print("Avg T2 = %dus"%T2_avg)
    return T1_avg, T2_avg


def load_backend(token, provider, device, qubit = None, verbose=False):
    try:
        backend = provider.get_backend('ibmq_'+device)
        if verbose: print("(q)",end='')
    except:
        backend = provider.get_backend('ibm_'+device)
        if verbose: print("(no-q)",end='')
    if verbose: print("Device:",device)
    if qubit is not None:
        print("T1 = %dus, T2 = %dus" % (backend.properties().t1(qubit)*1e6,backend.properties().t2(qubit)*1e6))
    return backend


def zero_cap(x):
    if type(x) in [list, tuple]:
        return [y if y>0 else 0 for y in x]
    return x if x>0 else 0


def compute_average(ps, lengths, batch_len):
    ps_avg = []
    ps_std = []
    for i in range(lengths):
        window = ps[i*batch_len:(i+1)*batch_len]
        window_avg = np.mean(window)
        window_std = np.std(window)
        ps_avg += [window_avg]
        ps_std += [window_std]
    return (ps_avg, ps_std)