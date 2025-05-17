from bloqade import start, cast, var, save, load
from bloqade.atom_arrangement import Chain, Square
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.nodes import Ridge
from sklearn.metrics import r2_score
from reservoirpy.observables import mse
import matplotlib.pyplot as plt

def normalize_it(array):
    return 2 * (array - np.min(array)) / (np.max(array) - np.min(array)) - 1

def make_detuning(data):
    detun_data = [0]
    durations = [0.5]
    for i in data:
        detun_data.append(55*i)
        durations.append(3)

    detun_data.append(55*data[-1])
    durations.append(1)

    return detun_data, durations
    
# def make_detuning(data):
#     d_zero = -55
#     durations = []
#     detun_data = []
    
#     lower_bounds = [-55 - 0.55 * x for x in data] 
#     upper_bounds = [55 + 0.55 * x for x in data]

#     if len(data)==1:
#         detun_data.append(lower_bounds[0])
#         durations.append(0.1)
#         detun_data.append(lower_bounds[0])
#         durations.append(3.8)
#         detun_data.append(upper_bounds[0])
#         durations.append(0.1)
#         detun_data.append(upper_bounds[0])
#         return detun_data, durations
    
#     else:
#         detun_data.append(lower_bounds[0])
#         durations.append(0.1)
#         for i in range(len(data)):
#                 if i%2==0:
#                     detun_data.append(lower_bounds[i])
#                     durations.append(3.8)
#                     detun_data.append(upper_bounds[i])
#                     if not i==len(data)-2:
#                         durations.append(0.2)
        
#                 else:
#                     detun_data.append(upper_bounds[i])
#                     durations.append(3.8)
#                     detun_data.append(lower_bounds[i])
#                     if not i==len(data)-2:
#                         durations.append(0.2)

#                 if i==len(data)-1:
#                     if len(data)%2==0:
#                         durations.append(0.1)
#                         detun_data.append(lower_bounds[-1])
#                     else:
#                         durations.append(0.1)
#                         detun_data.append(upper_bounds[-1])
        
#     return detun_data, durations

def multi_dim_encoding(X):
    encoded_X = []

    for t, sample in enumerate(X):
        timestep = t/len(X)
        encoded = [(1.0/3.0)*(np.sin(t*(i+1)))*sample[i] for i in range(len(sample))]
        encoded = np.sum(encoded)
        encoded_X.append(encoded)

    return encoded_X

def transform_dataset(data, isMultiDim):
    
    if isMultiDim:
        data = multi_dim_encoding(data)
        
    # durations, detun_data = make_detuning(data)
    final_durations = []
    final_data = []
    for i in range(5, len(data)):
        single_point = []
        single_dur = []
        for j in range(i+1):
            single_point.append(data[j])
        detun_pulse_values, durations = make_detuning(single_point)
        final_data.append(detun_pulse_values)
        final_durations.append(durations)

    
    # print(final_data)
    # print(final_durations)
    keys = ["detun", "dur"]
    values = [final_data, final_durations]
    pulse_data = dict(zip(keys, values))

    return pulse_data

def get_dataset(dataset):

    from reservoirpy.datasets import lorenz, mackey_glass, logistic_map
    isMultiDim = False

    if dataset == "mackey_glass":
        X = mackey_glass(n_timesteps=2000)
        X = normalize_it(X[::2].squeeze().tolist())
        
        
    elif dataset == "logistic_map":
        X = logistic_map(n_timesteps=2000)
        X = normalize_it(X[::2].squeeze().tolist())
        
    elif dataset == "ECG":
        X = np.load('ecg_data.npy')
        X = normalize_it(X[::2].squeeze().tolist())
        
    elif dataset == "Stocks":
        X = np.load('apple_stocks.npy')
        X = normalize_it(X[:1000].squeeze().tolist())
        # X = (np.concatenate((X[:,0],X[:,1],X[:,2]))[::3] - 0.5)*2

    elif dataset == "Lorenz":
        X = lorenz(4000, h=0.025)
        X = X[::4]*2 - 1
        isMultiDim = True

    else:
        X = np.load("HAC.npy")
        X = X[::4]*2 - 1
        isMultiDim = True
        
    scaler = MinMaxScaler()
    if isMultiDim:
        X = scaler.fit_transform(X)
        X = X*2 - 1

    pulse_data = transform_dataset(X, isMultiDim)
    pulse_data_train = {"detun": pulse_data["detun"][:600], "dur": pulse_data["dur"][:600]}
    Y_train = X[6:606]
    pulse_data_test = {"detun": pulse_data["detun"][600:-1], "dur": pulse_data["dur"][600:-1]}
    X_test = X[605:-1]
    Y_test = X[606:]
    
    if not isMultiDim:
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        X_test = X_test.reshape(-1,1)

    return pulse_data_train, Y_train, pulse_data_test, X_test, Y_test, scaler

def get_register(lattice, n_atoms):
    if lattice=='Square':
        register = Square(n_atoms//2, 2, lattice_spacing=8)
    elif lattice=='Chain':
        register = Chain(n_atoms, lattice_spacing=8)
    elif lattice=='Triangular':
        register = Triangular(n_atoms//3, 3, lattice_spacing=8)
    else:
        register = Honeycomb(n_atoms//6, 3, lattice_spacing=8)

    return register

def measure_reservoir(single_pulse, time, register):
    # print(len(single_pulse["dur"]))
    # print(len(single_pulse["detun"]))
    # print(single_pulse["dur"])
    # print(single_pulse["detun"])
    # print(len(single_pulse["dur"]))
    # print(len(single_pulse["detun"]))
    # print(time)
    program = (
        register
        .rydberg.rabi.amplitude.uniform.piecewise_linear([0.1, time, 0.1], [0, 15.8, 15.8, 0])
        .rydberg.detuning.uniform.piecewise_constant(
            durations=single_pulse["dur"],
            values=single_pulse["detun"]
        )
    )
    return program

def get_reports(pulse_data, register, n_shots):
    # print(pulse_data["dur"])
    total_times = [np.sum(i) - 0.2 for i in pulse_data["dur"]]

    pulse_durations_var = var("pulse_durations")
    pulse_values_var = var("pulse_values")
    total_time_var = var("total_time")

    reports = []
    for i in range(len(pulse_data["detun"])):
        single_pulse = {"detun": pulse_data["detun"][i], "dur": pulse_data["dur"][i]}
        reports.append(measure_reservoir(single_pulse, total_times[i], register).bloqade.python().run(n_shots).report())

    return reports

def get_expectation_value(counts, n_shots):
    p0 = counts.get('1', 0) #inverted keys received in output
    p1 = counts.get('0', 0)
    
    return (p1 - p0) / n_shots


def get_reservoir_states(reports, n_shots, n_atoms):
    reservoir_states = []
    for report in reports:
        for r in report.counts():
            d = dict(r)
            res = []
            for i in range(n_atoms):
                counts = {'0': 0, '1': 0}
                for k, v in d.items():
                    counts[k[i]] += v
                res.append(get_expectation_value(counts, n_shots))
            reservoir_states.append(res)
    
    return np.array(reservoir_states)

def get_readout(readout):
    if readout == "Ridge":
        actual_readout = Ridge(ridge=1e-4) # Training states

    elif readout == "NN":
        actual_readout = ScikitLearnNode(
            model=MLPRegressor,
            # name="mlp_readout1",
            model_hypers={
                "hidden_layer_sizes": (4, 4),  # Two-layer MLP with sizes 4, 4
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-4,
                "max_iter": 1000,
                "random_state": 42})
    else:
        # SVR
        actual_readout = ScikitLearnNode(
            model=SVR,  # pass the model class, not an instance
            model_hypers={
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"})

    return actual_readout
