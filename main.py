# from bloqade import start, cast, var, save, load
# from bloqade.atom_arrangement import Chain, Square
import numpy as np
# from reservoirpy.datasets import lorenz
from sklearn.preprocessing import MinMaxScaler
# from reservoirpy.nodes import Ridge
from sklearn.metrics import r2_score
from reservoirpy.observables import mse
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from reservoirpy.datasets import lorenz, mackey_glass, logistic_map

datasets = ["mackey_glass", "logistic_map", "ECG", "Stocks", "Lorenz", "HAC"]
n_atoms = [2, 4, 6, 9, 12]
shots = [1, 10, 100, 500, 1000]
shapes = ["Chain", "Square", "Triangular", "Honeycomb"]
readouts = ["Ridge", "NN", "SVR"]

df = pd.read_csv(r'all_runs.csv')
results_df = pd.DataFrame(columns = list(df.columns) + ['mse', "r2"])


# for index in range(40, len(df)):
    # if index<0:
    #     continue

[dataset, lattice, n_atoms, n_shots, readout] = list(df.iloc[index])
n_atoms = int(n_atoms)
n_shots = int(n_shots)

labels = {"mackey_glass": f"Mackey Glass Time Series with {n_atoms} atoms",
     "logistic_map": f"Logistic Map Time Series with {n_atoms} atoms",
       "ECG":  f"MIT-BIH Arrhythmia ECG data with {n_atoms} atoms",
        "Stocks": f"Apple Stocks data with {n_atoms} atoms "}

print(f"Run No. {index+1}: Loading Dataset", end="\r", flush=True)
pulse_data_train, Y_train, pulse_data_test, X_test, Y_test, scaler = get_dataset(dataset)
train_states = []
print(Y_train.shape, X_test.shape, Y_test.shape)

print(f"Run No. {index+1}: Running reservoir on train data", end="\r", flush=True)
register = get_register(lattice, n_atoms)
train_reports = get_reports(pulse_data_train, register, n_shots)

if lattice=='Square' and n_atoms%2!=0:
    n_atoms = 2*(n_atoms//2)
    
train_states = get_reservoir_states(train_reports, n_shots, n_atoms)

print(f"Run No. {index+1}: Training readout on train data", end="\r", flush=True)
actual_readout = get_readout(readout).fit(train_states, Y_train)

print(f"Run No. {index+1}: Running Reservoir on test data", end="\r", flush=True)
test_reports = get_reports(pulse_data_test, register, n_shots)
test_states = get_reservoir_states(test_reports, n_shots, n_atoms)

print(f"Run No. {index+1}: Making Prediction on test states", end="\r", flush=True)
# Y_pred = actual_readout.run(test_states)
Y_pred = normalize_it(actual_readout.run(test_states))

print(f"Run No. {index+1}: Evaluation", end="\r", flush=True)
square_mse = mse(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

if index <60:
    secno = '4'
elif index <70:
    secno = '5'
elif index <76:
    secno = '6'
else:
    secno = '7'

new_dict = {'dataset': dataset, 'lattice': lattice, 'n_atoms': n_atoms, 'n_shots': n_shots, 'readout': readout, 'mse': square_mse, 'r2': r2}
new_df = pd.DataFrame(new_dict, index=[index])
results_df = pd.concat([results_df, new_df], ignore_index=True)
results_df.to_csv(f"Results/metrics.csv")


if dataset=='Lorenz' or dataset=='HAC':
    N = len(Y_test)
    Y_pred = scaler.inverse_transform(Y_pred)
    Y_test = scaler.inverse_transform(Y_test)

    fig = plt.figure(figsize=(15, 10))
    ax  = fig.add_subplot(121, projection='3d')
    ax.set_title("Predicted Dataset")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.grid(False)
    
    for i in range(N-1):
        ax.plot(Y_pred[i:i+2, 0], Y_pred[i:i+2, 1], Y_pred[i:i+2, 2], color=plt.cm.magma((i / (Y_pred.shape[0] - 1)) * 0.4), lw=1.0)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Real Dataset")
    ax2.grid(False)
    
    for i in range(N-2):
        ax2.plot(Y_test[i:i+2, 0], Y_test[i:i+2, 1], Y_test[i:i+2, 2], color=plt.cm.magma((i / (Y_pred.shape[0] - 1)) * 0.4), lw=1.0)

    plt.savefig(f'Results/sec{secno}/{index}_{dataset}_{lattice}_{n_atoms}_{n_shots}_{readout}.png')


else:
    plt.figure(figsize=(10, 3))
    plt.title(labels[dataset])
    plt.xlabel("$t$")
    print(Y_train.shape, Y_pred.shape, Y_test.shape)
    plt.plot(Y_pred, label="Predicted", color="black")
    plt.plot(Y_test, label="Real", color="red")
    plt.legend()
    plt.savefig(f'Results/sec{secno}/{index}_{dataset}_{lattice}_{n_atoms}_{n_shots}_{readout}.png')
