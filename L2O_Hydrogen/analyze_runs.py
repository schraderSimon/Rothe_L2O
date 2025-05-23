import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

def analyze_runs(tmin, tmax, tmin_test, tmax_test):
    runs_dir = pathlib.Path("runs")
    best_test_error = float("inf")
    best_config = None
    best_run = None

    all_train = []
    all_test = []
    labels = []

    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue

        config_path = run / "config.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        if cfg["tmin"] != tmin or cfg["tmax"] != tmax or cfg["tmin_test"] != tmin_test or cfg["tmax_test"] != tmax_test:
            continue

        log_path = run / "log.csv"
        if not log_path.exists():
            continue

        data = np.genfromtxt(log_path, delimiter=",")
        if data.ndim == 1:
            data = data[None, :]

        if data.shape[1] != 3:
            continue
        
        epochs, train_errors, test_errors = data.T
        all_train.append(train_errors)
        all_test.append(test_errors)
        labels.append(run.name)
        min_test = np.min(test_errors)
        argmin_test = np.argmin(test_errors)
        best_argmin_test=argmin_test
        if min_test < best_test_error:
            best_test_error = min_test
            best_argmin_test = argmin_test
            best_config = cfg
            best_run = run.name

    for i, (train, label) in enumerate(zip(all_train, labels)):
        plt.plot(epochs,train, label=label)
    plt.title("Train Error")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Train error")
    #plt.legend(fontsize=6)
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0,1.5)

    plt.show()

    for i, (test, label) in enumerate(zip(all_test, labels)):
        plt.plot(epochs,test, label=label)
    plt.title("Test Error")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Test error")
    #plt.legend(fontsize=6)
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0,1.5)

    plt.show()
    print("Best test error:", best_test_error)
    print("Best test epoch:", best_argmin_test)
    print("Best run:", best_run)
    print("Best config:", json.dumps(best_config, indent=2))

if __name__ == "__main__":
    tmin=100
    tmax=200
    tmin_test=210
    tmax_test=330
    analyze_runs(tmin=tmin, tmax=tmax, tmin_test=tmin_test, tmax_test=tmax_test)
