import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import graph
from pathlib import Path
current_dir = Path(__file__).resolve().parent


def add_results(key, value):
    file_path = Path(f"{current_dir}/data/results.json")
    results_data = json.loads(file_path.read_text(encoding="utf-8")) if file_path.exists() else {}
    results_data[key] = value
    file_path.write_text(json.dumps(results_data, indent=4), encoding="utf-8")

def IQ_extract(data):
    I = np.array(data['result']['I_raw'])
    Q = np.array(data['result']['Q_raw'])
    I_clean = np.squeeze(I)
    Q_clean = np.squeeze(Q)
    I_avg = np.mean(I_clean, axis=0)
    Q_avg = np.mean(Q_clean, axis=0)
    return [np.abs(I_avg + 1j * Q_avg), np.atan2(Q_avg,I_avg), I_avg, Q_avg]

def resonator_bias_sweep():
    with open(f"{current_dir}/data/resonator_bias_sweep.json", "r") as f:
        run_data = json.load(f)

    avg_p, num_points, span, frequency, LO_f = run_data["setup"]
    IQ_abs,_, _, _ = IQ_extract(run_data)

    frequencies = np.linspace(frequency - span / 2 + LO_f, frequency + span / 2 + LO_f, num_points) * 1e-9  # Convert to GHz
    im = plt.imshow(IQ_abs, extent=[frequencies[0], frequencies[-1], -1, 1], aspect='auto', origin='lower', cmap='viridis')
    
    plt.colorbar(im, label='Amplitude')
    plt.xlabel("Resonator RF Frequency (GHz)")
    plt.ylabel("Z Flux Amplitude (a.u.)")
    plt.title("Resonator Spectroscopy vs. Z Flux")
    plt.legend(loc="upper right")
    plt.show()




def resonator_sweep():
    with open(f"{current_dir}/data/resonator_sweep.json", "r") as f:
        run_data = json.load(f)
    
    avg_p, num_points, span, frequency, LO_f = run_data["setup"]

    IQ_abs,_, _, _ = IQ_extract(run_data)
    frequencies = np.linspace(frequency - span / 2 + LO_f, frequency + span / 2 + LO_f, num_points) * 1e-9  # Convert to GHz

    min_idx = np.argmin(IQ_abs)
    resonator_frequency = frequencies[min_idx]*1e9

    #write resonator result to json
    add_results("resonator_IF", resonator_frequency - LO_f)

    g = graph.Graph(zip(frequencies, IQ_abs), None, None, None, title="resonator sweep", x_label="pulse frequency", y_label="amplitude", x_unit="[Ghz]", y_unit="[a.u.]" )
    g.plot_fit()
    plt.show()



def bias_sweep():
    with open(f"{current_dir}\\data\\bias_sweep.json", "r") as f:
        run_data = json.load(f)
    avg_p, amp_lst, LO_f = run_data["setup"]
    IQ_abs, _,_, _ = IQ_extract(run_data)
    bias_idx = np.argmin(IQ_abs)

    #write bias result to json
    add_results("bias_amp", amp_lst[bias_idx])
    
    g = graph.Graph(zip(amp_lst, IQ_abs), None, None, None, title="bias sweep", x_label="bias flux", y_label="amplitude", x_unit="[a.u.]", y_unit="[a.u.]" )
    g.plot_fit()
    plt.show()


def qubit_sweep():
    with open(f"{current_dir}\\data\\qubit_sweep.json", "r") as f:
        run_data = json.load(f)
    avg_p, num_points, span, frequency, LO_f = run_data["setup"]
    IQ_abs, phase,I, Q = IQ_extract(run_data)
    
    frequencies = np.linspace(frequency - span / 2 + LO_f, frequency + span / 2 + LO_f, num_points) * 1e-9  # Convert to GHz
    g = graph.Graph(zip(frequencies, IQ_abs), None, None, None, title="qubit drive sweep", x_label="drive frequency", y_label="amplitude", x_unit="[Ghz]", y_unit="[a.u.]" )
    g.plot_fit()
    plt.show()

    min_idx = np.argmax(IQ_abs)
    resonator_frequency = frequencies[min_idx]*1e9

    #write resonator result to json
    add_results("drive_IF", resonator_frequency - LO_f)

def main():
    qubit_sweep()

if __name__ == "__main__":
    main()