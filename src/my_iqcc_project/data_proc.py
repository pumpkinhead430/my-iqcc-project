import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import graph
from scipy.signal import find_peaks
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

    g = graph.Graph(zip(frequencies, IQ_abs), None, None, None, title="resonator sweep", x_label="pulse frequency", y_label="amplitude", x_unit="[Ghz]", y_unit="[a.u.]", style="line" )
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
    
    g = graph.Graph(zip(amp_lst, IQ_abs), None, None, None, title="bias sweep", x_label="bias flux", y_label="amplitude", x_unit="[a.u.]", y_unit="[a.u.]", style="line" )
    g.plot_fit()
    plt.show()

def qubit_sweep():
    with open(f"{current_dir}\\data\\qubit_sweep.json", "r") as f:
        run_data = json.load(f)
    avg_p, num_points, span, frequency, LO_f = run_data["setup"]
    IQ_abs, phase,I, Q = IQ_extract(run_data)
    
    frequencies = np.linspace(frequency - span / 2 + LO_f, frequency + span / 2 + LO_f, num_points) * 1e-9  # Convert to GHz
    g = graph.Graph(zip(frequencies, IQ_abs), None, None, None, title="qubit drive sweep", x_label="drive frequency", y_label="amplitude", x_unit="[Ghz]", y_unit="[a.u.]", style="line" )
    g.plot_fit()
    plt.show()

    min_idx = np.argmax(IQ_abs)
    resonator_frequency = frequencies[min_idx]*1e9

    #write resonator result to json
    add_results("drive_IF", resonator_frequency - LO_f)


def rabi_find():
    with open(f"{current_dir}\\data\\rabi_find.json", "r") as f:
        run_data = json.load(f)
    avg_p, num_points,amp_lst, LO_f = run_data["setup"]
    IQ_abs, phase,I, Q = IQ_extract(run_data)
    g = graph.Graph(zip(amp_lst, IQ_abs), None, None, None, title="", x_label="drive amplitude", y_label="amplitude", x_unit="[Ghz]", y_unit="[a.u.]" )
    g.plot_fit()
    plt.show()

    peak_indices, _ = find_peaks(IQ_abs,prominence=0.2e-4, distance=20)
    add_results("pi_pulse_amp",amp_lst[peak_indices[0]])

def discrimination():
    with open(f"{current_dir}\\data\\discrimination.json", "r") as f:
        run_data = json.load(f)
    avg_p, LO_f = run_data["setup"]
    I_0, Q_0, I_1, Q_1,_,_,_ = list(run_data["result"].values())
    I_0 =  np.squeeze(np.array(I_0))
    I_1 =  np.squeeze(np.array(I_1))
    Q_0 =  np.squeeze(np.array(Q_0))
    Q_1 =  np.squeeze(np.array(Q_1))
    avg_I_0 = np.average(I_0)
    avg_I_1 = np.average(I_1)
    avg_Q_0 = np.average(Q_0)
    avg_Q_1 = np.average(Q_1)
    m = (avg_Q_0 - avg_Q_1) / (avg_I_0 - avg_I_1)
    b  = avg_Q_0 - m *  avg_I_0

    mid_Q = (avg_Q_0 + avg_Q_1) / 2
    mid_I= (avg_I_0 + avg_I_1) / 2
    perp_m = -1/m
    perp_b = mid_Q - perp_m * mid_I
    x_points = np.linspace(np.min(I_1), np.max(I_0))
    y_points = perp_m * x_points +  perp_b

    sep_line = graph.Graph(zip(x_points, y_points), None, None, None, title="IQ_plane", x_label="I", y_label="Q", x_unit="[a.u.]", y_unit="[a.u.]", label="seperation line", style="line")
    p_0 = graph.Graph(zip(I_0, Q_0), None, None, None, title="IQ_plane", x_label="I", y_label="Q", x_unit="[a.u.]", y_unit="[a.u.]", label="0")
    p_1 = graph.Graph(zip(I_1, Q_1), None, None, None, title="IQ_plane", x_label="I", y_label="Q", x_unit="[a.u.]", y_unit="[a.u.]", label="1")
    p_tot = p_0 +  p_1 + sep_line
    p_tot.plot_fit()
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
    plt.legend()
    plt.show()
    add_results("sep_line", {"m": perp_m, "b" : perp_b})

def consecutive_pulse():
    with open(f"{current_dir}\\data\\consecutive_pulse.json", "r") as f:
        run_data = json.load(f)
    with open(f"{current_dir}/data/results.json", 'r') as f:
        results_data = json.load(f)

    avg_p, num_pulses, LO_f = run_data["setup"]
    m,b = results_data["sep_line"]["m"], results_data["sep_line"]["b"]
    sep_line =  lambda x: m * x + b
    I = np.squeeze(np.array(run_data['result']['I_raw']))
    Q = np.squeeze(np.array(run_data['result']['Q_raw']))
    p_1 = []
    for j in range(num_pulses):
        #p_1.append(sum([1 if sep_line(I[i, j]) < Q[i, j] else 0 for i in range(len(I[:, j]))]) / avg_p)
        plt.plot(I[:, j], Q[:, j], marker='.', linestyle="none", markersize=10)
        plt.show()






def main():
    discrimination()


if __name__ == "__main__":
    main()