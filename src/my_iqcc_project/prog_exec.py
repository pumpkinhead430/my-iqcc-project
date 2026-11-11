import numpy as np
from iqcc_cloud_client import IQCC_Cloud
import matplotlib.pyplot as plt
from importlib import reload
from qm.qua import *
import matplotlib.pyplot as plt
import json
from pathlib import Path

from my_iqcc_project.config import main_config
reload(main_config)

config = main_config.config
current_dir = Path(__file__).resolve().parent

token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGF1LWRhbmllbGVpemVyLXBoeXNpY3MtbGFiLTMiLCJxcHUiOnsicW9sYWIiOnsiZnJvbSI6MTc5MzM5NzYwMC4wLCJ0byI6MTc5NjA3NjAwMC4wLCJhY2Nlc3MiOltdLCJwcm9qZWN0IjpbXX19LCJleHBpcmVzIjoxNzk2MDc2MDAwLjB9.SkDSbepYGBwTYN_rTE5aUP4KPP3XIYkczqUlD7a3pl_LBsqNc461kZIKEV9fHNNhzTRR3JPi37aMuDrkZll8JjTNFtMgfb16EPxTYBMS_MSv8KWr9aA42_F41Et9gvbBDYBiuSI8vHl7mLcc2zfewyCr7Cm2UQs--jq96MPVblSebyv6U_4Ok62p-TMbkv9dXt5_uehTcUKE35bgzKE-oa_8bLANzzqtoOv_C9Z4H-HWJQ4j1Jj6lB8rxlDdN1tD0wuOTZAEozKf_Tj2ZLGErT-3T0XUyCjcFsmdbA5CoH8t8Nkqd-0F7yEk7yTji9QuOotYAdD_MzuidUpORKMwfA"

def resonator_bias_sweep(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    frequency = config['elements'][res_element]['intermediate_frequency']
    span = 10e6
    num_points = 100
    avg_p = 100
    step_size = span / num_points
    start_if = frequency - (span / 2) # Center your sweep around your base IF
    amp_points = 20
    amp_step = 4/ amp_points

    with program() as prog:
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)
        j = declare(int)
        i = declare(int)
        a = declare(float)
        
        with for_(n, 0, n < avg_p, n + 1):
            assign(a, -2)
            with for_(i, 0, i < amp_points, i + 1):
                with for_(j, 0, j < num_points, j + 1):
                    update_frequency(res_element, start_if + j * step_size)
                    align(res_element, z_element)
                    play("const", z_element,amplitude_scale=a)
                    measure(
                        "readout",
                        res_element,
                        dual_demod.full("iw1", "iw2", I),
                        dual_demod.full("iw3", "iw1", Q),
                    )
                    wait(1000, res_element, z_element)#wait for resonator and z_element to reset
                    save(I, I_st)
                    save(Q, Q_st)
                assign(a, a + amp_step)
        
        with stream_processing():
            I_st.buffer(num_points).buffer(amp_points).save_all("I_raw")
            Q_st.buffer(num_points).buffer(amp_points).save_all("Q_raw")
    return prog, [avg_p, num_points, span, frequency], "resonator_bias_sweep"

def resonator_sweep(qubit,results_data):
    res_element = qubit[1]
    xy_element  =  qubit[0]
    frequency = config['elements'][res_element]['intermediate_frequency']
    span = 50e6        # 10 MHz total span
    num_points = 500  # Number of frequency points
    avg_p = 500
    step_size = span / num_points # 50 kHz step
    start_if = frequency - (span / 2) # Center your sweep around your base IF
    with program() as prog:
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)  # QUA variable for the loop index
        j = declare(int)  # QUA variable for the loop index
        i = declare(int)  # QUA variable for the loop index
        wait(5000, res_element) #reseting qubit
        with for_(n, 0, n < avg_p, n + 1):
            update_frequency(xy_element, results_data["drive_IF"])
            with for_(j, 0, j < num_points, j + 1):
                update_frequency(res_element, start_if + j * step_size)
                measure(
                    "readout",
                    res_element,
                    dual_demod.full("iw1", "iw2", I),
                    dual_demod.full("iw3", "iw1", Q),
                )
                wait(1000, res_element)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(num_points).save_all("I_raw")
            Q_st.buffer(num_points).save_all("Q_raw")
    return prog, [avg_p, num_points, span, frequency], "resonator_sweep"

def bias_sweep(qubit,results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    xy_element =  qubit[0]
    IF_frequency = results_data["resonator_IF"]
    amp_lst = np.linspace(-2, 2, 200)
    step_size = (amp_lst[-1] - amp_lst[0])/len(amp_lst)
    avg_p = 1000
    with program() as prog:
        I = declare(fixed)
        Q = declare(fixed)
        n = declare(int)
        j = declare(int) 
        a = declare(float)
        I_st = declare_stream()
        Q_st = declare_stream()
        wait(5000, res_element, z_element) #reseting qubit
        with for_(n, 0, n < avg_p, n + 1):
            assign(a, amp_lst[0])
            update_frequency(res_element, IF_frequency)
            with for_(j, 0, j < len(amp_lst), j + 1):
                align(res_element, z_element)
                play("const", z_element, amplitude_scale=a)
                measure("readout", res_element, dual_demod.full("iw1", "iw2", I), dual_demod.full("iw3", "iw1", Q))
                wait(1000, res_element, z_element)
                assign(a, a + step_size)
                save(I, I_st)
                save(Q, Q_st)

        with stream_processing():
            I_st.buffer(len(amp_lst)).save_all("I_raw")
            Q_st.buffer(len(amp_lst)).save_all("Q_raw")
        
    return prog, [avg_p, list(amp_lst)], "bias_sweep"

def qubit_sweep(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    xy_element = qubit[0]
    xy_IF =  config['elements'][xy_element]['intermediate_frequency']
    resonator_frequency, bias_amp = results_data["resonator_IF"], results_data["bias_amp"]
    span = 100e6        # 10 MHz total span
    num_points = 100  # Number of frequency points
    avg_p = 500
    step_size = span / num_points # 50 kHz step
    start_if = xy_IF - (span / 2) # Center your sweep around your base IF
    with program() as prog:
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)  # QUA variable for the loop index
        j = declare(int)  # QUA variable for the loop index
        
        with for_(n, 0, n < avg_p, n + 1):
            update_frequency(res_element, resonator_frequency)
            with for_(j, 0, j < num_points, j + 1):
                align()
                update_frequency(xy_element, start_if + j * step_size)
                play("saturation", xy_element)
                align(xy_element, res_element, z_element)

                play("const", z_element, amplitude_scale=bias_amp)
                measure(
                    "readout",
                    res_element,
                    dual_demod.full("iw1", "iw2", I),
                    dual_demod.full("iw3", "iw1", Q),
                )
                wait(1000, res_element)
                save(I, I_st)
                save(Q, Q_st)

        with stream_processing():
            I_st.buffer(num_points).save_all("I_raw")
            Q_st.buffer(num_points).save_all("Q_raw")
    return prog, [avg_p, num_points, span, xy_IF], "qubit_sweep"


def rabi_find(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    xy_element = qubit[0]
    xy_IF =  results_data["drive_IF"]
    resonator_frequency, bias_amp = results_data["resonator_IF"], results_data["bias_amp"]


    
    num_points = 400  # Number of frequency points
    avg_p = 500
    amp_lst = np.linspace(0,2, num_points)
    durations = [int(x) for x in np.arange(4, 5000, 4)]
    step_size  = (amp_lst[-1] - amp_lst[0])/num_points
    with program() as prog:
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        I_st = declare_stream()
        Q_st = declare_stream()
        t = declare(int)  # QUA variable for the loop index
        j = declare(int)  # QUA variable for the loop index
        a  =  declare(fixed)
        
        with for_(j, 0 , j < avg_p, j+1):
            assign(a, amp_lst[0])
            update_frequency(res_element, resonator_frequency)
            with for_each_(t, durations):
                align()
                update_frequency(xy_element, xy_IF)
                play("x180_DragCosine", xy_element,duration=t)
                align(xy_element, res_element, z_element)

                play("const", z_element, amplitude_scale=bias_amp)
                measure(
                    "readout",
                    res_element,
                    dual_demod.full("iw1", "iw2", I),
                    dual_demod.full("iw3", "iw1", Q),
                    )
                wait(20000, res_element)
                save(I, I_st)
                save(Q, Q_st)
                assign(a,  a +  step_size)

        with stream_processing():
            I_st.buffer(len(durations)).save_all("I_raw")
            Q_st.buffer(len(durations)).save_all("Q_raw")
    return prog, [avg_p, num_points, durations], "rabi_find"

def discrimination(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    xy_element = qubit[0]
    xy_IF,  pi_pulse_amp =  results_data["drive_IF"], results_data["pi_pulse_amp"]
    resonator_frequency, bias_amp = results_data["resonator_IF"], results_data["bias_amp"]

    avg_p = 10000


    with program() as prog:
        I_0 = declare(fixed)
        Q_0 = declare(fixed)
        I_st_0 = declare_stream()
        Q_st_0 = declare_stream()
        I_1 = declare(fixed)
        Q_1 = declare(fixed)
        I_st_1 = declare_stream()
        Q_st_1 = declare_stream()
        I = declare(fixed)
        Q = declare(fixed)
        
        n = declare(int)
        wait(500000, z_element, res_element, xy_element)
        with for_(n, 0, n < avg_p, n + 1):
            align()
            play("const", z_element, amplitude_scale=bias_amp)

            update_frequency(res_element, resonator_frequency)
            measure(
                "readout",
                res_element,
                dual_demod.full("iw1", "iw2", I_0),
                dual_demod.full("iw3", "iw1", Q_0),
                )
            wait(20000, xy_element, res_element, z_element)
            save(I_0, I_st_0)
            save(Q_0, Q_st_0)

        with for_(n, 0, n < avg_p, n + 1):
            align()
            update_frequency(xy_element, xy_IF)
            play("x180_DragCosine", xy_element,duration=12)
            align(xy_element, res_element, z_element)
            play("const", z_element, amplitude_scale=bias_amp)
            update_frequency(res_element, resonator_frequency)
            measure(
                "readout",
                res_element,
                dual_demod.full("iw1", "iw2", I_1),
                dual_demod.full("iw3", "iw1", Q_1),
                )
            wait(20000, xy_element, res_element, z_element)
            save(I_1, I_st_1)
            save(Q_1, Q_st_1)

        with stream_processing():
            I_st_0.save_all("I_raw_0")
            Q_st_0.save_all("Q_raw_0")
            I_st_1.save_all("I_raw_1")
            Q_st_1.save_all("Q_raw_1")
    return prog, [avg_p], "discrimination"

def consecutive_pulse(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    xy_element = qubit[0]
    xy_IF,  pi_pulse_amp =  results_data["drive_IF"], results_data["pi_pulse_amp"]
    resonator_frequency, bias_amp = results_data["resonator_IF"], results_data["bias_amp"]
    avg_p = 1000
    num_pulses = 10
    with program() as prog:
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            n = declare(int)
            i = declare(int)
            j = declare(int)
            with for_(i, 0, i < avg_p, i + 1):
                with for_(n, 0, n < num_pulses, n + 1):
                    with for_(j, 0, j < n, j + 1):
                        align()
                        update_frequency(xy_element, xy_IF)
                        play("x180_Square"*  amp(pi_pulse_amp), xy_element)
                        wait(25, xy_element)

                    update_frequency(res_element, resonator_frequency)
                    align()
                    play("const", z_element, amplitude_scale=bias_amp)
                    measure(
                        "readout",
                        res_element,
                        dual_demod.full("iw1", "iw2", I),
                        dual_demod.full("iw3", "iw1", Q),
                        )
                    wait(50000, res_element)#wait for qubit de-excatation
                    save(I, I_st)
                    save(Q, Q_st)
        
            with stream_processing():
                I_st.buffer(num_pulses).save_all("I_raw")
                Q_st.buffer(num_pulses).save_all("Q_raw")
    return prog, [avg_p, num_pulses], "consecutive_pulse"


LO_f = config['controllers']['con1']['fems'][1]['analog_outputs'][1]['upconverter_frequency']
qubit = "Q1"
qubit1 = [f"{qubit}.xy", f"{qubit}.resonator",  f"{qubit}.z"]

def main():
    qc = IQCC_Cloud(quantum_computer_backend="qolab", api_token=token)
    with open(f"{current_dir}/data/results.json", 'r') as f:
        results_data = json.load(f)
    prog, extra, name = rabi_find(qubit1, results_data)
    run_data = qc.execute(prog, config, False)
    run_data["setup"] = extra + [LO_f]
    file_path = current_dir / "data"/ name
    with open(f"{file_path}.json", "w") as f:
        json.dump(run_data, f,indent=4)


if __name__ == '__main__':
        main()