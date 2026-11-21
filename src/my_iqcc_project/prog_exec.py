import os
import numpy as np
from iqcc_cloud_client import IQCC_Cloud
from importlib import reload
from qm.qua import *
import json
from pathlib import Path

from config import main_config
reload(main_config)

config = main_config.config

config = main_config.config
current_dir = Path(__file__).resolve().parent

with open(f"{current_dir}\\data\\token.txt", "r") as f:
    token = f.read()

# Choose a single experiment each run. Part 4: t1, ramsey, echo.
EXPERIMENT = "echo"  # resonator_sweep | resonator_bias_sweep | bias_sweep | qubit_sweep |
                   # rabi_find | discrimination | consecutive_pulse | t1 | ramsey | echo

# QUA time unit: 1 clock cycle = 4 ns on this controller.
CLOCK_NS = 4
RESET_WAIT_NS = 100_000       # Confirm >= ~5*T1 once T1 is known
READOUT_RECOVERY_NS = 2_000
T1_MAX_NS = 120_000
T1_STEP_NS = 2_000
T1_AVG = 200
RAMSEY_MAX_NS = 10_000
RAMSEY_STEP_NS = 40
RAMSEY_AVG = 150
RAMSEY_DETUNING_HZ = 2_000_000
ECHO_MAX_NS = 80_000
ECHO_STEP_NS = 400
ECHO_AVG = 150

# Using the operations already configured for Q1.xy.
# They must be verified experimentally as calibrated X90/X180 gates.
X90_OPERATION = "x90"
X180_OPERATION = "x180"
# The old spectroscopy experiments apply a short Z pulse before readout.
# Keeping this for compatibility does NOT establish a persistent DC flux bias.
USE_Z_PULSE_BEFORE_READOUT = True


def ns_to_cycles(ns):
    return int(round(ns / CLOCK_NS))


def readout_IQ(res_element, I, Q):
    measure(
        "readout", res_element, None,
        dual_demod.full("iw1", "iw2", I),
        dual_demod.full("iw3", "iw1", Q),
    )


def optional_z_readout(z_element, bias_amp):
    if USE_Z_PULSE_BEFORE_READOUT:
        play("const", z_element, amplitude_scale=float(bias_amp))



def resonator_bias_sweep(qubit, results_data):
    z_element = qubit[2]
    res_element = qubit[1]
    frequency = int(round(config['elements'][res_element]['intermediate_frequency']))
    span = 10e6
    num_points = 100
    avg_p = 100
    step_size = int(round(span / num_points))
    start_if = int(round(frequency - (span / 2))) # Center your sweep around your base IF
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
    frequency = int(round(config['elements'][res_element]['intermediate_frequency']))
    span = 50e6        # 10 MHz total span
    num_points = 500  # Number of frequency points
    avg_p = 500
    step_size = int(round(span / num_points)) # frequency step in Hz
    start_if = int(round(frequency - (span / 2))) # Center your sweep around your base IF
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
    IF_frequency = int(round(results_data["resonator_IF"]))
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
    xy_IF = int(round(config['elements'][xy_element]['intermediate_frequency']))
    resonator_frequency = int(round(results_data["resonator_IF"]))
    bias_amp = float(results_data.get("bias_amp", 0.0))
    span = 100e6        # 10 MHz total span
    num_points = 100  # Number of frequency points
    avg_p = 500
    step_size = int(round(span / num_points)) # frequency step in Hz
    start_if = int(round(xy_IF - (span / 2))) # Center your sweep around your base IF
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
    xy_IF = int(round(results_data["drive_IF"]))
    resonator_frequency = int(round(results_data["resonator_IF"]))
    bias_amp = float(results_data.get("bias_amp", 0.0))


    
    avg_p = 200
    # Variable t is in QUA clock cycles, NOT nanoseconds.
    # This is a duration scan of a stretched DRAG pulse, not an amplitude scan.
    # Minimum 12 cycles = 48 ns, the native configured DRAG pulse length.
    # Step 1 cycle (= 4 ns) to resolve fast oscillations without aliasing.
    durations = list(range(12, 250, 1))
    with program() as prog:
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        I_st = declare_stream()
        Q_st = declare_stream()
        t = declare(int)  # QUA variable for the loop index
        j = declare(int)  # QUA variable for the loop index
        
        with for_(j, 0 , j < avg_p, j+1):
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

        with stream_processing():
            I_st.buffer(len(durations)).save_all("I_raw")
            Q_st.buffer(len(durations)).save_all("Q_raw")
    return prog, [avg_p, len(durations), durations], "rabi_find"

def discrimination(qubit, results_data):
    """Interleaved ground/excited single-shot IQ clouds.

    Both states start after an identical relaxation wait, and the excited
    preparation uses exactly the same configured X180 operation as part 4.
    """
    xy_element, res_element, z_element = qubit
    xy_if = int(round(results_data["drive_IF"]))
    resonator_if = int(round(results_data["resonator_IF"]))
    bias_amp = float(results_data.get("bias_amp", 0.0))
    avg_p = 2000

    with program() as prog:
        I_0 = declare(fixed)
        Q_0 = declare(fixed)
        I_1 = declare(fixed)
        Q_1 = declare(fixed)
        I_st_0 = declare_stream()
        Q_st_0 = declare_stream()
        I_st_1 = declare_stream()
        Q_st_1 = declare_stream()
        n = declare(int)
        update_frequency(xy_element, xy_if)
        update_frequency(res_element, resonator_if)
        with for_(n, 0, n < avg_p, n + 1):
            # Ground-state shot (no X drive).
            wait(ns_to_cycles(RESET_WAIT_NS), xy_element, res_element, z_element)
            align(xy_element, res_element, z_element)
            optional_z_readout(z_element, bias_amp)
            readout_IQ(res_element, I_0, Q_0)
            save(I_0, I_st_0)
            save(Q_0, Q_st_0)

            # Excited-state shot (identical reset, then the configured X180).
            wait(ns_to_cycles(RESET_WAIT_NS), xy_element, res_element, z_element)
            align(xy_element, res_element, z_element)
            reset_if_phase(xy_element)
            play(X180_OPERATION, xy_element)
            align(xy_element, res_element, z_element)
            optional_z_readout(z_element, bias_amp)
            readout_IQ(res_element, I_1, Q_1)
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
    xy_IF = int(round(results_data["drive_IF"]))
    resonator_frequency = int(round(results_data["resonator_IF"]))
    bias_amp = float(results_data.get("bias_amp", 0.0))
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
                    wait(ns_to_cycles(RESET_WAIT_NS), xy_element, res_element, z_element)
                    with for_(j, 0, j < n, j + 1):
                        align()
                        update_frequency(xy_element, xy_IF)
                        play(X180_OPERATION, xy_element)
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



# -----------------------------------------------------------------------------
# Part 4: coherence measurements (T1, Ramsey / T2*, Hahn echo / T2)
# -----------------------------------------------------------------------------

def _coherence_params(qubit, results_data):
    xy_element, res_element, z_element = qubit
    missing = [k for k in ("drive_IF", "resonator_IF") if k not in results_data]
    if missing:
        raise KeyError(f"Run spectroscopy and save calibration first: {missing}")
    return (xy_element, res_element, z_element,
            int(round(results_data["drive_IF"])),
            int(round(results_data["resonator_IF"])),
            float(results_data.get("bias_amp", 0.0)))


def _setup_coherence_qua(xy_element, res_element, drive_if, resonator_if):
    update_frequency(xy_element, drive_if)
    update_frequency(res_element, resonator_if)


def t1_experiment(qubit, results_data):
    xy, res, z, drive_if, res_if, bias = _coherence_params(qubit, results_data)
    # A dynamically computed wait of 0 cycles can be invalid on QOP.
    # Start from 16 ns (4 cycles) and save the actual hardware delays.
    delays = list(range(16, T1_MAX_NS + 1, T1_STEP_NS))
    if any(d % CLOCK_NS for d in delays):
        raise ValueError("T1 delays must be multiples of the 4 ns clock")
    with program() as prog:
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)
        j = declare(int)
        _setup_coherence_qua(xy, res, drive_if, res_if)
        with for_(n, 0, n < T1_AVG, n + 1):
            with for_(j, 0, j < len(delays), j + 1):
                wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
                align(xy, res, z)
                reset_if_phase(xy)
                play(X180_OPERATION, xy)
                wait(ns_to_cycles(16) + (T1_STEP_NS // CLOCK_NS) * j, xy)
                align(xy, res, z)
                optional_z_readout(z, bias)
                readout_IQ(res, I, Q)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(len(delays)).save_all("I_raw")
            Q_st.buffer(len(delays)).save_all("Q_raw")
    return prog, {
        "averages": T1_AVG, "tau_ns": delays, "experiment": "t1",
        "drive_IF": drive_if, "resonator_IF": res_if,
        "x180_operation": X180_OPERATION, "reset_wait_ns": RESET_WAIT_NS,
        "z_readout_pulse": USE_Z_PULSE_BEFORE_READOUT,
    }, "t1"


def ramsey_experiment(qubit, results_data):
    xy, res, z, drive_if, res_if, bias = _coherence_params(qubit, results_data)
    delays = list(range(16, RAMSEY_MAX_NS + 1, RAMSEY_STEP_NS))
    if any(d % CLOCK_NS for d in delays):
        raise ValueError("Ramsey delays must be multiples of the 4 ns clock")
    with program() as prog:
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)
        j = declare(int)
        _setup_coherence_qua(xy, res, drive_if + int(RAMSEY_DETUNING_HZ), res_if)
        with for_(n, 0, n < RAMSEY_AVG, n + 1):
            with for_(j, 0, j < len(delays), j + 1):
                wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
                align(xy, res, z)
                # Reset phase once per shot (do not reset between the X90 pulses).
                reset_if_phase(xy)
                play(X90_OPERATION, xy)
                wait(ns_to_cycles(16) + (RAMSEY_STEP_NS // CLOCK_NS) * j, xy)
                play(X90_OPERATION, xy)
                align(xy, res, z)
                optional_z_readout(z, bias)
                readout_IQ(res, I, Q)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(len(delays)).save_all("I_raw")
            Q_st.buffer(len(delays)).save_all("Q_raw")
    return prog, {
        "averages": RAMSEY_AVG, "tau_ns": delays, "experiment": "ramsey",
        "detuning_hz": RAMSEY_DETUNING_HZ, "drive_IF": drive_if,
        "resonator_IF": res_if, "x90_operation": X90_OPERATION,
        "reset_wait_ns": RESET_WAIT_NS,
        "z_readout_pulse": USE_Z_PULSE_BEFORE_READOUT,
    }, "ramsey"


def echo_experiment(qubit, results_data):
    xy, res, z, drive_if, res_if, bias = _coherence_params(qubit, results_data)
    delays = list(range(32, ECHO_MAX_NS + 1, ECHO_STEP_NS))
    if any(d % (2 * CLOCK_NS) for d in delays):
        raise ValueError("Echo total free delays must be multiples of 8 ns")
    with program() as prog:
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n = declare(int)
        j = declare(int)
        _setup_coherence_qua(xy, res, drive_if, res_if)
        with for_(n, 0, n < ECHO_AVG, n + 1):
            with for_(j, 0, j < len(delays), j + 1):
                wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
                align(xy, res, z)
                reset_if_phase(xy)
                play(X90_OPERATION, xy)
                # Total free evolution tau = 2 * j * ECHO_STEP_NS/2.
                wait(ns_to_cycles(16) + (ECHO_STEP_NS // (2 * CLOCK_NS)) * j, xy)
                play(X180_OPERATION, xy)
                wait(ns_to_cycles(16) + (ECHO_STEP_NS // (2 * CLOCK_NS)) * j, xy)
                play(X90_OPERATION, xy)
                align(xy, res, z)
                optional_z_readout(z, bias)
                readout_IQ(res, I, Q)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(len(delays)).save_all("I_raw")
            Q_st.buffer(len(delays)).save_all("Q_raw")
    return prog, {
        "averages": ECHO_AVG, "tau_ns": delays, "experiment": "echo",
        "drive_IF": drive_if, "resonator_IF": res_if,
        "x90_operation": X90_OPERATION, "x180_operation": X180_OPERATION,
        "reset_wait_ns": RESET_WAIT_NS,
        "z_readout_pulse": USE_Z_PULSE_BEFORE_READOUT,
    }, "echo"


# ----------------------------------------------------------------------
# Bonus: driven quantum Zeno effect (non-selective intermediate readout).
# ----------------------------------------------------------------------
ZENO_AVG = 300
ZENO_REFERENCE_AVG = 300
ZENO_SEGMENTS = [1, 2, 3, 4, 6, 8, 12, 16]
ZENO_X180_SCALE = 1.0       # use the existing calibrated pulse amplitude
ZENO_MEASUREMENT_SCALE = 1.0  # never exceeds the configured readout amplitude
ZENO_RINGDOWN_NS = 2000
ZENO_FINAL_Z = USE_Z_PULSE_BEFORE_READOUT  # same final-readout convention as earlier runs


def zeno_experiment(qubit, results_data):
    """Four interleaved arms: driven / undriven, measurement / zero-amplitude sham.

    N scaled X180 pulses have total nominal rotation pi. N-1 intermediate
    readout slots use the same measurement operation in every arm, with
    amplitude 1 (physical measurement) or 0 (timing-matched sham).
    Only final readout IQ is saved; no postselection, feedback or classifier.
    Separate N=1 reference shots define an independent IQ projection axis.
    """
    xy, res, z = qubit
    drive_if = int(round(results_data["drive_IF"]))
    res_if = int(round(results_data["resonator_IF"]))
    bias = float(results_data.get("bias_amp", 0.0))
    ns = list(ZENO_SEGMENTS)
    if len(ns) < 3 or ns[0] != 1 or sorted(set(ns)) != ns or any(type(v) is not int or v < 1 for v in ns):
        raise ValueError("ZENO_SEGMENTS must be increasing distinct positive integers starting at 1.")
    if ZENO_AVG < 2 or ZENO_REFERENCE_AVG < 2:
        raise ValueError("At least two shots per setting are needed for SEM.")
    if not 0 < ZENO_X180_SCALE <= 1 or not 0 < ZENO_MEASUREMENT_SCALE <= 1:
        raise ValueError("Use only positive scales <= the configured amplitude; recalibrate with instructor first.")
    op = X180_OPERATION
    xy_pulse = config["elements"][xy]["operations"][op]
    ro_pulse = config["elements"][res]["operations"]["readout"]
    xy_ns = int(config["pulses"][xy_pulse]["length"])
    ro_ns = int(config["pulses"][ro_pulse]["length"])
    tof_ns = int(config["elements"][res].get("time_of_flight", 0))
    smear_ns = int(config["elements"][res].get("smearing", 0))
    ring_ns = CLOCK_NS * int(np.ceil(max(ZENO_RINGDOWN_NS, tof_ns + 2*smear_ns, 16)/CLOCK_NS))
    arms = {"drive_meas": (True, True), "drive_sham": (True, False),
            "ground_meas": (False, True), "ground_sham": (False, False)}
    print("Zeno settings: drive IF", drive_if, "readout IF", res_if, "bias", bias)
    print("X180 [ns]", xy_ns, "readout [ns]", ro_ns, "ringdown [ns]", ring_ns)
    print("Segments:", ns, "averages:", ZENO_AVG, "final Z:", ZENO_FINAL_Z)

    with program() as prog:
        I, Q = declare(fixed), declare(fixed)
        I_mid, Q_mid = declare(fixed), declare(fixed)
        rep, ref = declare(int), declare(int)
        ks = list(arms) + ["ref_ground", "ref_excited"]
        streams = {key: (declare_stream(), declare_stream()) for key in ks}
        update_frequency(xy, drive_if)
        update_frequency(res, res_if)

        def one_shot(N, drive_on, measure_on, stream_key):
            align(xy, res, z)
            wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
            align(xy, res, z)
            reset_if_phase(xy)  # only once, preserve phase between drive segments
            for j in range(N):  # static unrolling; no real-time divisions
                play(op, xy, amplitude_scale=float(ZENO_X180_SCALE/N if drive_on else 0.0))
                align(xy, res, z)
                if j < N-1:
                    # Identical acquisition instructions for the physical and sham arms.
                    # Intermediate results are ignored intentionally (non-selective).
                    measure("readout", res,
                            dual_demod.full("iw1", "iw2", I_mid),
                            dual_demod.full("iw3", "iw1", Q_mid),
                            amplitude_scale=float(ZENO_MEASUREMENT_SCALE if measure_on else 0.0))
                    align(xy, res, z)
                    wait(ns_to_cycles(ring_ns), xy, res, z)
                    align(xy, res, z)
            # Same final readout (and optional flux pulse) in every arm.
            if ZENO_FINAL_Z:
                play("const", z, amplitude_scale=bias)
            readout_IQ(res, I, Q)
            save(I, streams[stream_key][0])
            save(Q, streams[stream_key][1])
            align(xy, res, z)

        # Independent reference data, never included in the main fit.
        with for_(ref, 0, ref < ZENO_REFERENCE_AVG, ref+1):
            one_shot(1, False, False, "ref_ground")
            one_shot(1, True, False, "ref_excited")
        with for_(rep, 0, rep < ZENO_AVG, rep+1):
            for idx, N in enumerate(ns):
                # Rotate ordering by setting; all four arms are acquired per repeat.
                keys = list(arms)
                for key in keys[idx % 4:] + keys[:idx % 4]:
                    drive_on, measure_on = arms[key]
                    one_shot(N, drive_on, measure_on, key)
        with stream_processing():
            for key, (Is, Qs) in streams.items():
                if key.startswith("ref_"):
                    Is.save_all("I_" + key)
                    Qs.save_all("Q_" + key)
                else:
                    Is.buffer(len(ns)).save_all("I_" + key)
                    Qs.buffer(len(ns)).save_all("Q_" + key)

    return prog, {
        "experiment": "zeno", "schema_version": 1,
        "averages": ZENO_AVG, "reference_averages": ZENO_REFERENCE_AVG,
        "segments": ns, "intermediate_measurements": [n-1 for n in ns],
        "drive_IF": drive_if, "resonator_IF": res_if, "bias_amp": bias,
        "x180_operation": op, "x180_scale": ZENO_X180_SCALE,
        "intermediate_readout_scale": ZENO_MEASUREMENT_SCALE,
        "x180_length_ns": xy_ns, "readout_length_ns": ro_ns,
        "time_of_flight_ns": tof_ns, "ringdown_ns": ring_ns,
        "reset_wait_ns": RESET_WAIT_NS, "final_z_pulse": ZENO_FINAL_Z,
        "intermediate_z_pulse": False, "arms": list(arms),
        "note": "N-1 non-selective readouts, final IQ only; sham uses zero readout amplitude."
    }, "zeno"


# ----------------------------------------------------------------------
# Bonus variant: excited-state survival under a segmented coherent drive.
# ----------------------------------------------------------------------
SURVIVAL_AVG = 300
SURVIVAL_PI_SEGMENTS = 8      # each drive step is a nominal pi/8 rotation
SURVIVAL_STEPS = list(range(9))  # 0 through 8 steps, inclusive
SURVIVAL_X180_SCALE = 1.0
SURVIVAL_READOUT_SCALE = 1.0
SURVIVAL_RINGDOWN_NS = 2000
SURVIVAL_FINAL_Z = USE_Z_PULSE_BEFORE_READOUT
SURVIVAL_CONTROLS = True   # excited state, no further XY drive; diagnostic only


def zeno_survival_experiment(qubit, results_data):
    """Prepare |1>, then rotate toward |0> with / without intermediate readouts.

    For each k, both arms have identical timing and k nominal pi/M rotations.
    Intermediate readouts are ignored (no postselection or feedback).
    A zero-amplitude readout is the timing-matched sham.
    Save final raw IQ and snapshot the EXISTING discriminator for offline use.
    """
    xy, res, z = qubit
    drive_if = int(round(results_data['drive_IF']))
    res_if = int(round(results_data['resonator_IF']))
    bias = float(results_data.get('bias_amp', 0.0))
    w = np.asarray(results_data['discriminator_w'], dtype=float)
    b = float(results_data['discriminator_b'])
    fid = float(results_data['discriminator_balanced_fidelity'])
    if w.shape != (2,) or not np.isfinite(w).all() or not np.linalg.norm(w) > 0 or not np.isfinite(b) or not 0 <= fid <= 1:
        raise ValueError('Invalid saved discriminator. Analyze discrimination first.')
    M, steps = SURVIVAL_PI_SEGMENTS, list(SURVIVAL_STEPS)
    if type(M) is not int or M < 2 or len(steps) < 3 or steps[0] != 0 or steps[-1] != M or sorted(set(steps)) != steps or any(type(k) is not int or not 0 <= k <= M for k in steps):
        raise ValueError('Use ordered integer steps from 0 to SURVIVAL_PI_SEGMENTS.')
    if SURVIVAL_AVG < 2 or not 0 < SURVIVAL_X180_SCALE <= 1 or not 0 < SURVIVAL_READOUT_SCALE <= 1:
        raise ValueError('Invalid averages or amplitude scales; do not exceed calibrated amplitudes.')
    op = X180_OPERATION
    xy_ns = int(config['pulses'][config['elements'][xy]['operations'][op]]['length'])
    ro_ns = int(config['pulses'][config['elements'][res]['operations']['readout']]['length'])
    tof_ns = int(config['elements'][res].get('time_of_flight', 0))
    smear_ns = int(config['elements'][res].get('smearing', 0))
    ring_ns = CLOCK_NS * int(np.ceil(max(SURVIVAL_RINGDOWN_NS, tof_ns + 2*smear_ns, 16)/CLOCK_NS))
    arms = {'meas': (True, True), 'sham': (True, False)}
    if SURVIVAL_CONTROLS:
        arms.update({'idle_meas': (False, True), 'idle_sham': (False, False)})
    print('Zeno survival: drive IF', drive_if, 'readout IF', res_if, 'bias', bias)
    print('Saved assignment fidelity:', fid, '| steps:', steps, '| averages:', SURVIVAL_AVG)
    print('X180 [ns]:', xy_ns, 'readout [ns]:', ro_ns, 'ringdown [ns]:', ring_ns)
    if fid < 0.65:
        print('WARNING: weak saved classifier; final fractions are not calibrated populations.')

    with program() as prog:
        I, Q = declare(fixed), declare(fixed)
        Im, Qm = declare(fixed), declare(fixed)
        rep, k, j = declare(int), declare(int), declare(int)
        streams = {key: (declare_stream(), declare_stream()) for key in list(arms)+['ref_ground']}
        update_frequency(xy, drive_if)
        update_frequency(res, res_if)

        def prepare(excited):
            align(xy, res, z)
            wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
            align(xy, res, z)
            reset_if_phase(xy)
            play(op, xy, amplitude_scale=float(SURVIVAL_X180_SCALE if excited else 0.0))
            align(xy, res, z)

        def final_readout(key):
            if SURVIVAL_FINAL_Z:
                play('const', z, amplitude_scale=bias)
            readout_IQ(res, I, Q)
            save(I, streams[key][0])
            save(Q, streams[key][1])
            align(xy, res, z)

        def one_shot(drive_on, measure_on, key):
            prepare(True)
            # Hardware loop, not Python unrolling over k or j.
            with for_(j, 0, j < k, j+1):
                play(op, xy, amplitude_scale=float(SURVIVAL_X180_SCALE/M if drive_on else 0.0))
                align(xy, res, z)
                with if_(j < k-1):
                    measure('readout', res,
                            dual_demod.full('iw1', 'iw2', Im),
                            dual_demod.full('iw3', 'iw1', Qm),
                            amplitude_scale=float(SURVIVAL_READOUT_SCALE if measure_on else 0.0))
                    align(xy, res, z)
                    wait(ns_to_cycles(ring_ns), xy, res, z)
                    align(xy, res, z)
            final_readout(key)

        with for_(rep, 0, rep < SURVIVAL_AVG, rep+1):
            # Reference is validation only, not training a new discriminator.
            prepare(False)
            final_readout('ref_ground')
            with for_each_(k, steps):
                for key, (drive_on, measure_on) in arms.items():
                    one_shot(drive_on, measure_on, key)
        with stream_processing():
            for key, (Is, Qs) in streams.items():
                if key == 'ref_ground':
                    Is.save_all('I_'+key)
                    Qs.save_all('Q_'+key)
                else:
                    Is.buffer(len(steps)).save_all('I_'+key)
                    Qs.buffer(len(steps)).save_all('Q_'+key)
    return prog, {
        'experiment': 'zeno_survival', 'schema_version': 1,
        'averages': SURVIVAL_AVG, 'steps': steps, 'pi_segments': M,
        'intermediate_readouts': [max(k-1,0) for k in steps],
        'nominal_elapsed_ns': [k*xy_ns+max(k-1,0)*(ro_ns+ring_ns) for k in steps],
        'arms': list(arms), 'drive_IF': drive_if, 'resonator_IF': res_if,
        'bias_amp': bias, 'x180_operation': op, 'x180_scale': SURVIVAL_X180_SCALE,
        'intermediate_readout_scale': SURVIVAL_READOUT_SCALE,
        'x180_length_ns': xy_ns, 'readout_length_ns': ro_ns, 'ringdown_ns': ring_ns,
        'reset_wait_ns': RESET_WAIT_NS, 'final_z_pulse': SURVIVAL_FINAL_Z,
        'discriminator_w': w.tolist(), 'discriminator_b': b, 'discriminator_balanced_fidelity': fid,
        'note': 'Nominal elapsed duration excludes scheduler overhead; same sequence timing within each pair. Final IQ only; no postselection.'
    }, 'zeno_survival'


LO_f = config['controllers']['con1']['fems'][1]['analog_outputs'][1]['upconverter_frequency']
# ----------------------------------------------------------------------
# Bonus: simple segmented pi rotation, using the saved binary classifier.
# ----------------------------------------------------------------------
SIMPLE_ZENO_AVG = 300
SIMPLE_ZENO_SEGMENTS = [1, 2, 3, 4, 6, 8]
SIMPLE_ZENO_RINGDOWN_NS = 2000


def zeno_simple_experiment(qubit, results_data):
    """Start in |0>; N nominal pi/N pulses; N-1 readouts or zero-amplitude sham.

    Save only final IQ. The existing classifier is applied offline, without
    retraining, feedback, postselection, or readout-error correction.
    """
    xy, res, z = qubit
    ns = list(SIMPLE_ZENO_SEGMENTS)
    if len(ns) < 3 or ns[0] != 1 or sorted(set(ns)) != ns or any(type(n) is not int or n < 1 for n in ns):
        raise ValueError('Use distinct increasing positive integers, starting with N=1.')
    if type(SIMPLE_ZENO_AVG) is not int or SIMPLE_ZENO_AVG < 2:
        raise ValueError('At least two repeats are required.')
    if not {'discriminator_w', 'discriminator_b'}.issubset(results_data):
        raise RuntimeError('Run and analyze discrimination first.')
    w = np.asarray(results_data['discriminator_w'], dtype=float)
    b = float(results_data['discriminator_b'])
    fid = float(results_data.get('discriminator_balanced_fidelity', 0.5))
    if w.shape != (2,) or not np.isfinite(w).all() or np.linalg.norm(w) == 0 or not np.isfinite(b) or not 0 <= fid <= 1:
        raise ValueError('Invalid saved discriminator.')
    drive_if = int(round(results_data['drive_IF']))
    res_if = int(round(results_data['resonator_IF']))
    bias = float(results_data.get('bias_amp', 0.0))
    xy_ns = int(config['pulses'][config['elements'][xy]['operations'][X180_OPERATION]]['length'])
    ro_ns = int(config['pulses'][config['elements'][res]['operations']['readout']]['length'])
    tof_ns = int(config['elements'][res].get('time_of_flight', 0))
    smear_ns = int(config['elements'][res].get('smearing', 0))
    ring_ns = CLOCK_NS * int(np.ceil(max(SIMPLE_ZENO_RINGDOWN_NS, tof_ns+2*smear_ns, 16)/CLOCK_NS))
    print('Zeno simple: N =', ns, '| repeats per arm =', SIMPLE_ZENO_AVG)
    print('Drive IF:', drive_if, 'readout IF:', res_if, 'bias:', bias)
    print('X180 [ns]:', xy_ns, 'readout [ns]:', ro_ns, 'ringdown [ns]:', ring_ns)
    print('Saved discriminator fidelity:', fid)
    if fid < 0.65:
        print('WARNING: weak discriminator; classified fractions only.')

    with program() as prog:
        rep, N, j = declare(int), declare(int), declare(int)
        scale = declare(fixed)
        I, Q, Im, Qm = declare(fixed), declare(fixed), declare(fixed), declare(fixed)
        streams = {key: (declare_stream(), declare_stream()) for key in ('meas', 'sham')}
        update_frequency(xy, drive_if)
        update_frequency(res, res_if)

        def one_shot(key, readout_scale):
            align(xy, res, z)
            wait(ns_to_cycles(RESET_WAIT_NS), xy, res, z)
            align(xy, res, z)
            reset_if_phase(xy)
            with for_(j, 0, j < N, j+1):
                play(X180_OPERATION, xy, amplitude_scale=scale)
                align(xy, res, z)
                with if_(j < N-1):
                    # Same acquisition timing in both arms; no intermediate
                    # results saved, no feedback, no selective conditioning.
                    measure('readout', res,
                            dual_demod.full('iw1', 'iw2', Im),
                            dual_demod.full('iw3', 'iw1', Qm),
                            amplitude_scale=readout_scale)
                    align(xy, res, z)
                    wait(ns_to_cycles(ring_ns), xy, res, z)
                    align(xy, res, z)
            optional_z_readout(z, bias)
            readout_IQ(res, I, Q)
            save(I, streams[key][0])
            save(Q, streams[key][1])
            align(xy, res, z)

        with for_(rep, 0, rep < SIMPLE_ZENO_AVG, rep+1):
            # Sweep both N and the precomputed 1/N; no real-time division.
            with for_each_((N, scale), (ns, [1.0/n for n in ns])):
                one_shot('sham', 0.0)
                one_shot('meas', 1.0)
        with stream_processing():
            for key, (Is, Qs) in streams.items():
                Is.buffer(len(ns)).save_all('I_'+key)
                Qs.buffer(len(ns)).save_all('Q_'+key)

    return prog, {
        'experiment': 'zeno_simple', 'schema_version': 1,
        'averages': SIMPLE_ZENO_AVG, 'segments': ns,
        'intermediate_readouts': [n-1 for n in ns],
        'nominal_elapsed_ns': [n*xy_ns+(n-1)*(ro_ns+ring_ns) for n in ns],
        'drive_IF': drive_if, 'resonator_IF': res_if, 'bias_amp': bias,
        'x180_operation': X180_OPERATION, 'x180_length_ns': xy_ns,
        'readout_length_ns': ro_ns, 'ringdown_ns': ring_ns,
        'reset_wait_ns': RESET_WAIT_NS, 'final_z_pulse': USE_Z_PULSE_BEFORE_READOUT,
        'discriminator_w': w.tolist(), 'discriminator_b': b,
        'discriminator_balanced_fidelity': fid,
        'note': 'Two arms only; timing matched at each N, not across N. No intermediate Z pulse. Only final IQ saved.'
    }, 'zeno_simple'


qubit = "Q1"
qubit1 = [f"{qubit}.xy", f"{qubit}.resonator",  f"{qubit}.z"]

EXPERIMENTS = {
    "resonator_sweep": resonator_sweep,
    "resonator_bias_sweep": resonator_bias_sweep,
    "bias_sweep": bias_sweep,
    "qubit_sweep": qubit_sweep,
    "rabi_find": rabi_find,
    "discrimination": discrimination,
    "consecutive_pulse": consecutive_pulse,
    "t1": t1_experiment,
    "ramsey": ramsey_experiment,
    "echo": echo_experiment,
    "zeno": zeno_experiment,
    "zeno_survival": zeno_survival_experiment,
    "zeno_simple": zeno_simple_experiment,
}


def main():
    if not token:
        raise RuntimeError("Set IQCC_API_TOKEN in your environment before running.")
    if EXPERIMENT not in EXPERIMENTS:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")
    results_path = current_dir / "data" / "results.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            results_data = json.load(f)
    else:
        results_data = {}

    prog, extra, name = EXPERIMENTS[EXPERIMENT](qubit1, results_data)
    qc = IQCC_Cloud(quantum_computer_backend="qolab", api_token=token)
    run_data = qc.execute(prog, config, False)
    run_data["setup"] = extra + [LO_f] if isinstance(extra, list) else extra
    output_dir = current_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Existing names retained for compatibility; copy old files before rerunning.
    output_path = output_dir / f"{name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4)
    print(f"Saved {name} to {output_path}")


if __name__ == "__main__":
    main()
