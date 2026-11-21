"""Parts 2, 3 and 4 analysis for the user's Q1 IQCC project.

Data files keep the names used by prog_exec_new.py: resonator_sweep,
resonator_bias_sweep, bias_sweep, qubit_sweep, rabi_find, discrimination,
consecutive_pulse, plus new t1, ramsey and echo.

Style: classic matplotlib, white figure, grid, blue data/errorbars, red fit,
separate residuals graph, reduced chi-square and p-value in the legend.

Calibration candidate results are *not* automatically accepted or written over
existing calibrated IFs, unless explicitly requested.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_distribution, binomtest

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RESULTS_PATH = DATA_DIR / "results.json"
REPORT_PATH = DATA_DIR / "fit_reports.json"

ANALYSIS = "echo"  # resonator_sweep | resonator_bias_sweep | bias_sweep | qubit_sweep |
                 # rabi_find | readout_amp_sweep | discrimination | consecutive_pulse |
                 # t1 | ramsey | echo | compare

AUTO_SAVE_SPECTROSCOPY_CALIBRATION = False  # deliberate: protects existing IF values
MIN_VALID_DISCRIMINATION_FIDELITY = 0.65
CLOCK_NS = 4.0
MAX_ERROR_BARS = 60  # plot every point; draw vertical bars for <=60 points
RANDOM_SEED = 12345

plt.style.use("classic")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.7,
    "font.size": 11, "axes.titlesize": 14,
    "lines.markersize": 4,
})


# ----------------------------------------------------------------------
# File helpers; compatible with the list-based setups in parts 2 and 3.
# ----------------------------------------------------------------------
def read_file(name):
    path = DATA_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run '{name}' in prog_exec_234.py first.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def saved_results():
    if not RESULTS_PATH.exists():
        return {}
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_results(values):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    previous = saved_results()
    previous.update(to_jsonable(values))
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(previous, f, indent=4, allow_nan=False)


def save_report(name, report):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    if REPORT_PATH.exists():
        with REPORT_PATH.open(encoding="utf-8") as f:
            all_reports = json.load(f)
    all_reports[name] = to_jsonable(report)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=4, allow_nan=False)


def to_jsonable(v):
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list, np.ndarray)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def as_iq(data, keys=("I_raw", "Q_raw"), target_shape=None):
    arrays = []
    for key in keys:
        if key not in data["result"]:
            raise KeyError(f"Missing result['{key}'] in JSON file")
        arr = np.asarray(data["result"][key], dtype=float)
        arr = np.squeeze(arr)
        if target_shape is not None:
            if arr.size != int(np.prod(target_shape)):
                raise ValueError(f"{key}: shape {arr.shape} / size {arr.size}; expected {target_shape}")
            arr = arr.reshape(tuple(target_shape))
        arrays.append(arr)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(f"I and Q have different shapes: {arrays[0].shape}, {arrays[1].shape}")
    return tuple(arrays)


def get_lo_from_config(element="Q1.xy"):
    # The old setup field LO_f contains the *resonator* LO, even in qubit_sweep.
    # Read the XY output's own LO to avoid a wrong RF frequency axis.
    try:
        try:
            from config import main_config
        except ModuleNotFoundError:
            from my_iqcc_project.config import main_config
        cfg = main_config.config
        controller, fem, port = cfg["elements"][element]["MWInput"]["port"]
        return float(cfg["controllers"][controller]["fems"][fem]["analog_outputs"][port]["upconverter_frequency"])
    except Exception as exc:
        raise RuntimeError("Cannot determine XY LO from config. Run inside the project with its config package.") from exc


def mean_iq_amplitude_and_sem(I, Q):
    """Amplitude A=|mean(I+iQ)| and its propagated 1-sigma SEM.

    Axis 0 contains independent repeated scans. Covariance between I and Q
    is included in the delta-method variance of A.
    """
    I, Q = np.asarray(I, float), np.asarray(Q, float)
    if I.shape != Q.shape or I.ndim < 2:
        raise ValueError(f"Expected I/Q shape (averages, scan points...), got {I.shape}, {Q.shape}")
    n = I.shape[0]
    mi, mq = np.mean(I, axis=0), np.mean(Q, axis=0)
    a = np.hypot(mi, mq)
    if n < 2:
        raise ValueError("At least 2 repeats are required to measure a statistical SEM")
    di, dq = I-mi, Q-mq
    vi = np.sum(di**2, axis=0) / (n*(n-1))
    vq = np.sum(dq**2, axis=0) / (n*(n-1))
    cov = np.sum(di*dq, axis=0) / (n*(n-1))
    da_di = np.divide(mi, a, out=np.zeros_like(a), where=a>1e-15)
    da_dq = np.divide(mq, a, out=np.zeros_like(a), where=a>1e-15)
    va = da_di**2*vi + da_dq**2*vq + 2*da_di*da_dq*cov
    va = np.where(a>1e-15, va, vi+vq)  # gradient undefined at zero
    sa = np.sqrt(np.maximum(0, va))
    return mi, mq, a, ensure_errors(sa)


def iq_projected_shots(I, Q):
    """PCA IQ projection, retaining per-shot samples for statistical SEM."""
    I = np.asarray(I, float)
    Q = np.asarray(Q, float)
    center = np.stack([I.mean(axis=0), Q.mean(axis=0)], axis=-1)
    origin = center.mean(axis=0)
    _, _, vh = np.linalg.svd(center - origin, full_matrices=False)
    d = vh[0]
    shots = (I - origin[0])*d[0] + (Q - origin[1])*d[1]
    return shots.mean(axis=0), ensure_errors(shots.std(axis=0, ddof=1)/np.sqrt(I.shape[0]))


def ensure_errors(sigma, fallback=None):
    sigma = np.asarray(sigma, float)
    good = sigma[np.isfinite(sigma) & (sigma>0)]
    floor = (max(float(np.median(good))*1e-6, 1e-12) if good.size else
             (float(fallback) if fallback is not None else 1e-12))
    return np.where(np.isfinite(sigma) & (sigma>floor), sigma, floor)


def wilson_p(k, n, z=1.0):
    """Binomial Wilson 68.3% (1-sigma-equivalent) confidence interval."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k/n
    denom = 1+z*z/n
    c = (p + z*z/(2*n))/denom
    h = z/denom * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, np.maximum(p - np.clip(c-h,0,1),0), np.maximum(np.clip(c+h,0,1)-p,0)


def get_discriminator():
    results = saved_results()
    if "discriminator_w" not in results or "discriminator_b" not in results:
        raise RuntimeError("Analyze discrimination before using classified populations.")
    w = np.asarray(results["discriminator_w"],float)
    b = float(results["discriminator_b"])
    fid = float(results.get("discriminator_balanced_fidelity",0.5))
    return w, b, fid


def p1_estimate(I, Q):
    w, b, fid = get_discriminator()
    assignments = ((w[0]*I+w[1]*Q+b)>0).astype(int)
    n = assignments.shape[0]
    p,lo,hi = wilson_p(assignments.sum(axis=0), n)
    sy = ensure_errors((lo+hi)/2, fallback=0.5/np.sqrt(n))
    return p, np.vstack([lo,hi]), sy, fid


# ----------------------------------------------------------------------
# Scientific fitting and uniform plotting: classic blue/white/red style.
# ----------------------------------------------------------------------
def robust_fit(model, x, y, sy, guess, bounds, names):
    x, y = np.asarray(x,float), np.asarray(y,float)
    sy = ensure_errors(sy)
    par, cov = curve_fit(model, x, y, p0=guess, bounds=bounds,
                         sigma=sy, absolute_sigma=True, maxfev=250000)
    err = np.sqrt(np.maximum(0, np.diag(cov)))
    residual = y-model(x,*par)
    chi_sq = float(np.sum((residual/sy)**2))
    dof = len(y)-len(par)
    reduced = float(chi_sq/dof) if dof>0 else float("nan")
    p_value = float(chi2_distribution.sf(chi_sq,dof)) if dof>0 else float("nan")
    report = {
        "parameters": {name: {"value":float(p), "error_1sigma":float(e)} for name,p,e in zip(names,par,err)},
        "chi_square":chi_sq, "dof":dof, "reduced_chi_square":reduced,
        "p_value":p_value, "residuals":residual.tolist(),
        "y_sem_used":sy.tolist(),
        "fit_method":"Weighted nonlinear least squares with absolute_sigma=True",
    }
    return par,err,report


def show_fit(x, y, sy, model, parameters, report, title, xlabel, ylabel, y_asym=None):
    x,y,sy=np.asarray(x,float),np.asarray(y,float),ensure_errors(sy)
    err_plot = y_asym if y_asym is not None else sy
    grid = np.linspace(np.min(x),np.max(x),max(500,len(x)*5))
    sampled=np.linspace(0,len(x)-1,min(len(x),MAX_ERROR_BARS),dtype=int)
    legend = (rf"Fit: $\chi^2_{{red}}$={report['reduced_chi_square']:.3g}, "
              rf"p={report['p_value']:.3g}")
    plt.style.use("classic")
    fig,ax=plt.subplots(figsize=(9,5),facecolor="w")
    ax.grid(True)
    # Plot all points opaque, blue; avoid dense overlapping errorbars.
    ax.plot(x,y,"bo",markersize=4,label="Measurement data")
    ax.errorbar(x[sampled],y[sampled],
                yerr=err_plot[:,sampled] if np.ndim(err_plot)==2 else err_plot[sampled],
                fmt="none",ecolor="b",elinewidth=0.9,capsize=2)
    ax.plot(grid,model(grid,*parameters),"r-",linewidth=1.5,label=legend)
    ax.set(title=title+" - Fit",xlabel=xlabel,ylabel=ylabel)
    ax.legend()
    fig.tight_layout()

    res=y-model(x,*parameters)
    fig2,ax2=plt.subplots(figsize=(9,4),facecolor="w")
    ax2.grid(True)
    ax2.plot(x,res,"bo",markersize=4,label="Residuals")
    ax2.errorbar(x[sampled],res[sampled],yerr=sy[sampled],fmt="none",
                 ecolor="b",elinewidth=0.9,capsize=2)
    ax2.axhline(0,color="r",linestyle="--")
    ax2.set(title=title+" - Residuals",xlabel=xlabel,ylabel="Measured - Fit ("+ylabel+")")
    ax2.legend()
    fig2.tight_layout()


def show_data(x,y,sy,title,xlabel,ylabel):
    plt.style.use("classic")
    fig,ax=plt.subplots(figsize=(9,5),facecolor="w")
    ax.grid(True)
    ax.errorbar(x,y,yerr=sy,fmt="bo",linestyle="-",capsize=2,label="Measurement data")
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.legend()
    fig.tight_layout()


def print_report(label, report):
    print("\n=======",label,"=======")
    for name,v in report["parameters"].items():
        print(f"{name}: {v['value']:.6g} +/- {v['error_1sigma']:.3g}")
    print(f"reduced chi-square = {report['reduced_chi_square']:.5g}")
    print(f"p-value            = {report['p_value']:.5g}")
    print("============================\n")


def signed_lorentzian(f_mhz,c,a,f0,gamma):
    return c+a/(1+((f_mhz-f0)/gamma)**2)


def exp_decay(t_us,c,a,T_us):
    return c+a*np.exp(-np.maximum(0,t_us)/T_us)


def ramsey_model(t_us,c,a,T_us,f_mhz,phase):
    return c+a*np.exp(-np.maximum(0,t_us)/T_us)*np.cos(2*np.pi*f_mhz*t_us+phase)


def parity_model(n, c, contrast, lam):
    return c-contrast*(-lam)**n/2


def rabi_model(t_us,c,a,f_mhz,T_us,phase):
    return c+a*np.cos(2*np.pi*f_mhz*t_us+phase)*np.exp(-np.maximum(0,t_us)/T_us)


def flat_signal(y,sy):
    y=np.asarray(y,float)
    sy=ensure_errors(sy)
    # Variation must be larger than several standard errors to be resolvable.
    return float(np.percentile(y,90)-np.percentile(y,10)) < 3*float(np.median(sy))


def fit_spectrum(name, lo_source="resonator"):
    data=read_file(name)
    avg,n_points,span,center_if,stored_res_lo=data["setup"]
    I,Q=as_iq(data,target_shape=(int(avg),int(n_points)))
    _,_,y,sy=mean_iq_amplitude_and_sem(I,Q)
    # The original program uses span/N as the step and stops before upper limit.
    if_values=float(center_if)-float(span)/2 + np.arange(int(n_points))*(float(span)/int(n_points))
    rf_lo=(float(stored_res_lo) if lo_source=="resonator" else get_lo_from_config("Q1.xy"))
    rf_ghz=(rf_lo+if_values)/1e9
    title="resonator sweep" if name=="resonator_sweep" else "qubit drive sweep"
    xlabel="pulse frequency [GHz]" if lo_source=="resonator" else "drive frequency [GHz]"
    if flat_signal(y,sy):
        show_data(rf_ghz,y,sy,title,xlabel,"amplitude [a.u.]")
        print(name+": No resolved resonance above measurement noise; fit skipped")
        save_report(name,{"status":"no_resonance_resolved","uncertainty":"propagated I/Q SEM"})
        plt.show()
        return
    x_mhz=if_values/1e6
    edge=max(3,len(y)//10)
    background=float(np.median(np.r_[y[:edge],y[-edge:]]))
    peak=int(np.argmax(abs(y-background)))
    p0=[background,float(y[peak]-background),float(x_mhz[peak]),max(float(span/1e6)/20,0.05)]
    bounds=([-np.inf,-np.inf,float(x_mhz.min()),0.001],
            [np.inf,np.inf,float(x_mhz.max()),2*float(span/1e6)])
    try:
        par,err,rep=robust_fit(signed_lorentzian,x_mhz,y,sy,p0,bounds,
                               ["offset","amplitude","center_IF_MHz","HWHM_MHz"])
    except (ValueError,RuntimeError) as e:
        print(f"{name}: Lorentzian fit failed: {e}")
        show_data(rf_ghz,y,sy,title,xlabel,"amplitude [a.u.]")
        plt.show()
        return
    def fit_in_rf_ghz(x,c,a,f0,gamma):
        return signed_lorentzian((x*1e9-rf_lo)/1e6,c,a,f0,gamma)
    show_fit(rf_ghz,y,sy,fit_in_rf_ghz,par,rep,title,xlabel,"amplitude [a.u.]")
    rep.update({"uncertainty":"SEM propagated through |mean(I+iQ)| including IQ covariance",
                "x_uncertainty":"not known; sweep spacing is not instrument uncertainty",
                "fit_model":"phenomenological Lorentzian; inspect residuals",
                "RF_LO_Hz":rf_lo,
                "center_IF_candidate_Hz":float(par[2]*1e6),
                "center_IF_stat_error_Hz":float(err[2]*1e6)})
    save_report(name,rep)
    print_report(name,rep)
    print("Candidate IF:",par[2]*1e6,"+/-",err[2]*1e6,"Hz")
    if AUTO_SAVE_SPECTROSCOPY_CALIBRATION and np.isfinite(err[2]):
        save_results({"resonator_IF" if lo_source=="resonator" else "drive_IF":float(par[2]*1e6)})
    plt.show()


def resonator_sweep():
    fit_spectrum("resonator_sweep", "resonator")


def qubit_sweep():
    fit_spectrum("qubit_sweep", "qubit")


def bias_sweep():
    data=read_file("bias_sweep")
    avg,amp_lst,*_=data["setup"]
    n=len(amp_lst)
    I,Q=as_iq(data,target_shape=(int(avg),n))
    _,_,a,sy=mean_iq_amplitude_and_sem(I,Q)
    # Original QUA uses denominator n rather than n-1 for the step.
    start,end=amp_lst[0],amp_lst[-1]
    x=float(start)+np.arange(n)*(float(end)-float(start))/n
    show_data(x,a,sy,"bias sweep","bias flux [a.u.]","amplitude [a.u.]")
    index=int(np.argmin(a))
    print(f"Bias minimum sampled at {x[index]:.6g} a.u.; not a fitted optimum")
    save_report("bias_sweep", {"status":"no_physical_model_specified",
                             "minimum_sampled_bias_candidate":float(x[index]),
                             "uncertainty":"propagated I/Q SEM; no X resolution known",
                             "chi_square_red":None,"p_value":None})
    plt.show()


def resonator_bias_sweep():
    data=read_file("resonator_bias_sweep")
    avg,nfreq,span,center_if,lo=data["setup"]
    rawI,rawQ=as_iq(data)
    # QUA buffers frequency inner, then bias outer; accepts stray singleton axes.
    total=rawI.size
    if total%(int(avg)*int(nfreq))!=0:
        raise ValueError(f"Unexpected 2D scan size {rawI.shape}")
    namp=total//(int(avg)*int(nfreq))
    I,Q=as_iq(data,target_shape=(int(avg),int(namp),int(nfreq)))
    _,_,a,sy=mean_iq_amplitude_and_sem(I,Q)
    freq=(float(lo)+float(center_if)-float(span)/2+
          np.arange(int(nfreq))*float(span)/int(nfreq))/1e9
    amp=-2+np.arange(int(namp))*4/float(namp)
    plt.style.use("classic")
    for values,title,cb in ((a,"Resonator spectroscopy vs Z flux","Amplitude [a.u.]"),
                            (sy,"Resonator bias sweep - SEM","Amplitude SEM [a.u.]")):
        fig,ax=plt.subplots(figsize=(9,5),facecolor="w")
        image=ax.imshow(values,origin="lower",aspect="auto",cmap="viridis",
                        extent=(freq[0],freq[-1],amp[0],amp[-1]))
        fig.colorbar(image,ax=ax,label=cb)
        ax.set(xlabel="Resonator RF frequency [GHz]",ylabel="Z flux amplitude [a.u.]",title=title)
        ax.grid(False)
        fig.tight_layout()
    save_report("resonator_bias_sweep",{
        "status":"map_and_uncertainty_map; no fit model, chi-square_red and p-value not applicable",
        "averages":int(avg),"num_bias":int(namp),"num_freq":int(nfreq),
        "uncertainty":"SEM from repeated complex I/Q scans (including covariance)"})
    plt.show()


def readout_amp_sweep():
    data=read_file("readout_amp_sweep")
    amp_lst=data["setup"][2]
    avg=int(data["setup"][0])
    I,Q=as_iq(data,target_shape=(avg,len(amp_lst)))
    _,_,a,sy=mean_iq_amplitude_and_sem(I,Q)
    show_data(np.asarray(amp_lst,float),a,sy,"readout amplitude sweep",
              "readout amplitude [a.u.]","amplitude [a.u.]")
    print("No fit: no line-shape model specified for the sweep")
    plt.show()


def rabi_find():
    data=read_file("rabi_find")
    avg,legacy_count,durations,*_=data["setup"]
    # This QUA program plays duration=t where t is in cycles, not nanoseconds,
    # and does NOT sweep the old amp_lst.  Correct the plotted axis accordingly.
    time_ns=np.asarray(durations,float)*CLOCK_NS
    I,Q=as_iq(data,target_shape=(int(avg),len(time_ns)))
    y,sy=iq_projected_shots(I,Q)
    x=time_ns/1000.0
    if flat_signal(y,sy):
        show_data(x,y,sy,"rabi find - signal not resolved","Drive pulse duration [us]","IQ projection [a.u.]")
        print("Rabi signal is not resolved above noise. No fit or pi duration is reported.")
        save_report("rabi_find",{"status":"no_signal_resolved","statistical_error":"IQ projection SEM"})
        plt.show();return
    f_fft=np.fft.rfftfreq(len(x), d=float(np.mean(np.diff(x))))
    spectrum=np.abs(np.fft.rfft(y-y.mean()))
    f0=float(f_fft[1+np.argmax(spectrum[1:])]) if len(spectrum)>1 else 1.0
    amp=float((np.percentile(y,90)-np.percentile(y,10))/2)
    guesses=[max(f0,0.05),max(f0*0.8,0.05),max(f0*1.2,0.05)]
    candidates=[]
    for f in guesses:
        try:
            par,err,rep=robust_fit(
                rabi_model,x,y,sy,[float(y.mean()),amp,f,max(float(x.max()),0.1),0],
                ([-np.inf,-np.inf,0.01,0.01,-2*np.pi],
                 [np.inf,np.inf,1000.0,1e7,2*np.pi]),
                ["offset","amplitude","rabi_frequency_MHz","rabi_decay_us","phase_rad"])
            candidates.append((rep["chi_square"],par,err,rep))
        except (RuntimeError,ValueError):
            pass
    if not candidates:
        show_data(x,y,sy,"rabi find","Drive pulse duration [us]","IQ projection [a.u.]")
        print("Unable to fit Rabi; leaving calibration unchanged")
        plt.show();return
    _,par,err,rep=min(candidates,key=lambda z:z[0])
    show_fit(x,y,sy,rabi_model,par,rep,"rabi find","Drive pulse duration [us]","IQ projection [a.u.]")
    rep["note"]=("The QUA time scan uses DRAG duration stretching, not calibrated X180 gate amplitude. "
                 "A fitted period is not a validated X180 gate calibration. Check pulse shape and IQ contrast.")
    if par[3]>2*x.max():
        print("WARNING: decay time is > twice the scanned range; treat as weakly constrained")
        rep["decay_reliable"]=False
    else:
        rep["decay_reliable"]=True
    save_report("rabi_find",rep)
    save_results({"rabi_frequency_MHz_candidate":float(par[2]),
                  "rabi_frequency_MHz_stat_error":float(err[2]),
                  "rabi_decay_us_candidate":float(par[3]) if rep["decay_reliable"] else None})
    print_report("rabi_find",rep)
    print("Indicative half-period (NOT a calibrated DRAG X180):",1/(2*par[2]),"us")
    plt.show()


def discrimination():
    data=read_file("discrimination")
    I0,Q0=as_iq(data,keys=("I_raw_0","Q_raw_0"))
    I1,Q1=as_iq(data,keys=("I_raw_1","Q_raw_1"))
    X0=np.column_stack([I0.ravel(),Q0.ravel()])
    X1=np.column_stack([I1.ravel(),Q1.ravel()])
    if min(len(X0),len(X1))<10:
        raise ValueError("Need at least 10 shots per prepared state")
    rng=np.random.default_rng(RANDOM_SEED)
    X0=X0[rng.permutation(len(X0))]
    X1=X1[rng.permutation(len(X1))]
    n0,n1=len(X0)//2,len(X1)//2
    X0train,X0test=X0[:n0],X0[n0:]
    X1train,X1test=X1[:n1],X1[n1:]
    m0,m1=X0train.mean(axis=0),X1train.mean(axis=0)
    cov=(np.cov(X0train,rowvar=False)+np.cov(X1train,rowvar=False))/2
    reg=max(float(np.trace(cov))*1e-8,1e-16)
    w=np.linalg.solve(cov+reg*np.eye(2),m1-m0)
    b=-float((m0+m1)@w/2)
    p0=((X0test@w+b)>0)
    p1=((X1test@w+b)>0)
    correct0=int((~p0).sum()); correct1=int(p1.sum())
    f0,lo0,hi0=wilson_p(correct0,len(X0test))
    f1,lo1,hi1=wilson_p(correct1,len(X1test))
    f=float((f0+f1)/2)
    ferr=float(np.sqrt(((lo0+hi0)/2)**2 + ((lo1+hi1)/2)**2)/2)
    success=correct0+correct1
    p_chance=float(binomtest(success,len(X0test)+len(X1test),0.5,alternative="greater").pvalue)
    plt.style.use("classic")
    fig,axes=plt.subplots(1,2,figsize=(12,5),facecolor="w")
    ax=axes[0]
    ax.grid(True)
    ax.scatter(X0test[:,0],X0test[:,1],c="b",s=9,marker="o",alpha=0.45,label="Prepared |0>")
    ax.scatter(X1test[:,0],X1test[:,1],c="r",s=10,marker="x",alpha=0.55,label="Prepared |1>")
    xmin=min(X0test[:,0].min(),X1test[:,0].min())
    xmax=max(X0test[:,0].max(),X1test[:,0].max())
    xx=np.linspace(xmin,xmax,200)
    if abs(w[1])>1e-12:
        ax.plot(xx,-(w[0]*xx+b)/w[1],"k--",label="Separation line")
    elif abs(w[0])>1e-12:
        ax.axvline(-b/w[0],color="k",linestyle="--",label="Separation line")
    ax.scatter(*m0,marker="D",c="b",s=55)
    ax.scatter(*m1,marker="D",c="r",s=55)
    ax.set(xlabel="I [a.u.]",ylabel="Q [a.u.]",title=f"IQ plane: F={f:.4f} +/- {ferr:.4f}")
    ax.legend()
    ax2=axes[1]
    ax2.grid(True)
    ax2.hist(X0test@w+b,bins=40,density=True,alpha=0.5,color="b",label="Prepared |0>")
    ax2.hist(X1test@w+b,bins=40,density=True,alpha=0.5,color="r",label="Prepared |1>")
    ax2.axvline(0,color="k",linestyle="--")
    ax2.set(xlabel="IQ discrimination score",ylabel="Density",title="Classification overlap")
    ax2.legend()
    fig.tight_layout()
    report={"balanced_fidelity":f,"balanced_fidelity_sem_approx":ferr,
            "ground_fidelity":float(f0),"excited_fidelity":float(f1),
            "test_shots_0":len(X0test),"test_shots_1":len(X1test),
            "p_value_vs_random_guess":p_chance,
            "note":"Classification, not curve fitting. No reduced chi-square is defined.",
            "centroid_distance":float(np.linalg.norm(m1-m0))}
    save_report("discrimination",report)
    save_results({"discriminator_w":w.tolist(),"discriminator_b":b,
                  "discriminator_balanced_fidelity":f,
                  "discriminator_balanced_fidelity_error_1sigma":ferr})
    print(f"Discrimination fidelity: {f:.5f} +/- {ferr:.3g}")
    print(f"p-value vs random classifier: {p_chance:.3g}")
    print("No chi-square_red is defined for an IQ classification boundary.")
    if f<MIN_VALID_DISCRIMINATION_FIDELITY:
        print("WARNING: weak single-shot state contrast; do not trust T1/Ramsey/Echo gate fidelity")
    plt.show()


def _probability_fit_or_diagnostic(name,x,I,Q,model,p0,bounds,names,title,x_label):
    p,asy,sy,fid=p1_estimate(I,Q)
    if fid<MIN_VALID_DISCRIMINATION_FIDELITY or flat_signal(p,sy):
        reason=(f"readout assignment fidelity {fid:.3f} is too low" if fid<MIN_VALID_DISCRIMINATION_FIDELITY
                else "measured P1 is flat within binomial errors")
        print(f"{name}: NO PHYSICAL FIT: {reason}")
        plt.style.use("classic")
        fig,ax=plt.subplots(figsize=(9,5),facecolor="w")
        ax.grid(True)
        ax.plot(x,p,"bo",ms=3,label="Classified P(|1>) - not calibrated")
        idx=np.linspace(0,len(x)-1,min(len(x),MAX_ERROR_BARS),dtype=int)
        ax.errorbar(x[idx],p[idx],yerr=asy[:,idx],fmt="none",ecolor="b",capsize=2)
        ax.set(xlabel=x_label,ylabel="Classified |1> fraction",title=title+" - Diagnostic only")
        ax.legend();fig.tight_layout()
        save_report(name,{"status":"fit_skipped","reason":reason,
                          "chi_square_red":None,"p_value":None,
                          "uncertainty":"68.3% Wilson binomial interval"})
        return None
    try:
        par,err,rep=robust_fit(model,x,p,sy,p0,bounds,names)
    except (RuntimeError,ValueError) as exc:
        print(f"{name}: fit failed: {exc}")
        show_data(x,p,sy,title,x_label,"Classified P(|1>)")
        return None
    show_fit(x,p,sy,model,par,rep,title,x_label,"Excited-state probability",y_asym=asy)
    rep["measurement_uncertainty"]="68.3% Wilson binomial interval; symmetric mean half-width for fit"
    rep["calibration_caveat"]="Classifier preparation errors, drift and bias are not part of binomial errorbars."
    save_report(name,rep)
    print_report(name,rep)
    return par,err,rep


def consecutive_pulse():
    data=read_file("consecutive_pulse")
    avg,num_pulses,*_=data["setup"]
    I,Q=as_iq(data,target_shape=(int(avg),int(num_pulses)))
    x=np.arange(int(num_pulses))
    out=_probability_fit_or_diagnostic(
        "consecutive_pulse",x,I,Q,parity_model,
        [0.5,0.9,0.97],([0,-1,0],[1,1,1]),
        ["baseline","contrast","lambda"],
        "consecutive pi pulses","Number of consecutive calibrated pi pulses")
    if out:
        par,err,rep=out
        print("Model-dependent pi average gate fidelity =",(1+par[2])/2,
              "+/-",err[2]/2)
        print("This assumes isotropic depolarization, not established by this experiment.")
        if abs(par[1]) > 3 * np.median(np.asarray(rep["y_sem_used"])) and err[2] < 0.25:
            save_results({"pi_gate_fidelity_model_candidate":float((1+par[2])/2),
                          "pi_gate_fidelity_model_stat_error":float(err[2]/2)})
    plt.show()


def t1():
    data=read_file("t1")
    setup=data["setup"]
    x=np.asarray(setup["tau_ns"],float)/1000.0
    avg=int(setup["averages"])
    I,Q=as_iq(data,target_shape=(avg,len(x)))
    w,b,fid=get_discriminator()
    p,asy,sy,_=p1_estimate(I,Q)
    p0=[float(p[-1]),float(p[0]-p[-1]),max(float(x.max()/3),0.5)]
    out=_probability_fit_or_diagnostic("t1",x,I,Q,exp_decay,p0,
                  ([0,-1,0.01],[1,1,1e7]),["offset","contrast","T1_us"],
                  "T1 relaxation","Wait time [us]")
    if out:
        par,err,_=out
        if (abs(par[1])<3*np.median(sy) or
            not np.isfinite(err[2]) or err[2]>=par[2] or par[2]>3*x.max()):
            print("T1 amplitude or decay time is poorly constrained; no T1 result saved")
        else:
            save_results({"T1_us":float(par[2]),"T1_us_stat_error":float(err[2])})
    plt.show()


def ramsey():
    data=read_file("ramsey")
    setup=data["setup"]
    x=np.asarray(setup["tau_ns"],float)/1000.0
    avg=int(setup["averages"])
    I,Q=as_iq(data,target_shape=(avg,len(x)))
    p,asy,sy,fid=p1_estimate(I,Q)
    f_mhz=abs(float(setup["detuning_hz"]))/1e6
    out=_probability_fit_or_diagnostic(
        "ramsey",x,I,Q,ramsey_model,
        [float(np.mean(p)),max(float(np.ptp(p))/2,1e-3),max(float(x.max()/2),0.3),f_mhz,0.0],
        ([0,-1,0.01,max(f_mhz*0.5,0.001),-2*np.pi],
         [1,1,1e7,max(f_mhz*1.5,0.01),2*np.pi]),
        ["offset","amplitude","T2star_us","fringe_frequency_MHz","phase_rad"],
        "Ramsey experiment (T2*)","Free evolution time [us]")
    if out:
        par,err,_=out
        if (abs(par[1])>3*np.median(sy) and
            np.isfinite(err[2]) and err[2]<par[2] and par[2]<=3*x.max()):
            save_results({"T2star_us":float(par[2]),"T2star_us_stat_error":float(err[2]),
                          "ramsey_fringe_frequency_MHz":float(par[3])})
        else:
            print("Ramsey coherence time is poorly constrained; result not saved")
    plt.show()


def echo():
    data=read_file("echo")
    setup=data["setup"]
    x=np.asarray(setup["tau_ns"],float)/1000.0
    avg=int(setup["averages"])
    I,Q=as_iq(data,target_shape=(avg,len(x)))
    p,asy,sy,fid=p1_estimate(I,Q)
    out=_probability_fit_or_diagnostic(
        "echo",x,I,Q,exp_decay,
        [float(p[-1]),float(p[0]-p[-1]),max(float(x.max()/2),0.2)],
        ([0,-1,0.01],[1,1,1e7]),["offset","contrast","T2echo_us"],
        "Hahn echo (T2)","Total free evolution time [us]")
    if out:
        par,err,_=out
        if (abs(par[1])>3*np.median(sy) and
            np.isfinite(err[2]) and err[2]<par[2] and par[2]<=3*x.max()):
            save_results({"T2echo_us":float(par[2]),"T2echo_us_stat_error":float(err[2])})
        else:
            print("Echo coherence time is poorly constrained; result not saved")
    plt.show()


def compare():
    """Compare calibrated coherence times; optional illustrative Shor estimate."""
    r=saved_results()
    print("\n========= Timescales (accepted fits only) =========")
    names={"T1_us":"T1", "T2star_us":"T2* (Ramsey)", "T2echo_us":"T2 echo"}
    vals=[]
    for key,label in names.items():
        if key in r and r[key] is not None:
            value=float(r[key]);error=float(r.get(key+"_stat_error",np.nan))
            print(f"{label}: {value:.4g} +/- {error:.3g} us")
            vals.append((label,value,error))
        else:
            print(f"{label}: not yet calibrated")
    if len(vals):
        fig,ax=plt.subplots(figsize=(8,4),facecolor="w")
        ax.grid(True)
        positions=np.arange(len(vals))
        ax.errorbar(positions,[v[1] for v in vals],
                    yerr=[v[2] for v in vals],fmt="bo",capsize=3)
        ax.set_xticks(positions,[v[0] for v in vals])
        ax.set(ylabel="Coherence time [us]",title="Measured coherence times")
        fig.tight_layout()
        plt.show()
    # Example only; this is not a Qolab hardware runtime estimate.
    depth=1.8e12
    illustrative_logical_layer_ns=1000.0
    seconds=depth*illustrative_logical_layer_ns*1e-9
    print("\nIllustrative RSA-2048 Shor runtime, NOT a hardware prediction:")
    print(f"depth={depth:.2e} logical layers, assumed layer time=1 us")
    print(f"runtime={seconds:.3e} seconds = {seconds/86400:.1f} days")
    print("Surface-code feasibility cannot be inferred from single-qubit data alone;")
    print("two-qubit, measurement, leakage, correlated errors and decoding overhead matter.")


# ----------------------------------------------------------------------
# Bonus: non-selective driven quantum Zeno experiment.
# ----------------------------------------------------------------------
def zeno_ideal_probability(n):
    """N equal pi/N rotations, N-1 ideal non-selective projective measurements."""
    n = np.asarray(n, dtype=float)
    return (1.0 - np.cos(np.pi/n)**n)/2.0


def zeno():
    data = read_file("zeno")
    setup = data["setup"]
    N = np.asarray(setup["segments"], dtype=int)
    navg, nref = int(setup["averages"]), int(setup["reference_averages"])
    if N.ndim != 1 or len(N) < 3 or N[0] != 1 or np.any(np.diff(N) <= 0) or navg < 2 or nref < 2:
        raise ValueError("Invalid Zeno setup; check the acquisition metadata.")
    def load(key, shape):
        I,Q = as_iq(data, keys=("I_"+key, "Q_"+key), target_shape=shape)
        if not (np.isfinite(I).all() and np.isfinite(Q).all()):
            raise ValueError("Non-finite raw IQ data in " + key)
        return np.stack((I,Q),axis=-1)
    gref = load("ref_ground", (nref,))
    eref = load("ref_excited", (nref,))
    g0, e0 = gref.mean(axis=0), eref.mean(axis=0)
    delta = e0-g0
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-15:
        raise ValueError("Reference centroids coincide; cannot define an IQ projection axis.")
    axis = delta/norm
    # Do not divide by the weak reference contrast. Keep the units in a.u.
    # This independent reference fixes the axis; no classifier or postselection.
    projected = {k: (load(k,(navg,len(N)))-g0)@axis for k in
                 ("drive_meas","drive_sham","ground_meas","ground_sham")}
    def summarize(shots):
        return shots.mean(axis=0), ensure_errors(shots.std(axis=0,ddof=1)/np.sqrt(len(shots)))
    dm, ds = projected["drive_meas"], projected["drive_sham"]
    gm, gs = projected["ground_meas"], projected["ground_sham"]
    # Paired differences include the within-repeat covariances automatically.
    measured_shots = dm-gm
    sham_shots = ds-gs
    suppression_shots = sham_shots-measured_shots
    y_m, se_m = summarize(measured_shots)
    y_s, se_s = summarize(sham_shots)
    y_d, se_d = summarize(suppression_shots)
    y_gm, se_gm = summarize(gm)
    y_gs, se_gs = summarize(gs)
    y_gdiff, se_gdiff = summarize(gm-gs)
    # Independent N=1 check for the reference direction and detectable drive.
    first_shots = (measured_shots[:,0]+sham_shots[:,0])/2
    first_mean, first_sem = summarize(first_shots)
    first_z = float(first_mean/first_sem)
    ref_diff_shots = (eref-gref)@axis
    _, ref_sem = summarize(ref_diff_shots)
    # Since the reference direction is optimized on these same shots, its SEM
    # is conditional only; the independent N=1 check above is more important.
    ideal = zeno_ideal_probability(N)
    # Fit the paired contrast difference to B*[1-P_Z(N)]. B can be negative;
    # we never force a positive Zeno signal. This is a shape-only diagnostic,
    # NOT a complete model of partial measurement, T1, T2 or pulse errors.
    shape = 1.0-ideal
    weights = 1.0/se_d**2
    den = float(np.sum(weights*shape**2))
    B = float(np.sum(weights*shape*y_d)/den)
    Berr = float(1.0/np.sqrt(den))
    fit = B*shape
    residual = y_d-fit
    chi = float(np.sum((residual/se_d)**2))
    dof = len(N)-1
    # Null benchmark D=0, without fitting a parameter.
    null_chi = float(np.sum((y_d/se_d)**2))
    null_p = float(chi2_distribution.sf(null_chi,len(N)))
    pval = float(chi2_distribution.sf(chi,dof))
    # Direct endpoint test avoids relying only on an ideal measurement model.
    # Nominal Gaussian z values; not corrected for multiple comparisons.
    last_z = float(y_d[-1]/se_d[-1])
    # A large no-drive measurement response suggests measurement-induced
    # population changes or instrumental offsets beyond this simple model.
    ground_chi = float(np.sum((y_gdiff/se_gdiff)**2))
    ground_p = float(chi2_distribution.sf(ground_chi,len(N)))
    flags = []
    if first_z < 3:
        flags.append("Independent N=1 drive/reference contrast is not resolved at 3 SEM.")
    if B <= 0 or B/Berr < 3:
        flags.append("Positive suppression amplitude is not resolved at 3 SEM.")
    if pval < 0.05:
        flags.append("Ideal suppression-shape fit has p < 0.05.")
    if ground_p < 0.05:
        flags.append("No-drive controls differ: possible measurement-induced offsets/heating.")
    if abs(y_d[0]) > 3*se_d[0]:
        flags.append("N=1 arms disagree though they have no intermediate measurements.")
    if np.any(y_s[1:] < 3*se_s[1:]):
        flags.append("Timing-matched sham loses resolved drive contrast at some settings.")
    # Even if flags are absent, measurement strength, QND behavior, drift and
    # pulse calibration require independent verification before a Zeno claim.
    status = "inconclusive_diagnostics" if flags else "positive_suppression_candidate"
    report = {
        "status": status, "segments": N.tolist(), "intermediate_measurements": (N-1).tolist(),
        "setup": setup, "unit": "a.u. (IQ projection)",
        "reference_centroids": {"ground":g0, "excited":e0}, "projection_axis":axis,
        "reference_contrast_au":norm, "reference_conditional_sem_au":float(ref_sem),
        "independent_N1_contrast_au":float(first_mean),"independent_N1_sem_au":float(first_sem),
        "independent_N1_z":first_z,
        "measured_drive_minus_ground_au":y_m, "measured_sem_au":se_m,
        "sham_drive_minus_ground_au":y_s, "sham_sem_au":se_s,
        "ground_meas_au":y_gm, "ground_meas_sem_au":se_gm,
        "ground_sham_au":y_gs, "ground_sham_sem_au":se_gs,
        "ground_difference_au":y_gdiff, "ground_difference_sem_au":se_gdiff,
        "ground_difference_chi_square":ground_chi, "ground_difference_p_value":ground_p,
        "suppression_au":y_d, "suppression_sem_au":se_d,
        "ideal_excited_probability":ideal,
        "parameters":{"suppression_scale_au":{"value": B,"error_1sigma": Berr}},
        "chi_square":chi, "dof":dof,"reduced_chi_square":chi/dof,"p_value":pval,
        "residuals":residual, "null_chi_square":null_chi,"null_p_value":null_p,
        "suppression_scale_z":B/Berr,"last_setting_suppression_z":last_z,
        "fit_model":"D(N)=B*[1-(1-cos(pi/N)^N)/2]",
        "uncertainty":"Paired per-repeat SEM, conditioned on independent IQ axis; diagonal WLS; correlations across settings and systematic uncertainty not modeled.",
        "flags":flags,
        "caveat":"Final IQ contrast, not calibrated excited population; no single-shot fidelity assumed. Suppression is not proof of QZE: measurement-induced relaxation, ordinary dephasing, leakage and calibration drift can also affect the signal."
    }
    save_report("zeno",report)
    print_report("zeno",report)
    print("Independent N=1 driven contrast:", float(first_mean), "+/-",float(first_sem),"a.u.")
    print("Null (no suppression) p-value:",null_p)
    print("Positive scale z:",B/Berr,"; last setting z:",last_z)
    print("No-drive control difference p-value:",ground_p)
    print("Status:",status)
    for flag in flags: print("WARNING:",flag)
    print("No coherence time or gate fidelity is inferred from this experiment.")
    print("N | intermediate readouts | measured contrast +/- SEM | sham contrast +/- SEM | suppression +/- SEM")
    for i,n in enumerate(N):
        print(f"{n:2d} | {n-1:2d} | {y_m[i]:.6g} +/- {se_m[i]:.3g} | {y_s[i]:.6g} +/- {se_s[i]:.3g} | {y_d[i]:.6g} +/- {se_d[i]:.3g}")

    folder = DATA_DIR/"zeno_figures"
    folder.mkdir(parents=True,exist_ok=True)
    xlabel = "Number of intermediate readouts (N - 1)"
    x = N-1
    def make_plot(title,ylabel):
        fig,ax = plt.subplots(figsize=(9,5),facecolor="w")
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        ax.set_xticks(x)
        ax.grid(True)
        return fig,ax
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+".png"),dpi=300,bbox_inches="tight")
    fig,ax = make_plot("Quantum Zeno: driven IQ contrast", "Drive minus no-drive IQ [a.u.]")
    ax.errorbar(x,y_s,yerr=se_s,fmt="bo-",capsize=3,label="Timing-matched sham")
    ax.errorbar(x,y_m,yerr=se_m,fmt="rs-",capsize=3,label="Intermediate readouts")
    ax.axhline(0,color="k",linewidth=0.8,linestyle="--")
    ax.legend(loc="best",fontsize=10)
    finish(fig,"zeno")
    fig,ax = make_plot("Quantum Zeno: differential suppression fit", "Sham minus measured contrast [a.u.]")
    ax.errorbar(x,y_d,yerr=se_d,fmt="bo",capsize=3,label="Paired measurement")
    ax.plot(x,fit,"r-",linewidth=1.5,label="Ideal-shape fit at sampled settings")
    ax.axhline(0,color="k",linestyle="--",linewidth=0.8)
    # Fit statistics are printed/saved, not written over the graph.
    finish(fig,"zeno_fit")
    fig,ax = make_plot("Quantum Zeno: fit residuals", "Measured - fit [a.u.]")
    ax.errorbar(x,residual,yerr=se_d,fmt="bo",capsize=3)
    ax.axhline(0,color="r",linestyle="--")
    finish(fig,"zeno_residuals")
    fig,ax = make_plot("No-drive control: effect of readouts", "Measured minus sham IQ [a.u.]")
    ax.errorbar(x,y_gdiff,yerr=se_gdiff,fmt="bo",capsize=3)
    ax.axhline(0,color="r",linestyle="--")
    finish(fig,"zeno_controls")
    print("Saved plots in",folder)
    plt.show()


# ----------------------------------------------------------------------
# Bonus variant: excited-state survival, existing binary discriminator.
# ----------------------------------------------------------------------
def zeno_survival():
    data = read_file('zeno_survival')
    s = data['setup']
    k = np.asarray(s['steps'], dtype=int)
    M, n = int(s['pi_segments']), int(s['averages'])
    if n < 2 or M < 2 or len(k) < 3 or k[0] != 0 or k[-1] != M or np.any(np.diff(k) <= 0):
        raise ValueError('Invalid survival setup.')
    w = np.asarray(s['discriminator_w'], float)
    b, fid = float(s['discriminator_b']), float(s['discriminator_balanced_fidelity'])
    if w.shape != (2,) or not np.isfinite(w).all() or not np.linalg.norm(w) > 0 or not np.isfinite(b) or not 0 <= fid <= 1:
        raise ValueError('Invalid acquisition-time classifier.')
    def classify(key, shape):
        I,Q = as_iq(data, ('I_'+key,'Q_'+key), target_shape=shape)
        if not (np.isfinite(I).all() and np.isfinite(Q).all()):
            raise ValueError('Non-finite IQ in '+key)
        return (w[0]*I+w[1]*Q+b > 0).astype(int)
    def fraction(shots):
        p,lo,hi = wilson_p(shots.sum(axis=0), len(shots))
        return p, np.stack((lo,hi)), (lo+hi)/2
    arms = {key: classify(key,(n,len(k))) for key in s['arms']}
    if not {'meas','sham'}.issubset(arms):
        raise ValueError('Missing driven arms.')
    stats = {key: fraction(a) for key,a in arms.items()}
    g = classify('ref_ground',(n,))
    pg,eg,sg = fraction(g)
    # Independent final-readout check: saved classifier, no refitting.
    e = np.concatenate((arms['meas'][:,0],arms['sham'][:,0]))
    pe,ee,se = fraction(e)
    # Use per-repeat mean for reference contrast, accounting for pairing.
    refdiff = (arms['meas'][:,0]+arms['sham'][:,0])/2-g
    refcontrast = float(refdiff.mean())
    refsem = float(refdiff.std(ddof=1)/np.sqrt(n))
    if refsem <= 0:
        refsem = float(np.hypot(sg,se))
    current_f = float((1+pe-pg)/2)
    # Wilson for individual fractions; paired SEM for differences of binary shots.
    def paired_difference(a,b):
        d = arms[a]-arms[b]
        y = d.mean(axis=0)
        se = d.std(axis=0,ddof=1)/np.sqrt(n)
        fallback = np.hypot(stats[a][2], stats[b][2])
        return y, np.where(se>0,se,fallback)
    delta, sd = paired_difference('meas','sham')
    # Ideal non-selective projective measurements between equal rotations.
    # Start in |1>; no intermediate readout at k=0 or k=1.
    ideal_sham = (1+np.cos(k*np.pi/M))/2
    ideal_meas = (1+np.cos(np.pi/M)**k)/2
    shape = ideal_meas-ideal_sham
    shape[k<=1] = 0.0
    wt = 1/sd**2
    den = np.sum(wt*shape**2)
    if den <= 0:
        raise ValueError('No nonzero ideal difference to fit.')
    A = float(np.sum(wt*shape*delta)/den)
    Ae = float(1/np.sqrt(den))
    fit = A*shape
    resid = delta-fit
    chi = float(np.sum((resid/sd)**2))
    dof = len(k)-1
    null_chi = float(np.sum((delta/sd)**2))
    null_p = float(chi2_distribution.sf(null_chi,len(k)))
    flags = []
    if fid < 0.65:
        flags.append('Saved assignment fidelity below 0.65; exploratory fit only.')
    if refcontrast < 3*refsem:
        flags.append('Current nominal |0>/|1> contrast not resolved at 3 SEM.')
    if A < 3*Ae:
        flags.append('Positive survival enhancement not resolved at 3 SEM.')
    if null_p >= 0.05:
        flags.append('Zero difference cannot be rejected by the omnibus null test.')
    if abs(delta[0]) > 3*sd[0] or np.any(np.abs(delta[k==1]) > 3*sd[k==1]):
        flags.append('Arms differ at a setting with no intermediate readout.')
    sham_drop = arms['sham'][:,0]-arms['sham'][:,-1]
    drop = float(sham_drop.mean())
    dropsem = float(sham_drop.std(ddof=1)/np.sqrt(n))
    if dropsem <= 0:
        dropsem = float(np.hypot(stats['sham'][2][0],stats['sham'][2][-1]))
    if drop < 3*dropsem:
        flags.append('No resolved final population decrease in the driven sham.')
    control = None
    if {'idle_meas','idle_sham'}.issubset(arms):
        d0,s0 = paired_difference('idle_meas','idle_sham')
        control_p = float(chi2_distribution.sf(np.sum((d0/s0)**2),len(k)))
        control = {'difference':d0,'sem':s0,'null_p_value':control_p}
        if control_p < 0.05:
            flags.append('No-drive excited-state controls differ; measurement may change relaxation/readout.')
    else:
        flags.append('No no-drive controls in this run.')
    pval = float(chi2_distribution.sf(chi,dof))
    if pval < 0.05:
        flags.append('Ideal difference shape rejected at p<0.05.')
    report = {'status':'inconclusive_diagnostics' if flags else 'positive_survival_candidate',
              'setup':s,'parameters':{'survival_scale':{'value':A,'error_1sigma':Ae}},
              'chi_square':chi,'dof':dof,'reduced_chi_square':chi/dof,'p_value':pval,
              'null_chi_square':null_chi,'null_p_value':null_p,'scale_z':A/Ae,
              'saved_fidelity':fid,'current_nominal_assignment_fidelity':current_f,
              'ground_reference_fraction':float(pg),'excited_reference_fraction':float(pe),
              'reference_contrast':refcontrast,'reference_contrast_sem':refsem,
              'sham_drop':drop,'sham_drop_sem':dropsem,
              'fractions':{a:{'p':v[0],'wilson_errors':v[1]} for a,v in stats.items()},
              'difference':delta,'difference_sem':sd,'residuals':resid,
              'ideal_sham_population':ideal_sham,'ideal_measured_population':ideal_meas,
              'control':control,'flags':flags,
              'fit_model':'D(k)=A*(cos(pi/M)^k-cos(k*pi/M))/2',
              'uncertainty':'68.3% Wilson per fraction; paired SEM for differences (Wilson quadrature fallback at zero sample variance). Diagonal WLS, ignores correlations across settings and classifier drift.',
              'caveat':'Uncorrected classified fractions, NOT calibrated state populations. Shape-only diagnostic; not proof of QZE. No T1 or gate fidelity inferred.'}
    save_report('zeno_survival',report)
    print_report('zeno_survival',report)
    print('Saved classifier fidelity:',fid,'; current nominal reference fidelity:',current_f)
    print('Reference contrast:',refcontrast,'+/-',refsem)
    print('Null (no difference) p-value:',null_p,'; scale z:',A/Ae)
    if control is not None:
        print('No-drive control null p-value:',control['null_p_value'])
    print('Status:',report['status'])
    for flag in flags:
        print('WARNING:',flag)
    print('k | intermediate readouts | sham P1 +/- err | measurement P1 +/- err | difference +/- SEM')
    for i,step in enumerate(k):
        print(f'{step:2d} | {max(step-1,0):2d} | {stats["sham"][0][i]:.5f} +/- {stats["sham"][2][i]:.4f} | {stats["meas"][0][i]:.5f} +/- {stats["meas"][2][i]:.4f} | {delta[i]:.5f} +/- {sd[i]:.4f}')
    folder = DATA_DIR/'zeno_survival_figures'
    folder.mkdir(parents=True,exist_ok=True)
    def plot(title,ylabel):
        fig,ax = plt.subplots(figsize=(9,5),facecolor='w')
        ax.set(title=title,xlabel='Number of coherent drive steps k',ylabel=ylabel)
        ax.set_xticks(k)
        ax.grid(True)
        return fig,ax
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    fig,ax = plot('Quantum Zeno: excited-state survival','Classified |1> fraction')
    for key,fmt,label in [('sham','bo-','Timing-matched sham'),('meas','rs-','Intermediate readouts')]:
        ax.errorbar(k,stats[key][0],yerr=stats[key][1],fmt=fmt,capsize=3,label=label)
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    finish(fig,'zeno_survival')
    # Separate time-axis graph; duration is nominal, not hardware timestamp.
    elapsed = np.asarray(s['nominal_elapsed_ns'],float)/1000
    fig,ax = plot('Quantum Zeno: excited-state survival vs time','Classified |1> fraction')
    for key,fmt,label in [('sham','bo-','Timing-matched sham'),('meas','rs-','Intermediate readouts')]:
        ax.errorbar(elapsed,stats[key][0],yerr=stats[key][1],fmt=fmt,capsize=3,label=label)
    ax.set_xlabel('Nominal time after preparation [us]')
    ax.set_xticks(elapsed)
    ax.set_xticklabels([f'{t:.2f}' for t in elapsed],rotation=30)
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    finish(fig,'zeno_survival_time')
    fig,ax = plot('Quantum Zeno: survival enhancement fit','Measured minus sham fraction')
    ax.errorbar(k,delta,yerr=sd,fmt='bo',capsize=3)
    ax.plot(k,fit,'r-')
    ax.axhline(0,color='k',linestyle='--',linewidth=1)
    finish(fig,'zeno_survival_fit')
    fig,ax = plot('Quantum Zeno: fit residuals','Difference minus fit')
    ax.errorbar(k,resid,yerr=sd,fmt='bo',capsize=3)
    ax.axhline(0,color='r',linestyle='--')
    finish(fig,'zeno_survival_residuals')
    if control is not None:
        fig,ax = plot('No further drive: excited-state control','Classified |1> fraction')
        for key,fmt,label in [('idle_sham','bo-','Timing-matched sham'),('idle_meas','rs-','Intermediate readouts')]:
            ax.errorbar(k,stats[key][0],yerr=stats[key][1],fmt=fmt,capsize=3,label=label)
        ax.set_ylim(0,1)
        ax.legend(loc='best')
        finish(fig,'zeno_survival_controls')
        fig,ax = plot('No further drive: control difference','Measured minus sham fraction')
        ax.errorbar(k,control['difference'],yerr=control['sem'],fmt='bo',capsize=3)
        ax.axhline(0,color='r',linestyle='--')
        finish(fig,'zeno_survival_control_difference')
    print('Saved plots in',folder)
    plt.show()


# ----------------------------------------------------------------------
# Bonus: simple Zeno experiment, binary classification of final IQ only.
# ----------------------------------------------------------------------
def zeno_simple():
    data = read_file('zeno_simple')
    s = data['setup']
    N = np.asarray(s['segments'], dtype=int)
    M = int(s['averages'])
    if M < 2 or len(N) < 3 or N[0] != 1 or np.any(np.diff(N) <= 0):
        raise ValueError('Invalid Zeno setup.')
    w = np.asarray(s['discriminator_w'], dtype=float)
    b = float(s['discriminator_b'])
    fid = float(s['discriminator_balanced_fidelity'])
    if w.shape != (2,) or not np.isfinite(w).all() or np.linalg.norm(w) == 0 or not np.isfinite(b) or not 0 <= fid <= 1:
        raise ValueError('Invalid saved classifier.')

    def classify(key):
        I, Q = as_iq(data, ('I_'+key, 'Q_'+key), target_shape=(M, len(N)))
        if not (np.isfinite(I).all() and np.isfinite(Q).all()):
            raise ValueError('Non-finite IQ in '+key)
        return (w[0]*I+w[1]*Q+b > 0).astype(int)

    sham, meas = classify('sham'), classify('meas')
    # Wilson 68.3% intervals, as in the other classified-population experiments.
    ps, ls, hs = wilson_p(sham.sum(axis=0), M)
    pm, lm, hm = wilson_p(meas.sum(axis=0), M)
    paired = sham-meas
    D = paired.mean(axis=0)
    sd = paired.std(axis=0, ddof=1)/np.sqrt(M)
    fallback = np.hypot((ls+hs)/2, (lm+hm)/2)
    sd = np.where(sd>0, sd, fallback)

    # Ideal projective measurements, no natural decoherence or relaxation.
    ideal_meas = (1-np.cos(np.pi/N)**N)/2
    shape = 1-ideal_meas
    shape[N==1] = 0
    # One-parameter weighted fit of the difference, NOT a calibrated P1 fit.
    # The scale is unconstrained so a negative result is not forced positive.
    weights = 1/sd**2
    den = float(np.sum(weights*shape**2))
    S = float(np.sum(weights*shape*D)/den)
    S_error = float(1/np.sqrt(den))
    fitted = S*shape
    residuals = D-fitted
    chi = float(np.sum((residuals/sd)**2))
    dof = len(N)-1
    p_fit = float(chi2_distribution.sf(chi,dof))
    null_chi = float(np.sum((D/sd)**2))
    p_null = float(chi2_distribution.sf(null_chi,len(N)))
    flags = ['Two-arm experiment: no fresh no-drive controls; readout-induced effects cannot be excluded.']
    if fid < 0.65:
        flags.append('Low saved assignment fidelity; exploratory classified fractions only.')
    if S < 3*S_error:
        flags.append('Positive suppression not resolved at 3 SEM.')
    if p_null >= 0.05:
        flags.append('Zero difference not rejected by the omnibus null test.')
    if p_fit < 0.05:
        flags.append('Ideal suppression shape rejected at p<0.05.')
    if abs(D[0]) > 3*sd[0]:
        flags.append('Arms differ at N=1 (no intermediate readouts in either).')
    report = {
        'status': 'exploratory_two_arm', 'setup': s,
        'parameters': {'suppression_scale': {'value':S,'error_1sigma':S_error}},
        'chi_square':chi, 'dof':dof, 'reduced_chi_square':chi/dof, 'p_value':p_fit,
        'null_chi_square':null_chi, 'null_p_value':p_null,
        'scale_z':S/S_error, 'saved_fidelity':fid, 'segments':N,
        'sham_p1':ps, 'sham_wilson_errors':[ls,hs],
        'meas_p1':pm, 'meas_wilson_errors':[lm,hm],
        'difference':D, 'difference_sem':sd, 'fitted_difference':fitted,
        'residuals':residuals, 'ideal_meas_p1':ideal_meas,
        'fit_model':'D(N)=S*(1+cos(pi/N)**N)/2', 'flags':flags,
        'uncertainty':'68.3% Wilson intervals for assigned fractions; paired SEM for differences, Wilson quadrature fallback if zero variance; diagonal WLS, no systematic or classifier uncertainty.',
        'caveat':'Uncorrected classified fractions. No retraining or correction. N changes elapsed time. No new controls. Fit is an ideal-shape diagnostic, not proof of QZE.'
    }
    save_report('zeno_simple', report)
    print_report('zeno_simple', report)
    print('Saved discrimination fidelity:', fid)
    print('Null (no suppression) p-value:', p_null, '| scale z:', S/S_error)
    print('N | intermediate readouts | sham assigned P1 +/- err | measured assigned P1 +/- err | suppression +/- SEM')
    for i,n in enumerate(N):
        print(f'{n:2d} | {n-1:2d} | {ps[i]:.5f} +/- {(ls[i]+hs[i])/2:.4f} | {pm[i]:.5f} +/- {(lm[i]+hm[i])/2:.4f} | {D[i]:.5f} +/- {sd[i]:.4f}')
    for flag in flags:
        print('WARNING:',flag)
    folder = DATA_DIR/'zeno_simple_figures'
    folder.mkdir(parents=True,exist_ok=True)
    def plot(title,ylabel):
        fig,ax = plt.subplots(figsize=(9,5),facecolor='w')
        ax.set(title=title,xlabel='Number of segments N',ylabel=ylabel)
        ax.set_xticks(N)
        ax.grid(True)
        return fig,ax
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    fig,ax = plot('Quantum Zeno: classified excited-state fraction', 'Assigned P1')
    ax.errorbar(N,ps,yerr=[ls,hs],fmt='bo-',capsize=3,label='Without intermediate readouts (sham)')
    ax.errorbar(N,pm,yerr=[lm,hm],fmt='rs-',capsize=3,label='With intermediate readouts')
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    finish(fig,'zeno_simple')
    fig,ax = plot('Quantum Zeno: suppression fit','Sham minus measured assigned P1')
    ax.errorbar(N,D,yerr=sd,fmt='bo',capsize=3,label='Paired differences')
    ax.plot(N,fitted,'r-',label='Ideal-shape fit at sampled N')
    ax.axhline(0,color='k',linestyle='--',linewidth=1)
    ax.legend(loc='best')
    finish(fig,'zeno_simple_fit')
    fig,ax = plot('Quantum Zeno: fit residuals','Difference minus fit')
    ax.errorbar(N,residuals,yerr=sd,fmt='bo',capsize=3)
    ax.axhline(0,color='r',linestyle='--')
    finish(fig,'zeno_simple_residuals')
    print('Saved plots in',folder)
    plt.show()


FUNCTIONS={
    "resonator_sweep":resonator_sweep,
    "resonator_bias_sweep":resonator_bias_sweep,
    "bias_sweep":bias_sweep,
    "qubit_sweep":qubit_sweep,
    "rabi_find":rabi_find,
    "readout_amp_sweep":readout_amp_sweep,
    "discrimination":discrimination,
    "consecutive_pulse":consecutive_pulse,
    "t1":t1,
    "ramsey":ramsey,
    "echo":echo,
    "compare":compare,
    "zeno":zeno,
    "zeno_survival":zeno_survival,
    "zeno_simple":zeno_simple,
}


def main():
    if ANALYSIS not in FUNCTIONS:
        raise ValueError(f"Unknown ANALYSIS {ANALYSIS!r}")
    FUNCTIONS[ANALYSIS]()


if __name__=="__main__":
    main()
