import numpy as np
import xarray as xr
import scipy
import pandas as pd
import time
import physio
from tqdm import tqdm
from tools import *
from configuration import *
# from overview_data_pycns import get_patient_list
from cpu_usage import start_measurement

# Import for P2P1
plugin_dir = base_folder / 'package_P2_P1' 
if str(plugin_dir) not in sys.path:
    sys.path.append(str(plugin_dir))
from p2p1.subpeaks import SubPeakDetector


# Import for PSI
plugin_dir = base_folder / 'ICMPWaveformClassificationPlugin' / 'plugin' / 'pulse_detection'
if str(plugin_dir) not in sys.path:
    sys.path.append(str(plugin_dir))
from classifier_pipeline import ProcessingPipeline
from pulse_detector import Segmenter
from pulse_classifier import Classifier


def load_raw_icp_resp(sub, duration_hours):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)

    icp_stream = cns_reader.streams['ICP']
    srate_icp = icp_stream.sample_rate
    i0 = 0
    i1 = int(srate_icp * 3600 * duration_hours)
    raw_icp, times_icp = icp_stream.get_data(isel = slice(i0, i1), apply_gain=True, with_times = True, time_as_second = True)

    co2_stream = cns_reader.streams['CO2']
    srate_co2 = co2_stream.sample_rate
    i0 = 0
    i1 = int(srate_co2 * 3600 * duration_hours)
    raw_co2, times_co2 = co2_stream.get_data(isel = slice(i0, i1), apply_gain=True, with_times = True, time_as_second = True)

    return raw_icp, srate_icp, times_icp, raw_co2, srate_co2, times_co2

def icp_heart_time_domain(raw_icp, srate):
    p =  {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter',
    'peak_prominence' : 0.5,
    'h_distance_s' : 0.3,
    'rise_amplitude_limits' : (0,20),
    'amplitude_at_trough_low_limit' : -10,
    }
    icp_features = compute_icp(raw_icp, srate, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'], peak_prominence = p['peak_prominence'], h_distance_s = p['h_distance_s'], rise_amplitude_limits=p['rise_amplitude_limits'], amplitude_at_trough_low_limit = p['amplitude_at_trough_low_limit'])
    return icp_features['rise_amplitude'].values

def icp_resp_time_domain(raw_icp, srate_icp, times_icp, raw_co2, srate_co2, times_co2):

    icp_filtered = iirfilt(raw_icp, srate_icp, highcut = 0.5)

    _, resp_cycles = physio.compute_respiration(raw_co2, srate_co2, parameter_preset = 'human_co2')
    for point_name in ['inspi','expi','next_inspi']:
        resp_cycles[f'{point_name}_time'] = times_co2[resp_cycles[f'{point_name}_index']]
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_icp[0]) & (resp_cycles['next_inspi_time'] < times_icp[-1])]
    med_cycle_ratio = resp_cycles['cycle_ratio'].median()

    cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values
    segment_ratios = None

    icp_by_resp_cycle = physio.deform_traces_to_cycle_template(data = icp_filtered, 
                                                            times = times_icp,
                                                            cycle_times = cycle_times,
                                                            segment_ratios = segment_ratios,
                                                            points_per_cycle = 40
                                                            )
    return np.mean(np.ptp(icp_by_resp_cycle, axis = 1))


def icp_heart_and_resp_freq_domain(raw_icp, srate):
    p = {
    'spectrogram_win_size_secs':60,
    'heart_fband':(0.8,2.5),
    'resp_fband':(0.15,0.55),
    'rolling_N_time_spectrogram':5,
    }
    nperseg = int(p['spectrogram_win_size_secs'] * srate)
    nfft = int(nperseg)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(raw_icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    Sxx_icp = np.sqrt(Sxx_icp)
    da = xr.DataArray(data = Sxx_icp, dims = ['freq','time'], coords = {'freq':freqs, 'time':times_spectrum_s})
    heart_fband = p['heart_fband']
    resp_fband = p['resp_fband']
    rolling_N_time = p['rolling_N_time_spectrogram']
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    return heart_amplitude.values, resp_amplitude.values

def psi_test(raw_icp, srate):
    raw_signal = raw_icp.copy()
    time = np.arange(raw_signal.size) / srate
    assert np.any(~np.isnan(raw_signal))
    pipeline = ProcessingPipeline()
    classes, times = pipeline.process_signal(raw_signal, time)
    classification_results = np.argmax(classes, axis=1)

    # Remove the artefact class from the classification results
    non_artefact_mask = classification_results != 4
    non_artefact_classes = classification_results[non_artefact_mask]
    non_artefact_times = np.array(times)[non_artefact_mask]

    # Use rolling window to calculate PSI
    window_length = 5 * 60
    window_step = 10
    starting_time = non_artefact_times[0]

    psi_vector = []
    psi_times = []

    for win_start in np.arange(starting_time, non_artefact_times[-1] - window_length, window_step):
        # Get the classes in the time window
        win_end = win_start + window_length
        win_mask = (non_artefact_times >= win_start) & (non_artefact_times < win_end)
        win_classes = non_artefact_classes[win_mask]

        # Calculate the PSI
        class_counts = np.unique(win_classes, return_counts=True)
        psi = 0
        if len(win_classes) > 0:
            sum_count = np.sum(class_counts[1])
            for c, count in zip(class_counts[0], class_counts[1]):
                psi += (c+1) * count / sum_count

        # Append the PSI to the vector
        psi_vector.append(psi)
        psi_times.append(win_start + window_length / 2)
    psi_times = np.array(psi_times)
    psi_vector = np.array(psi_vector)
    return psi_vector

def icp_to_P2P1(icp_sig, srate): # define function here because else not know SubPeakDetector
    sd = SubPeakDetector(all_preds=False)
    srate_detect = int(np.round(srate))
    sd.detect_pulses(signal = icp_sig, fs = srate_detect)
    onsets_inds, ratio_P1P2_vector = sd.compute_ratio()
    return list(ratio_P1P2_vector)

def ratio_P1P2(raw_icp, srate):
    raw_signal = raw_icp.copy()
    raw_signal[np.isnan(raw_signal)] = np.nanmedian(raw_signal) # signal must not contain Nan
    ratio_P1P2_vector = icp_to_P2P1(raw_signal, srate)
    return ratio_P1P2_vector

def compute_computation_resource_requirements(duration_signal_hours):
    functions = {
        'icp_heart_time_domain' : icp_heart_time_domain,
        'icp_resp_time_domain':icp_resp_time_domain,
        'icp_heart_and_resp_freq_domain' : icp_heart_and_resp_freq_domain,
        'psi':psi_test,
        'P2P1_ratio':ratio_P1P2,
    }

    # sub_list = get_patient_list(['ICP','CO2'], threshold_duration_mins=duration_signal_hours*60)
    remove_subs = ['P9','P19']
    sub_list=['P17', 'P71', 'P11', 'LJ8', 'P2', 'LA19', 'P41', 'SP2', 'P80',
       'P85', 'P69', 'P21', 'P37', 'P60', 'P83', 'P6', 'P4', 'P87', 'P90',
       'P39', 'P16', 'BM3', 'P86', 'NY15', 'P56', 'P43', 'P70', 'P64',
       'P42', 'P50', 'P67', 'MF12', 'P74', 'HA1', 'P65', 'P20', 'P73',
       'P76', 'P82', 'P32', 'P3', 'PL20', 'P96', 'P98', 'P79', 'P57',
       'FC13', 'P78', 'LD16', 'P81', 'P87_fin', 'BJ11', 'P12', 'P93',
       'P53', 'P40', 'P89', 'P63', 'NN7', 'GA9', 'P18_fin', 'P75', 'P27',
       'P77', 'P13', 'P68', 'P90_fin', 'P62', 'P14', 'JR10', 'WJ14',
       'P66', 'P84']
    sub_list = [s for s in sub_list if not s in remove_subs]

    rows = []

    for sub in tqdm(sub_list):
        raw_icp, srate_icp, times_icp, raw_co2, srate_co2, times_co2 = load_raw_icp_resp(sub, duration_signal_hours)

        for func_name, func in functions.items():
            try:
                if func_name != 'icp_resp_time_domain':
                    info = start_measurement(func, (raw_icp, srate_icp))
                else:
                    info = start_measurement(func, (raw_icp, srate_icp, times_icp, raw_co2, srate_co2, times_co2))
                info['metric'] = func_name
                info['duration_signal_hours'] = duration_signal_hours
                info['patient'] = sub
                rows.append(info)
            except:
                print(sub, func_name)
                continue

    computation_requirements = pd.DataFrame(rows)
    computation_requirements['mem_GB'] = computation_requirements['mem'] / 1e9
    computation_requirements['mem_MB'] = computation_requirements['mem'] / 1e6
    return computation_requirements

if __name__ == "__main__":
    
    duration_signal_hours = 1
    # print(get_patient_list(['ICP','CO2'], threshold_duration_mins=duration_signal_hours*60))
    computation_requirements = compute_computation_resource_requirements(duration_signal_hours)
    # computation_requirements.to_excel(base_folder / 'figures' / 'slow_icp_rises_figs' / 'resource_requirements' / 'res_requirements_laptop_v4.xlsx')
    # computation_requirements = pd.read_excel(base_folder / 'figures' / 'slow_icp_rises_figs' / 'resource_requirements' / 'res_requirements.xlsx', index_col = 0)
    # print(computation_requirements)
    # print(computation_requirements.loc[:,[c for c in computation_requirements.columns if not c in ['patient','mem','mem_GB','duration_signal_hours']]].groupby('metric').agg(['mean','std','min','max']).round(3).T)
    # print(computation_requirements.loc[:,[c for c in computation_requirements.columns if not c in ['patient','mem','mem_GB','duration_signal_hours']]].groupby('metric').describe().round(3))
