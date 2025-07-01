import numpy as np
import xarray as xr
import pandas as pd
import physio
import pycns
import sys
import os
from tools import *
from overview_data_pycns import get_patient_list
from configuration import *
from params_multi_projects_jobs import *
import jobtools
import joblib
import json
from tqdm import tqdm

# DETECT RESP JOB

def detect_resp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    co2_stream = cns_reader.streams['CO2']
    datetimes = co2_stream.get_times(as_second = False)
    times = co2_stream.get_times(as_second = True)
    raw_co2 = co2_stream.get_data(with_times=False, apply_gain=True)
    srate = 1/(np.median(np.diff(times)))
    co2, resp_cycles = physio.compute_respiration(raw_co2, srate, parameter_preset = 'human_co2')
    resp_cycles['inspi_time'] = times[resp_cycles['inspi_index']]
    resp_cycles['expi_time'] = times[resp_cycles['expi_index']]
    resp_cycles['next_inspi_time'] = times[resp_cycles['next_inspi_index']]
    resp_cycles['inspi_date'] = datetimes[resp_cycles['inspi_index']].astype('datetime64[ns]')
    resp_cycles['expi_date'] = datetimes[resp_cycles['expi_index']].astype('datetime64[ns]')
    resp_cycles['next_inspi_date'] = datetimes[resp_cycles['next_inspi_index']].astype('datetime64[ns]')
    resp_cycles['cycle_freq_cpm'] = resp_cycles['cycle_freq'] * 60
    resp_cycles['sliding_variability_cpm'] = resp_cycles['cycle_freq_cpm'].rolling(p['N_cycles_sliding_sd'], center = True).std().bfill().ffill()

    if p['threshold_controlled_ventilation_sd_cpm'] == 'auto':
        bins = np.arange(0, 1,0.025)
        count, bins = np.histogram(resp_cycles['sliding_variability_cpm'], bins) # compute distribution
        d1 = np.gradient(count) # compute first derivative
        minimums = np.where((d1[:-1] < 0) & ((d1[1:] >= 0)))[0] # get minimums indices of 1st derivative
        if minimums.size > 0:
            first_minimum = minimums[0] # get index of first minimum
            thresh = bins[1:][first_minimum] # loc threshold value of variability = first trough of the distribution
        else:
            thresh = 0
    else:
        thresh = p['threshold_controlled_ventilation_sd_cpm']
    resp_cycles['is_ventilation_controlled'] = (resp_cycles['sliding_variability_cpm'] < thresh).astype(int)
    resp_cycles['is_ventilation_controlled'] = resp_cycles['is_ventilation_controlled'].rolling(p['N_cycles_sliding_ventilation_bool']).median().bfill().ffill()
    resp_cycles.loc[resp_cycles['is_ventilation_controlled'] == 0.5,'is_ventilation_controlled'] = 0
    # resp_cycles['is_ventilation_controlled'] = resp_cycles['is_ventilation_controlled'].bfill().ffill()
    return xr.Dataset(resp_cycles)

def test_detect_resp(sub):
    print(sub)
    ds = detect_resp(sub, **detect_resp_params).to_dataframe()
    print(ds)

detect_resp_job = jobtools.Job(precomputedir, 'detect_resp', detect_resp_params, detect_resp)
jobtools.register_job(detect_resp_job)

# DETECT ECG JOB

def detect_ecg(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    ecg_stream = cns_reader.streams['ECG_II']
    datetimes = ecg_stream.get_times(as_second = False)
    times = ecg_stream.get_times(as_second = True)
    raw_ecg = ecg_stream.get_data(with_times=False, apply_gain=False)
    srate = 1/(np.median(np.diff(times)))
    ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset = 'human_ecg')
    ecg_peaks['peak_time'] = times[ecg_peaks['peak_index']]
    ecg_peaks['peak_date'] = datetimes[ecg_peaks['peak_index']].astype('datetime64[ns]')
    return xr.Dataset(ecg_peaks)

def test_detect_ecg(sub):
    print(sub)
    ds = detect_ecg(sub, **detect_ecg_params).to_dataframe()
    print(ds)

detect_ecg_job = jobtools.Job(precomputedir, 'detect_ecg', detect_ecg_params, detect_ecg)
jobtools.register_job(detect_ecg_job)

# IHR JOB

def ihr(sub, **p):
    reader = pycns.CnsReader(data_path / sub)
    ecg_stream = reader.streams['ECG_II']
    ecg_times = ecg_stream.get_times(as_second = True)
    ecg_datetimes = ecg_stream.get_times(as_second = False)
    ecg_peaks = detect_ecg_job.get(sub).to_dataframe()
    peak_times = ecg_peaks['peak_time'].values
    peak_dates = ecg_peaks['peak_date'].values
    sr = p['srate_interp']
    step_s = 1 / sr
    step_us = step_s * 1e6
    ecg_times_interp = np.arange(peak_times[0], peak_times[-1], 1 / sr)
    inds_dates = np.searchsorted(ecg_times, ecg_times_interp)
    ecg_datetimes_interp = ecg_datetimes[inds_dates].astype('datetime64[ns]')
    ihr = physio.compute_instantaneous_rate(ecg_peaks, ecg_times_interp, limits=p['limits_bpm'], units = 'bpm' , interpolation_kind = p['interpolation_kind'])
    da_ihr = xr.DataArray(data = ihr, dims = ['datetime'], coords = {'datetime':ecg_datetimes_interp} , attrs = {'time':ecg_times_interp, 'srate':sr})
    ds = xr.Dataset()
    ds['ihr'] = da_ihr
    return ds

def test_ihr(sub):
    print(sub)
    ds = ihr(sub, **ihr_params).to_dataframe()
    print(ds['ihr'])

ihr_job = jobtools.Job(precomputedir, 'ihr', ihr_params, ihr)
jobtools.register_job(ihr_job)

# IRR JOB

def irr(sub, **p):
    reader = pycns.CnsReader(data_path / sub)
    co2_stream = reader.streams['CO2']
    co2_times = co2_stream.get_times(as_second = True)
    co2_datetimes = co2_stream.get_times(as_second = False)
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    irr_by_cycle_cpm = resp_cycles['cycle_freq_cpm'].values
    inspi_times =  resp_cycles['inspi_time'].values
    inspi_datetimes =  resp_cycles['inspi_date'].values
    sr = p['srate_interp']
    step_s = 1 / sr
    step_us = step_s * 1e6
    co2_times_interp = np.arange(inspi_times[0], inspi_times[-1], 1 / sr)
    inds_dates = np.searchsorted(co2_times, co2_times_interp)
    co2_datetimes_interp = co2_datetimes[inds_dates].astype('datetime64[ns]')
    interp = scipy.interpolate.interp1d(inspi_times, irr_by_cycle_cpm, kind=p['interpolation_kind'], axis=0,bounds_error=False, fill_value='extrapolate')
    instantaneous_rate = interp(co2_times_interp)
    da_irr = xr.DataArray(data = instantaneous_rate, dims = ['datetime'], coords = {'datetime':co2_datetimes_interp} , attrs = {'time':co2_times_interp, 'srate':sr})
    ds = xr.Dataset()
    ds['irr'] = da_irr
    return ds

def test_irr(sub):
    print(sub)
    ds = irr(sub, **ihr_params).to_dataframe()
    print(ds)

irr_job = jobtools.Job(precomputedir, 'irr', irr_params, irr)
jobtools.register_job(irr_job)

# RSA JOB

def rsa(sub, **p):
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    ecg_peaks = detect_ecg_job.get(sub).to_dataframe()
    rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(resp_cycles,
                                                        ecg_peaks,
                                                        srate=10.,
                                                        two_segment=True,
                                                        points_per_cycle=40,
                                                        )
    rsa_cycles['cycle_date'] = resp_cycles['inspi_date'].astype('datetime64[ns]')
    rsa_cycles['cycle_time'] = resp_cycles['inspi_time']
    rsa_cycles_clean = rsa_cycles['decay_amplitude'].values
    rsa_cycles_clean[rsa_cycles_clean > 30] = np.nan
    rsa_cycles['rsa_amplitude_clean'] = pd.Series(rsa_cycles_clean).rolling(30).median().ffill().bfill()
    return xr.Dataset(rsa_cycles)

def test_rsa(sub):
    print(sub)
    ds = rsa(sub, **rsa_params).to_dataframe()
    print(ds)

rsa_job = jobtools.Job(precomputedir, 'rsa', rsa_params, rsa)
jobtools.register_job(rsa_job)

# IRSA JOB

def irsa(sub, **p):
    reader = pycns.CnsReader(data_path / sub)
    co2_stream = reader.streams['CO2']
    co2_times = co2_stream.get_times(as_second = True)
    co2_datetimes = co2_stream.get_times(as_second = False)
    rsa_cycles = rsa_job.get(sub).to_dataframe()
    rsa_by_cycle_bpm = rsa_cycles['rsa_amplitude_clean'].values
    cycle_times =  rsa_cycles['cycle_time'].values
    cycle_datetimes =  rsa_cycles['cycle_date'].values
    sr = p['srate_interp']
    step_s = 1 / sr
    step_us = step_s * 1e6
    rsa_times_interp = np.arange(cycle_times[0], cycle_times[-1], 1 / sr)
    inds_dates = np.searchsorted(co2_times, rsa_times_interp)
    rsa_datetimes_interp = co2_datetimes[inds_dates].astype('datetime64[ns]')
    interp = scipy.interpolate.interp1d(cycle_times, rsa_by_cycle_bpm, kind=p['interpolation_kind'], axis=0,bounds_error=False, fill_value='extrapolate')
    instantaneous_rsa = interp(rsa_times_interp)
    da_irsa = xr.DataArray(data = instantaneous_rsa, dims = ['datetime'], coords = {'datetime':rsa_datetimes_interp} , attrs = {'time':rsa_times_interp, 'srate':sr})
    ds = xr.Dataset()
    ds['irsa'] = da_irsa
    return ds

def test_irsa(sub):
    print(sub)
    ds = irsa(sub, **irsa_params).to_dataframe()
    print(ds)

irsa_job = jobtools.Job(precomputedir, 'irsa', irsa_params, irsa)
jobtools.register_job(irsa_job)

# DETECT ABP JOB

def detect_abp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    all_streams = cns_reader.streams.keys()
    concat = []
    for abp_name in ['ABP','ART']:
        if abp_name in all_streams:
            if abp_name == 'ART' and sub == 'BM3': # baptiste has to remove ART files because protected by in wx
                continue
            stream = cns_reader.streams[abp_name]
            datetimes = stream.get_times(as_second = False)
            times = stream.get_times(as_second = True)
            srate_sig = 1/(np.median(np.diff(times)))
            raw_sig = stream.get_data(with_times=False, apply_gain=True)
            detections = compute_abp(raw_sig, srate_sig, date_vector = datetimes, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'], peak_prominence = p['peak_prominence'], h_distance_s = p['h_distance_s'], rise_amplitude_limits=p['rise_amplitude_limits'], amplitude_at_trough_low_limit = p['amplitude_at_trough_low_limit'])
            detections['trough_time'] = times[detections['trough_ind']]
            detections['peak_time'] = times[detections['peak_ind']]
            detections['next_trough_time'] = times[detections['next_trough_ind']]
            detections['stream_source_name'] = abp_name
            concat.append(detections)
    assert len(concat) != 0, 'no ABP stream available under the names ABP or ART'
    if len(concat) == 1:
        abp_return = concat[0]
    else:
        abp_return = pd.concat(concat).sort_values(by = 'trough_date').reset_index(drop = True)
    return xr.Dataset(abp_return)

def test_detect_abp(sub):
    print(sub)
    ds = detect_abp(sub, **detect_abp_params).to_dataframe()
    print(ds)

detect_abp_job = jobtools.Job(precomputedir, 'detect_abp', detect_abp_params, detect_abp)
jobtools.register_job(detect_abp_job)

# DETECT ICP JOB

def detect_icp(sub, **p):
    raw_folder = data_path / sub
    cns_reader = pycns.CnsReader(raw_folder)
    stream = cns_reader.streams['ICP']
    datetimes = stream.get_times(as_second = False)
    times = stream.get_times(as_second = True)
    srate_sig = 1/(np.median(np.diff(times)))
    raw_sig = stream.get_data(with_times=False, apply_gain=True)
    detections = compute_icp(raw_sig, srate_sig, date_vector = datetimes, lowcut = p['lowcut'], highcut = p['highcut'], order = p['order'], ftype = p['ftype'], peak_prominence = p['peak_prominence'], h_distance_s = p['h_distance_s'], rise_amplitude_limits=p['rise_amplitude_limits'], amplitude_at_trough_low_limit = p['amplitude_at_trough_low_limit'])
    detections['trough_time'] = times[detections['trough_ind']]
    detections['peak_time'] = times[detections['peak_ind']]
    detections['next_trough_time'] = times[detections['next_trough_ind']]
    return xr.Dataset(detections)

def test_detect_icp(sub):
    print(sub)
    ds = detect_icp(sub, **detect_icp_params).to_dataframe()
    print(ds)

detect_icp_job = jobtools.Job(precomputedir, 'detect_icp', detect_icp_params, detect_icp)
jobtools.register_job(detect_icp_job)

# DETECT PRX JOB

def prx(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    prx_r, prx_pval, dates = compute_prx_and_keep_nans(cns_reader, wsize_mean_secs = p['wsize_mean_secs'], wsize_corr_mins = p['wsize_corr_mins'], overlap_corr_prop = p['overlap_corr_prop'])
    da = xr.DataArray(data = prx_r, dims = ['date'], coords = {'date':dates})
    ds = xr.Dataset()
    ds['prx'] = da
    return xr.Dataset(ds)

def test_prx(sub):
    print(sub)
    ds = prx(sub, **prx_params).to_dataframe()
    print(ds['prx'])

prx_job = jobtools.Job(precomputedir, 'prx', prx_params, prx)
jobtools.register_job(prx_job)

# PSI 
def psi(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub, event_time_zone='Europe/Paris')
    icp_stream = cns_reader.streams['ICP']

    # Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
    plugin_dir = base_folder / 'ICMPWaveformClassificationPlugin' / 'plugin' / 'pulse_detection'
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))

    # Import the necessary plugin modules
    from classifier_pipeline import ProcessingPipeline
    from pulse_detector import Segmenter
    from pulse_classifier import Classifier

    
    raw_signal, datetimes = icp_stream.get_data(with_times = True, apply_gain = True)
    time_referenced = icp_stream.get_times(as_second = True)
    srate = 1/(np.median(np.diff(time_referenced)))
    time = np.arange(time_referenced.size) / srate
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
    psi_dates = datetimes[np.searchsorted(time, psi_times)]

    psi_da = xr.DataArray(data = psi_vector, dims = ['date'], coords=  {'date':psi_dates})
    ds = xr.Dataset()
    ds['psi'] = psi_da
    return ds

def test_psi(sub):
    print(sub)
    ds = psi(sub, **psi_params)
    print(ds['psi'])

psi_job = jobtools.Job(precomputedir, 'psi', psi_params, psi)
jobtools.register_job(psi_job)
# 

###### COMPLIANCE #######
# RATIO P1 P2 
def ratio_P1P2(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    stream_name = 'ICP'
    icp_stream = cns_reader.streams[stream_name]

    # Add the plugin directory to the system path, for us it is in the plugin/pulse_detection directory
    plugin_dir = base_folder / 'package_P2_P1' 
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))

    # Import the necessary plugin modules
    from p2p1.subpeaks import SubPeakDetector

    def icp_to_P2P1(icp_sig, icp_dates, srate): # define function here because else not know SubPeakDetector
        sd = SubPeakDetector(all_preds=False)
        srate_detect = int(np.round(srate))
        sd.detect_pulses(signal = icp_sig, fs = srate_detect)
        onsets_inds, ratio_P1P2_vector = sd.compute_ratio()
        onsets_inds[onsets_inds >= icp_dates.size] = icp_dates.size - 1
        onsets_dates = icp_dates[onsets_inds]
        if onsets_dates.size == ratio_P1P2_vector.size + 1:
            onsets_dates = onsets_dates[:-1]
        elif onsets_dates.size == ratio_P1P2_vector.size - 1:
            ratio_P1P2_vector = ratio_P1P2_vector[:-1]
        return list(ratio_P1P2_vector), list(onsets_dates)

    datetimes = icp_stream.get_times()
    times = icp_stream.get_times(as_second = True)
    srate = 1/(np.median(np.diff(times)))
    raw_signal = icp_stream.get_data(with_times = False, apply_gain = True)
    raw_signal[np.isnan(raw_signal)] = np.nanmedian(raw_signal) # signal must not contain Nan
    if p['down_sample']:
        down_sample_factor = 2
        raw_signal = scipy.signal.decimate(raw_signal, q = down_sample_factor)
        datetimes = datetimes[::down_sample_factor]

    total_load_size = raw_signal.size
    win_load_size = int(p['win_compute_duration_hours'] * 3600 * srate)
    n_wins = total_load_size // win_load_size

    ratio_P1P2_vector_list_values = []
    ratio_P1P2_vector_list_dates = []

    start = 0
    for i in range(n_wins): # loop over slices because too much memory on all signal for some patients
    # for i in tqdm(range(n_wins)): # loop over slices because too much memory on all signal for some patients
        stop = start + win_load_size
        local_icp = raw_signal[start:stop]
        local_dates = datetimes[start:stop]

        try:
            ratio_P1P2_vector, onsets_dates = icp_to_P2P1(local_icp, local_dates, srate)
        except:
            start = stop
            continue

        ratio_P1P2_vector_list_values.extend(ratio_P1P2_vector)
        ratio_P1P2_vector_list_dates.extend(onsets_dates)

        start = stop
    try:
        last_values, last_dates = icp_to_P2P1(raw_signal[stop:], datetimes[stop:], srate)
        ratio_P1P2_vector_list_values.extend(last_values)
        ratio_P1P2_vector_list_dates.extend(last_dates)
    except:
        pass

    ratio_P1P2_da = xr.DataArray(data = ratio_P1P2_vector_list_values, dims = ['date'], coords=  {'date':np.array(ratio_P1P2_vector_list_dates, dtype='datetime64[ns]')})
    ds = xr.Dataset()
    ds['ratio_P1P2'] = ratio_P1P2_da
    return ds

def test_ratio_P1P2(sub):
    print(sub)
    ds = ratio_P1P2(sub, **ratio_P1P2_params)
    print(ds['ratio_P1P2'])

ratio_P1P2_job = jobtools.Job(precomputedir, 'ratio_P1P2', ratio_P1P2_params, ratio_P1P2)
jobtools.register_job(ratio_P1P2_job)

# HEART RESP IN ICP
def compute_heart_resp_spectral_ratio_in_icp(icp, srate, wsize_secs = 50, resp_fband = (0.12,0.6), heart_fband = (0.8,2.5), rolling_N_time = 5):
    nperseg = int(wsize_secs * srate)
    nfft = int(nperseg)

    # Compute spectro ICP
    freqs, times_spectrum_s, Sxx_icp = scipy.signal.spectrogram(icp, fs = srate, nperseg =  nperseg, nfft = nfft)
    Sxx_icp = np.sqrt(Sxx_icp)
    da = xr.DataArray(data = Sxx_icp, dims = ['freq','time'], coords = {'freq':freqs, 'time':times_spectrum_s})
    resp_amplitude = da.loc[resp_fband[0]:resp_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    resp_freq = da.loc[resp_fband[0]:resp_fband[1],:].idxmax('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    heart_amplitude = da.loc[heart_fband[0]:heart_fband[1],:].max('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    heart_freq = da.loc[heart_fband[0]:heart_fband[1],:].idxmax('freq').rolling(time = rolling_N_time).median().bfill('time').ffill('time')
    ratio_heart_resp = heart_amplitude / resp_amplitude
    res = {'times':times_spectrum_s,'heart_in_icp_spectrum':heart_amplitude.values,'heart_freq_from_icp':heart_freq, 'resp_in_icp_spectrum':resp_amplitude.values, 'resp_freq_from_icp':resp_freq, 'ratio_heart_resp_in_icp_spectrum':ratio_heart_resp.values}
    return res

def heart_resp_in_icp(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    stream_name = 'ICP'
    icp_stream = cns_reader.streams[stream_name]
    raw_times_referenced = icp_stream.get_times(as_second = True)
    srate = 1/(np.median(np.diff(raw_times_referenced)))
    raw_signal, dates = icp_stream.get_data(with_times = True, apply_gain = True)
    raw_times = np.arange(raw_signal.size) / srate
    # print(np.isnan(raw_signal).sum())
    # raw_signal[np.isnan(raw_signal)] = np.nanmedian(raw_signal) # signal must not contain Nan
    res = compute_heart_resp_spectral_ratio_in_icp(raw_signal,
                                                   srate,
                                                   wsize_secs=p['spectrogram_win_size_secs'],
                                                   resp_fband=p['resp_fband'],
                                                   heart_fband=p['heart_fband'],
                                                   rolling_N_time=p['rolling_N_time_spectrogram'],
                                                   )
    datetimes = dates[np.searchsorted(raw_times, res['times'])]
    df_res = pd.DataFrame(res)
    df_res['times'] = datetimes.copy()
    df_res = df_res.rename(columns = {'times':'datetime'}).set_index('datetime')
    da = xr.DataArray(data = df_res.values.T, 
                      dims = ['feature','datetime'],
                      coords = {'feature':df_res.columns.tolist(), 'datetime':df_res.index}
                      )
    ds = xr.Dataset()
    ds['heart_resp_in_icp'] = da
    return ds

def test_heart_resp_in_icp(sub):
    print(sub)
    ds = heart_resp_in_icp(sub, **heart_resp_in_icp_params)
    print(ds)

heart_resp_in_icp_job = jobtools.Job(precomputedir, 'heart_resp_in_icp', heart_resp_in_icp_params, heart_resp_in_icp)
jobtools.register_job(heart_resp_in_icp_job)

# heart_rate_by_resp_cycle
def heart_rate_by_resp_cycle(sub, **p):
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    med_cycle_ratio = resp_cycles['cycle_ratio'].median()

    ihr_da = ihr_job.get(sub)['ihr']
    times_ihr = ihr_da.attrs['time']
    ihr_trace = ihr_da.values
 
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_ihr[0]) & (resp_cycles['next_inspi_time'] < times_ihr[-1])]
    med_cycle_ratio = resp_cycles['cycle_ratio'].median()

    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
        segment_ratios = med_cycle_ratio    

    cyclic_cardiac_rate = physio.deform_traces_to_cycle_template(data = ihr_trace, 
                                                            times = times_ihr,
                                                            cycle_times = cycle_times,
                                                            segment_ratios = segment_ratios,
                                                            points_per_cycle = p['points_per_cycle']
                                                            )
    
    da = xr.DataArray(data = cyclic_cardiac_rate,
                      dims = ['cycle_date','phase'],
                      coords = {'cycle_date':resp_cycles['inspi_date'].values, 'phase':np.linspace(0,1,p['points_per_cycle'])},
                      attrs = {'med_cycle_ratio':med_cycle_ratio}
                      )
    ds = xr.Dataset()
    ds['heart_rate_by_resp_cycle'] = da
    return ds

def test_heart_rate_by_resp_cycle(sub):
    print(sub)
    ds = heart_rate_by_resp_cycle(sub, **heart_rate_by_resp_cycle_params)
    print(ds)

heart_rate_by_resp_cycle_job = jobtools.Job(precomputedir, 'heart_rate_by_resp_cycle', heart_rate_by_resp_cycle_params, heart_rate_by_resp_cycle)
jobtools.register_job(heart_rate_by_resp_cycle_job)

# abp_by_resp_cycle
def abp_by_resp_cycle(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    stream_names = list(cns_reader.streams)
    assert 'ABP' in stream_names or 'ART' in stream_names
    if 'ABP' in stream_names and not 'ART' in stream_names:
        abp_stream_name = 'ABP'
    elif 'ART' in stream_names and not 'ABP' in stream_names:
        abp_stream_name = 'ART'
    else:
        abp_test_stream = cns_reader.streams['ABP']
        art_test_stream = cns_reader.streams['ART']
        abp_stream_name = 'ABP' if abp_test_stream.shape[0] > art_test_stream.shape[0] else 'ART'

    abp_stream = cns_reader.streams[abp_stream_name]
    times_abp = abp_stream.get_times(as_second = True)
    srate_abp = 1/(np.median(np.diff(times_abp)))
    raw_abp = abp_stream.get_data(with_times = False, apply_gain = True)
    abp_filtered = iirfilt(raw_abp, srate_abp, highcut = p['highcut'], order = p['order'], ftype = p['ftype'])
 
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_abp[0]) & (resp_cycles['next_inspi_time'] < times_abp[-1])]
    med_cycle_ratio = resp_cycles['cycle_ratio'].median()

    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
        segment_ratios = med_cycle_ratio
    
    abp_by_resp_cycle = physio.deform_traces_to_cycle_template(data = abp_filtered, 
                                                            times = times_abp,
                                                            cycle_times = cycle_times,
                                                            segment_ratios = segment_ratios,
                                                            points_per_cycle = p['points_per_cycle']
                                                            )
    da = xr.DataArray(data = abp_by_resp_cycle,
                      dims = ['cycle_date','phase'],
                      coords = {'cycle_date':resp_cycles['inspi_date'].values, 'phase':np.linspace(0,1,p['points_per_cycle'])},
                      attrs = {'cycle_ratio':segment_ratios}
                      )
    ds = xr.Dataset()
    ds['abp_by_resp_cycle'] = da
    return ds

def test_abp_by_resp_cycle(sub):
    print(sub)
    ds = abp_by_resp_cycle(sub, **abp_by_resp_cycle_params)
    print(ds)

abp_by_resp_cycle_job = jobtools.Job(precomputedir, 'abp_by_resp_cycle', abp_by_resp_cycle_params, abp_by_resp_cycle)
jobtools.register_job(abp_by_resp_cycle_job)

# icp_by_resp_cycle
def icp_by_resp_cycle(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    icp_stream = cns_reader.streams['ICP']
    times_icp = icp_stream.get_times(as_second = True)
    srate_icp = 1/(np.median(np.diff(times_icp)))
    raw_icp = icp_stream.get_data(with_times = False, apply_gain = True)
    icp_filtered = iirfilt(raw_icp, srate_icp, highcut = p['highcut'], order = p['order'], ftype = p['ftype'])
 
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_icp[0]) & (resp_cycles['next_inspi_time'] < times_icp[-1])]
    med_cycle_ratio = resp_cycles['cycle_ratio'].median()

    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
        segment_ratios = med_cycle_ratio
    
    icp_by_resp_cycle = physio.deform_traces_to_cycle_template(data = icp_filtered, 
                                                            times = times_icp,
                                                            cycle_times = cycle_times,
                                                            segment_ratios = segment_ratios,
                                                            points_per_cycle = p['points_per_cycle']
                                                            )
    da = xr.DataArray(data = icp_by_resp_cycle,
                      dims = ['cycle_date','phase'],
                      coords = {'cycle_date':resp_cycles['inspi_date'].values, 'phase':np.linspace(0,1,p['points_per_cycle'])},
                      attrs = {'cycle_ratio':segment_ratios}
                      )
    ds = xr.Dataset()
    ds['icp_by_resp_cycle'] = da
    return ds

def test_icp_by_resp_cycle(sub):
    print(sub)
    ds = icp_by_resp_cycle(sub, **icp_by_resp_cycle_params)
    print(ds)

icp_by_resp_cycle_job = jobtools.Job(precomputedir, 'icp_by_resp_cycle', icp_by_resp_cycle_params, icp_by_resp_cycle)
jobtools.register_job(icp_by_resp_cycle_job)

# ICP / ABP resp modulated
def icp_resp_modulated(sub, **p):
    icp_by_resp_cycle = icp_by_resp_cycle_job.get(sub)['icp_by_resp_cycle']
    cycles_dates = icp_by_resp_cycle['cycle_date'].values
    icp_by_resp_cycle = icp_by_resp_cycle.values
    ptp = np.ptp(icp_by_resp_cycle, axis = 1)
    if np.any(np.isnan(ptp)):
        ptp = pd.Series(ptp).bfill().fill().values
    da = xr.DataArray(data = ptp, 
                      dims = ['datetime'],
                      coords = {'datetime':cycles_dates.astype('datetime64[ns]')}
                      )
    da.attrs['unit'] = 'mmHg'
    ds = xr.Dataset()
    ds['icp_resp_modulated'] = da
    return ds

def test_icp_resp_modulated(sub):
    print(sub)
    ds = icp_resp_modulated(sub, **icp_resp_modulated_params)
    print(ds['icp_resp_modulated'])

icp_resp_modulated_job = jobtools.Job(precomputedir, 'icp_resp_modulated', icp_resp_modulated_params, icp_resp_modulated)
jobtools.register_job(icp_resp_modulated_job)

def abp_resp_modulated(sub, **p):
    abp_by_resp_cycle = abp_by_resp_cycle_job.get(sub)['abp_by_resp_cycle']
    cycles_dates = abp_by_resp_cycle['cycle_date'].values
    abp_by_resp_cycle = abp_by_resp_cycle.values
    ptp = np.ptp(abp_by_resp_cycle, axis = 1)
    if np.any(np.isnan(ptp)):
        ptp = pd.Series(ptp).bfill().fill().values
    da = xr.DataArray(data = ptp, 
                      dims = ['datetime'],
                      coords = {'datetime':cycles_dates.astype('datetime64[ns]')}
                      )
    da.attrs['unit'] = 'mmHg'
    ds = xr.Dataset()
    ds['abp_resp_modulated'] = da
    return ds

def test_abp_resp_modulated(sub):
    print(sub)
    ds = abp_resp_modulated(sub, **abp_resp_modulated_params)
    print(ds['abp_resp_modulated'])

abp_resp_modulated_job = jobtools.Job(precomputedir, 'abp_resp_modulated', abp_resp_modulated_params, abp_resp_modulated)
jobtools.register_job(abp_resp_modulated_job)

# ICP / ABP PULSE BY RESP CYCLE FUNCTION
def pulse_by_resp_cycle(sub, detect_pulse_job, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    times_co2_referenced = cns_reader.streams['CO2'].get_times(as_second = True)
    resp_cycles = detect_resp_job.get(sub).to_dataframe()
    detections = detect_pulse_job.get(sub).to_dataframe()
    times_co2_referenced_cropped = times_co2_referenced[(times_co2_referenced >= detections['trough_time'].values[0]) & (times_co2_referenced < detections['next_trough_time'].values[-1])]
    f = scipy.interpolate.interp1d(detections['peak_time'], 
                                   detections['rise_amplitude'], 
                                   fill_value=(min(detections['rise_amplitude']),max(detections['rise_amplitude'])),
                                   bounds_error=False, 
                                   kind = 'linear'
                                   )
    xnew = times_co2_referenced_cropped.copy()
    pulse_amplitude_interpolated = f(xnew)
    resp_cycles = resp_cycles[(resp_cycles['inspi_time'] > times_co2_referenced_cropped[0]) & (resp_cycles['next_inspi_time'] < times_co2_referenced_cropped[-1])]
    assert p['segmentation_deformation'] in ['mono','bi'], 'should be "mono" or "bi"'
    if p['segmentation_deformation'] == 'mono':
        cycle_times = resp_cycles[['inspi_time','next_inspi_time']].values
        segment_ratios = None
    elif p['segmentation_deformation'] == 'bi':
        cycle_times = resp_cycles[['inspi_time','expi_time','next_inspi_time']].values
        segment_ratios = resp_cycles['cycle_ratio'].median()
    pulse_amplitude_by_resp = physio.deform_traces_to_cycle_template(data = pulse_amplitude_interpolated, 
                                                               times = times_co2_referenced_cropped,
                                                               cycle_times = cycle_times,
                                                               segment_ratios = segment_ratios,
                                                               points_per_cycle = p['points_per_cycle'],
                                                              )
    da = xr.DataArray(data = pulse_amplitude_by_resp,
                      dims = ['cycle_date','phase'],
                      coords = {'cycle_date':resp_cycles['inspi_date'].values, 'phase':np.linspace(0,1,p['points_per_cycle'])},
                      attrs = {'cycle_ratio':segment_ratios}
                      )
    return da

# ICP PULSE BY RESP
def icp_pulse_by_resp_cycle(sub, **p):
    da = pulse_by_resp_cycle(sub, detect_icp_job, **p)
    ds = xr.Dataset()
    ds['icp_pulse_by_resp_cycle'] = da
    return ds

def test_icp_pulse_by_resp_cycle(sub):
    print(sub)
    ds = icp_pulse_by_resp_cycle(sub, **icp_pulse_by_resp_cycle_params)
    print(ds['icp_pulse_by_resp_cycle'])
    # print(ds['icp_pulse_by_resp_cycle'].shape)

icp_pulse_by_resp_cycle_job = jobtools.Job(precomputedir, 'icp_pulse_by_resp_cycle', icp_pulse_by_resp_cycle_params, icp_pulse_by_resp_cycle)
jobtools.register_job(icp_pulse_by_resp_cycle_job)

# ABP PULSE BY RESP
def abp_pulse_by_resp_cycle(sub, **p):
    da = pulse_by_resp_cycle(sub, detect_abp_job, **p)
    ds = xr.Dataset()
    ds['abp_pulse_by_resp_cycle'] = da
    return ds

def test_abp_pulse_by_resp_cycle(sub):
    print(sub)
    ds = abp_pulse_by_resp_cycle(sub, **abp_pulse_by_resp_cycle_params)
    print(ds['abp_pulse_by_resp_cycle'])

abp_pulse_by_resp_cycle_job = jobtools.Job(precomputedir, 'abp_pulse_by_resp_cycle', abp_pulse_by_resp_cycle_params, abp_pulse_by_resp_cycle)
jobtools.register_job(abp_pulse_by_resp_cycle_job)

# ICP PULSE DEFORMED BY RESP
def icp_pulse_resp_modulated(sub, **p):
    da_pulse_by_resp = icp_pulse_by_resp_cycle_job.get(sub)['icp_pulse_by_resp_cycle']
    pulse_by_resp = da_pulse_by_resp.values
    ptp_for_each_cycle = np.ptp(pulse_by_resp, axis = 1)
    if p['unit_type'] == 'relative':
        m_for_each_cycle = np.mean(pulse_by_resp, axis = 1)
        trace_modulated = np.abs((ptp_for_each_cycle / m_for_each_cycle)) * 100
    elif p['unit_type'] == 'absolute':
        trace_modulated = ptp_for_each_cycle.copy()
    res = pd.Series(trace_modulated)
    res.index = da_pulse_by_resp['cycle_date'].values

    if res.isna().sum() > 0:
        res = res.ffill().bfill()
    da = xr.DataArray(data = res.values, 
                      dims = ['datetime'],
                      coords = {'datetime':res.index}
                      )
    if p['unit_type'] == 'absolute':
        unit = 'mmHg'
    elif p['unit_type'] == 'relative':
        unit = '%'
    da.attrs['unit'] = unit
    ds = xr.Dataset()
    ds['icp_pulse_resp_modulated'] = da
    return ds

def test_icp_pulse_resp_modulated(sub):
    print(sub)
    ds = icp_pulse_resp_modulated(sub, **icp_pulse_resp_modulated_params)
    print(ds['icp_pulse_resp_modulated'])
    print(ds['icp_pulse_resp_modulated'].shape)

icp_pulse_resp_modulated_job = jobtools.Job(precomputedir, 'icp_pulse_resp_modulated', icp_pulse_resp_modulated_params, icp_pulse_resp_modulated)
jobtools.register_job(icp_pulse_resp_modulated_job)

def abp_pulse_resp_modulated(sub, **p):
    da_pulse_by_resp = abp_pulse_by_resp_cycle_job.get(sub)['abp_pulse_by_resp_cycle']
    pulse_by_resp = da_pulse_by_resp.values
    ptp_for_each_cycle = np.ptp(pulse_by_resp, axis = 1)
    if p['unit_type'] == 'relative':
        m_for_each_cycle = np.mean(pulse_by_resp, axis = 1)
        trace_modulated = np.abs((ptp_for_each_cycle / m_for_each_cycle)) * 100
    elif p['unit_type'] == 'absolute':
        trace_modulated = ptp_for_each_cycle.copy()
    res = pd.Series(trace_modulated)
    res.index = da_pulse_by_resp['cycle_date'].values

    if res.isna().sum() > 0:
        res = res.ffill().bfill()
    da = xr.DataArray(data = res.values, 
                      dims = ['datetime'],
                      coords = {'datetime':res.index}
                      )
    if p['unit_type'] == 'absolute':
        unit = 'mmHg'
    elif p['unit_type'] == 'relative':
        unit = '%'
    da.attrs['unit'] = unit
    ds = xr.Dataset()
    ds['abp_pulse_resp_modulated'] = da
    return ds

def test_abp_pulse_resp_modulated(sub):
    print(sub)
    ds = abp_pulse_resp_modulated(sub, **abp_pulse_resp_modulated_params)
    print(ds['abp_pulse_resp_modulated'])

abp_pulse_resp_modulated_job = jobtools.Job(precomputedir, 'abp_pulse_resp_modulated', abp_pulse_resp_modulated_params, abp_pulse_resp_modulated)
jobtools.register_job(abp_pulse_resp_modulated_job)

# RAQ ICP / ABP
def raq_func(Arp, AAvp, **p):
    cycles_intersect_date = np.intersect1d(Arp['datetime'].values, AAvp['datetime'].values, assume_unique=True)
    raq = Arp.loc[cycles_intersect_date].values / AAvp.loc[cycles_intersect_date].values
    med = np.nanmedian(raq)
    mad = scipy.stats.median_abs_deviation(raq, nan_policy='omit', scale= 0.67449)
    mask_bad = (raq < 0) | (raq > med + p['n_mad_clean']  * mad)
    raq[mask_bad] = np.nan
    da_raq = xr.DataArray(data = raq, dims = ['cycle_date'], coords = {'cycle_date':cycles_intersect_date})
    return da_raq

def raq_icp(sub, **p):
    Arp = icp_resp_modulated_job.get(sub)['icp_resp_modulated']
    AAvp = icp_pulse_resp_modulated_job.get(sub)['icp_pulse_resp_modulated']
    da_raq = raq_func(Arp, AAvp, **p)
    ds = xr.Dataset()
    ds['raq_icp'] = da_raq
    return ds

def test_raq_icp(sub):
    print(sub)
    ds = raq_icp(sub, **raq_icp_params)
    print(ds['raq_icp'])

raq_icp_job = jobtools.Job(precomputedir, 'raq_icp', raq_icp_params, raq_icp)
jobtools.register_job(raq_icp_job)

def raq_abp(sub, **p):
    Arp = abp_resp_modulated_job.get(sub)['abp_resp_modulated']
    AAvp = abp_pulse_resp_modulated_job.get(sub)['abp_pulse_resp_modulated']
    da_raq = raq_func(Arp, AAvp, **p)
    ds = xr.Dataset()
    ds['raq_abp'] = da_raq
    return ds

def test_raq_abp(sub):
    print(sub)
    ds = raq_abp(sub, **raq_abp_params)
    print(ds['raq_abp'])

raq_abp_job = jobtools.Job(precomputedir, 'raq_abp', raq_abp_params, raq_abp)
jobtools.register_job(raq_abp_job)

######

def sub_eeg_chan_keys():
    keys = []
    for sub in get_patient_list(['Scalp','ECoG'], patient_type='SD_ICU'):
        chans = CnsReader(data_path / sub).streams['EEG'].channel_names
        for chan in chans:
            key = (sub, chan)
            keys.append(key)
    return keys


# IHR ROLL JOB
def ihr_roll(sub, **p):
    win_size_rolling_secs = p['win_size_rolling_secs']
    ihr_da = ihr_job.get(sub)['ihr']
    datetimes = ihr_da['datetime'].values
    srate = ihr_da.attrs['srate']
    n_roll_inds = int(srate * win_size_rolling_secs)
    med_roll = ihr_da.rolling(datetime = n_roll_inds, center = True).median().bfill('datetime').ffill('datetime')
    mad_roll = ihr_da.rolling(datetime = n_roll_inds, center = True).reduce(mad_func).bfill('datetime').ffill('datetime')
    ihr_roll_da = xr.DataArray(data = np.nan, 
                               dims = ['feature','datetime'], 
                               coords = {'feature':['med_trace','mad_trace'], 'datetime':datetimes},
                               attrs = {'srate':srate}
                              )
    ihr_roll_da.loc['med_trace',:] = med_roll
    ihr_roll_da.loc['mad_trace',:] = mad_roll
    ds = xr.Dataset()
    ds['ihr_roll'] = ihr_roll_da
    return ds

def test_ihr_roll(sub):
    print(sub)
    ds = ihr_roll(sub, **ihr_roll_params)
    print(ds)

ihr_roll_job = jobtools.Job(precomputedir, 'ihr_roll', ihr_roll_params, ihr_roll)
jobtools.register_job(ihr_roll_job)

# IRR ROLL JOB
def irr_roll(sub, **p):
    win_size_rolling_secs = p['win_size_rolling_secs']
    irr_da = irr_job.get(sub)['irr']
    datetimes = irr_da['datetime'].values
    srate = irr_da.attrs['srate']
    n_roll_inds = int(srate * win_size_rolling_secs)
    med_roll = irr_da.rolling(datetime = n_roll_inds, center = True).median().bfill('datetime').ffill('datetime')
    mad_roll = irr_da.rolling(datetime = n_roll_inds, center = True).reduce(mad_func).bfill('datetime').ffill('datetime')
    irr_roll_da = xr.DataArray(data = np.nan, 
                               dims = ['feature','datetime'], 
                               coords = {'feature':['med_trace','mad_trace'], 'datetime':datetimes},
                               attrs = {'srate':srate}
                              )
    irr_roll_da.loc['med_trace',:] = med_roll
    irr_roll_da.loc['mad_trace',:] = mad_roll
    ds = xr.Dataset()
    ds['irr_roll'] = irr_roll_da
    return ds

def test_irr_roll(sub):
    print(sub)
    ds = irr_roll(sub, **irr_roll_params)
    print(ds)

irr_roll_job = jobtools.Job(precomputedir, 'irr_roll', irr_roll_params, irr_roll)
jobtools.register_job(irr_roll_job)

# IRSA ROLL JOB
def irsa_roll(sub, **p):
    win_size_rolling_secs = p['win_size_rolling_secs']
    irsa_da = irsa_job.get(sub)['irsa']
    datetimes = irsa_da['datetime'].values
    srate = irsa_da.attrs['srate']
    n_roll_inds = int(srate * win_size_rolling_secs)
    med_roll = irsa_da.rolling(datetime = n_roll_inds, center = True).median().bfill('datetime').ffill('datetime')
    irsa_roll_da = xr.DataArray(data = np.nan, 
                               dims = ['feature','datetime'], 
                               coords = {'feature':['med_trace'], 'datetime':datetimes},
                               attrs = {'srate':srate}
                              )
    irsa_roll_da.loc['med_trace',:] = med_roll
    ds = xr.Dataset()
    ds['irsa_roll'] = irsa_roll_da
    return ds

def test_irsa_roll(sub):
    print(sub)
    ds = irsa_roll(sub, **irsa_roll_params)
    print(ds)

irsa_roll_job = jobtools.Job(precomputedir, 'irsa_roll', irsa_roll_params, irsa_roll)
jobtools.register_job(irsa_roll_job)

def compute_all():
    # jobtools.compute_job_list(load_one_eeg_job, sub_eeg_chan_keys(), force_recompute=False, engine='loop')

    jobtools.compute_job_list(detect_resp_job, [(sub,) for sub in get_patient_list(['CO2'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_resp_job, [(sub,) for sub in get_patient_list(['CO2'])], force_recompute=False, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(detect_ecg_job, [(sub,) for sub in get_patient_list(['ECG_II'])], force_recompute=False, engine = 'loop')


    # jobtools.compute_job_list(ihr_job, [(sub,) for sub in get_patient_list(['ECG_II'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(irr_job, [(sub,) for sub in get_patient_list(['CO2'])], force_recompute=False, engine = 'loop')

    # jobtools.compute_job_list(rsa_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2'])], force_recompute=True, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(irsa_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2'])], force_recompute=True, engine = 'joblib', n_jobs = 5)

    # abp_art_subs = get_patient_list(['ABP']) + get_patient_list(['ART'])
    # abp_art_subs = list(set(abp_art_subs))
    # jobtools.compute_job_list(detect_abp_job, [(sub,) for sub in abp_art_subs], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_abp_job, [(sub,) for sub in abp_art_subs], force_recompute=True, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(detect_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(detect_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=True, engine = 'joblib', n_jobs = 5)

    # icp_abp_art_subs = get_patient_list(['ICP','ABP']) + get_patient_list(['ICP','ART'])
    # icp_abp_art_subs = list(set(icp_abp_art_subs))
    # jobtools.compute_job_list(prx_job, [(sub,) for sub in icp_abp_art_subs], force_recompute=False, engine = 'loop')
    

    # jobtools.compute_job_list(psi_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(psi_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(ratio_P1P2_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')
    # jobtools.compute_job_list(ratio_P1P2_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'joblib', n_jobs = 10)

    # jobtools.compute_job_list(heart_resp_in_icp_job, [(sub,) for sub in get_patient_list(['ICP'])], force_recompute=False, engine = 'loop')

    # jobtools.compute_job_list(heart_rate_by_resp_cycle_job, [(s,) for s in get_patient_list(['ECG_II','CO2'])] , force_recompute=True, engine = 'loop')
    # jobtools.compute_job_list(abp_by_resp_cycle_job, [(s,) for s in list(set(get_patient_list(['ABP','CO2']) + get_patient_list(['ART','CO2'])))], force_recompute=True, engine = 'loop')
    # jobtools.compute_job_list(icp_by_resp_cycle_job, [(s,) for s in get_patient_list(['ICP','CO2'])], force_recompute=True, engine = 'loop')
    # jobtools.compute_job_list(icp_resp_modulated_job, [(s,) for s in get_patient_list(['ICP','CO2'])], force_recompute=True, engine = 'loop')
    # jobtools.compute_job_list(abp_resp_modulated_job, [(s,) for s in list(set(get_patient_list(['ABP','CO2']) + get_patient_list(['ART','CO2'])))], force_recompute=True, engine = 'loop')

    # jobtools.compute_job_list(icp_pulse_by_resp_cycle_job, [(sub,) for sub in get_patient_list(['ICP','CO2'])], force_recompute=False, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(abp_pulse_by_resp_cycle_job, [(sub,) for sub in list(set(get_patient_list(['ABP','CO2']) + get_patient_list(['ART','CO2'])))], force_recompute=False, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(icp_pulse_resp_modulated_job, [(sub,) for sub in get_patient_list(['ICP','CO2'])], force_recompute=True, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(abp_pulse_resp_modulated_job, [(sub,) for sub in list(set(get_patient_list(['ABP','CO2']) + get_patient_list(['ART','CO2'])))], force_recompute=True, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(raq_icp_job, [(sub,) for sub in get_patient_list(['ICP','CO2'])], force_recompute=True, engine = 'joblib', n_jobs = 5)
    # jobtools.compute_job_list(raq_abp_job, [(sub,) for sub in list(set(get_patient_list(['ABP','CO2']) + get_patient_list(['ART','CO2'])))], force_recompute=True, engine = 'joblib', n_jobs = 5)

    # jobtools.compute_job_list(ihr_roll_job, [(sub,) for sub in get_patient_list(['ECG_II'])], force_recompute=True, engine = 'joblib', n_jobs = 10)
    # jobtools.compute_job_list(irr_roll_job, [(sub,) for sub in get_patient_list(['CO2'])], force_recompute=True, engine = 'joblib', n_jobs = 10)
    # jobtools.compute_job_list(irsa_roll_job, [(sub,) for sub in get_patient_list(['ECG_II','CO2])], force_recompute=True, engine = 'joblib', n_jobs = 10)


if __name__ == "__main__":
    # test_load_one_eeg('MF12','ECoGA4')
    # test_detect_resp('NY15') # P61, P95 = flat co2
    # test_detect_ecg('MF12')
    # test_ihr('P47')
    # test_irr('MF12')
    # test_rsa('MF12')
    # test_irsa('MF12')
    # test_detect_abp('MF12')
    # test_detect_icp('MF12')
    # test_prx('P39')
    # test_psi('P43')
    # test_ratio_P1P2('P80')
    # test_heart_resp_in_icp('BJ11')
    # test_heart_rate_by_resp_cycle('MF12')
    # test_abp_by_resp_cycle('MF12')
    # test_icp_by_resp_cycle('MF12')
    # test_abp_resp_modulated('MF12')
    # test_icp_resp_modulated('MF12')
    # test_icp_pulse_by_resp_cycle('MF12')
    # test_abp_pulse_by_resp_cycle('MF12')
    # test_icp_pulse_resp_modulated('MF12')
    # test_abp_pulse_resp_modulated('MF12')
    # test_raq_icp('JR10')
    # test_raq_abp('MF12')
    # test_ihr_roll('P47')
    # test_irr_roll('GA9')
    # test_irsa_roll('MF12')

    compute_all()
