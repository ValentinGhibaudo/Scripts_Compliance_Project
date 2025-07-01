import numpy as np
import xarray as xr
import pandas as pd
from pycns import CnsStream, CnsReader
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import jobtools
from tqdm import tqdm
from matplotlib.lines import Line2D
from icca_tools import *
from icca_jobs import *
from multi_projects_jobs import heart_resp_in_icp_job, icp_pulse_resp_modulated_job, abp_pulse_resp_modulated_job, psi_job, ratio_P1P2_job, detect_icp_job, prx_job, detect_abp_job, icp_resp_modulated_job, abp_resp_modulated_job, detect_ecg_job, detect_resp_job, raq_icp_job, raq_abp_job
from params_multi_projects_jobs import *
from tools import *
from overview_data_pycns import get_patient_list, get_patient_durations_by_stream_job
from configuration import *

icp_filter_for_detection_params = {
        'params_filtering':{'half_period_filter_min':15,
                        'ftype':'bessel',
                        'order':10
                        },
        'down_sampling_factor':1000,
}

icp_filter_for_trough_filtering_params = {
    'highcut': 0.02,
    'order':4,
    'ftype':'butter',
    'down_sampling_factor':100,
    }

abp_filter_params = {
        'params_filtering':{'half_period_filter_min':15,
                        'ftype':'bessel',
                        'order':10
                        },
        'down_sampling_factor':1000,
}

slow_icp_rise_detection_params = {
        'icp_filter_for_detection_params':icp_filter_for_detection_params,
        'icp_filter_for_trough_filtering_params':icp_filter_for_trough_filtering_params,
        'params_detection':{'min_rise_duration_min':15,
                            'min_rise_amplitude_mmHg':5, 
                            'min_peak_amplitude_mmHg':20,
                            'max_peak_amplitude_raw_mmHg':80,
                            'min_decay_amplitude_mmHg':2,
                            'max_trough_amplitude_smoothed_mmHg':20
                            },
}

detection_fig_params = {
    'icp_filter_for_detection_params':icp_filter_for_detection_params,
    'icp_filter_for_trough_filtering_params':icp_filter_for_trough_filtering_params,
    'slow_icp_rise_detection_params':slow_icp_rise_detection_params,
    'margin_mins':30, # mins
    'highcut_sig':0.1, # Hz
}

N_win_by_phase = 5

slow_icp_detection_compliance_features_params = {
    'detect_icp_params':detect_icp_params,
    'slow_icp_rise_detection_params':slow_icp_rise_detection_params,
    'heart_resp_in_icp_params':heart_resp_in_icp_params,
    'icp_pulse_resp_modulated':icp_pulse_resp_modulated_params,
    'abp_pulse_resp_modulated':abp_pulse_resp_modulated_params,
    'prx_params':prx_params,
    'abp_filter_params':abp_filter_params,
    'N_wins_by_phase':N_win_by_phase,
}


waveform_icp_window_params = {
    'slow_icp_detection_compliance_features_params':slow_icp_detection_compliance_features_params,
    'win_size_s' : 1.5,
    'n_samples' : 200,
    'random_sampling' : False,
}

### TOOLS
def load_table_nory(sub = None):
    nory_file = base_folder / 'documents_valentin' / 'metadata_nory_clean.xlsx'
    df_nory = pd.read_excel(nory_file) # load event file
    if not sub is None:
        if '_fin' in sub:
            sub = sub.split('_')[0]
        if sub in df_nory['ID_pseudo'].unique().tolist():
            df_nory = df_nory.set_index('ID_pseudo').loc[sub] # mask on patient dataframe
            return df_nory
        else:
            return None

def detect_slow_icp_rises(raw_icp,
                            icp_filtered, 
                            icp_smoothed_for_trough,
                            times,
                          datetimes,
                          datetimes_raw,
                          datetimes_smoothed,
                          params_detection = {'min_rise_duration_min':15,'min_rise_amplitude_mmHg':5, 'min_peak_amplitude_mmHg':20,'max_peak_amplitude_raw_mmHg':80,'min_decay_amplitude_mmHg':2, 'max_trough_amplitude_smoothed_mmHg':20},
                          ):
    """
    Inputs:
    - raw_icp : np.array = the raw ICP signal
    - icp_filtered : np.array = the ICP filtered signal
    - icp_smoothed_for_trough : np.array = an ICP filtered signal that fit better to raw trace to filter for trough amplitude
    - srate : float or int = sampling rate of the filtered ICP signal
    - datetimes : np.datetime64 = datetime vector of the filtered ICP signal
    - datetimes_raw = : np.datetime64 = datetime vector of the raw ICP signal
    - datetimes_smoothed = : np.datetime64 = datetime vector of the smoothed ICP signal
    - params_detection : dict = dictionnary of parameters aiming to post-process the detection by applying a mask
    - verbose : bool = print the highcut frequency of the filter

    Outputs :
    - detection : pd.DataFrame = detections with events in rows and features in columns
    """
    firt_derivarive_sig = np.gradient(icp_filtered)
    detection = detect_cross(firt_derivarive_sig, thresh = 0)
    detection.columns = ['trough_ind','peak_ind']
    detection['next_trough_ind'] = np.append(detection.loc[1:,'trough_ind'].values, np.nan)
    detection = detection.dropna()
    detection = detection.astype(int)
    detection['trough_time_s'] = times[detection['trough_ind']]
    detection['peak_time_s'] = times[detection['peak_ind']]
    detection['next_trough_time_s'] = times[detection['next_trough_ind']]
    detection['trough_date'] = datetimes[detection['trough_ind']]
    detection['peak_date'] = datetimes[detection['peak_ind']]
    detection['next_trough_date'] = datetimes[detection['next_trough_ind']]
    detection['rise_duration_s'] = detection['peak_time_s'] - detection['trough_time_s']
    detection['rise_duration_min'] = detection['rise_duration_s'] / 60
    detection['decay_duration_s'] = detection['next_trough_time_s'] - detection['peak_time_s']
    detection['decay_duration_min'] = detection['decay_duration_s'] / 60
    detection['trough_amplitude_mmHg'] = icp_filtered[detection['trough_ind']]
    detection['peak_amplitude_mmHg'] = icp_filtered[detection['peak_ind']]
    detection['next_trough_amplitude_mmHg'] = icp_filtered[detection['next_trough_ind']]
    detection['rise_amplitude_mmHg'] = detection['peak_amplitude_mmHg'] - detection['trough_amplitude_mmHg']
    raw_peak_inds = np.searchsorted(datetimes_raw, detection['peak_date'].values)
    detection['peak_amplitude_raw_mmHg'] = raw_icp[raw_peak_inds]
    smoothed_trough_inds = np.searchsorted(datetimes_smoothed, detection['trough_date'].values)
    detection['trough_amplitude_smoothed_mmHg'] = icp_smoothed_for_trough[smoothed_trough_inds]
    detection['decay_amplitude_mmHg'] = detection['peak_amplitude_mmHg'] - detection['next_trough_amplitude_mmHg']
    params_detection = {k:v for k,v in params_detection.items() if not v is None}
    if len(params_detection) != 0:
        masking = pd.DataFrame(index = detection.index, columns = params_detection.keys(), dtype = bool)
        masking[:] = True
        for param_name, threshold in params_detection.items():
            metric_name = param_name[4:]
            if 'min' in param_name:
                masking.loc[:,param_name] = detection[metric_name] > threshold
            elif 'max' in param_name:
                masking.loc[:,param_name] = detection[metric_name] < threshold
        mask = masking.all(axis = 1)
        detection = detection[mask].reset_index(drop = True)
    return detection

### JOBS

# ICP FILTER FOR DETECTION JOB
def icp_filter_for_detection(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    stream = cns_reader.streams['ICP']
    times = stream.get_times(as_second = True)
    srate = 1 / (np.median(np.diff(times)))
    raw_icp, datetimes = stream.get_data(apply_gain = True, with_times = True)
    datetimes = datetimes.astype('datetime64[ns]')
    params_filtering = p['params_filtering']
    period_min = params_filtering['half_period_filter_min'] * 2
    period_s = period_min * 60
    highcut = 1 / period_s
    icp_filtered = iirfilt(raw_icp, srate, highcut = highcut, ftype = params_filtering['ftype'], order = params_filtering['order'])
    icp_filtered = icp_filtered[::p['down_sampling_factor']]
    datetimes = datetimes[::p['down_sampling_factor']]
    times = times[::p['down_sampling_factor']]
    da = xr.DataArray(data = icp_filtered, dims = ['datetime'], coords = {'datetime':datetimes}, attrs = {'srate':srate/p['down_sampling_factor'], 'time':times})
    ds = xr.Dataset()
    ds['icp_filter_for_detection'] = da
    return ds

def test_icp_filter_for_detection(sub):
    print(sub)
    ds = icp_filter_for_detection(sub, **icp_filter_for_detection_params)
    print(ds['icp_filter_for_detection'])

icp_filter_for_detection_job = jobtools.Job(precomputedir, 'icp_filter_for_detection', icp_filter_for_detection_params, icp_filter_for_detection)
jobtools.register_job(icp_filter_for_detection_job)

# ICP FILTER FOR DETECTION JOB
def icp_filter_for_trough_filtering(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    stream = cns_reader.streams['ICP']
    times = stream.get_times(as_second = True)
    srate = 1 / (np.median(np.diff(times)))
    raw_icp, datetimes = stream.get_data(apply_gain = True, with_times = True)
    datetimes = datetimes.astype('datetime64[ns]')
    icp_filtered = iirfilt(raw_icp, srate, highcut = p['highcut'], ftype = p['ftype'], order = p['order'])
    icp_filtered = icp_filtered[::p['down_sampling_factor']]
    datetimes = datetimes[::p['down_sampling_factor']]
    times = times[::p['down_sampling_factor']]
    da = xr.DataArray(data = icp_filtered, dims = ['datetime'], coords = {'datetime':datetimes}, attrs = {'srate':srate/p['down_sampling_factor'], 'time':times})
    ds = xr.Dataset()
    ds['icp_filter_for_trough_filtering'] = da
    return ds

def test_icp_filter_for_trough_filtering(sub):
    print(sub)
    ds = icp_filter_for_trough_filtering(sub, **icp_filter_for_trough_filtering_params)
    print(ds['icp_filter_for_trough_filtering'])

icp_filter_for_trough_filtering_job = jobtools.Job(precomputedir, 'icp_filter_for_trough_filtering', icp_filter_for_trough_filtering_params, icp_filter_for_trough_filtering)
jobtools.register_job(icp_filter_for_trough_filtering_job)

# ABP FILTER FOR DETECTION JOB
def abp_filter(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    stream_names = list(cns_reader.streams)
    if 'ABP' in stream_names:
        stream = cns_reader.streams['ABP']
    elif 'ART' in stream_names:
        stream = cns_reader.streams['ART']
    times = stream.get_times(as_second = True)
    srate = 1 / (np.median(np.diff(times)))
    raw_abp, datetimes = stream.get_data(apply_gain = True, with_times = True)
    datetimes = datetimes.astype('datetime64[ns]')
    params_filtering = p['params_filtering']
    period_min = params_filtering['half_period_filter_min'] * 2
    period_s = period_min * 60
    highcut = 1 / period_s
    abp_filtered = iirfilt(raw_abp, srate, highcut = highcut, ftype = params_filtering['ftype'], order = params_filtering['order'])
    abp_filtered = abp_filtered[::p['down_sampling_factor']]
    datetimes = datetimes[::p['down_sampling_factor']]
    times = times[::p['down_sampling_factor']]
    da = xr.DataArray(data = abp_filtered, dims = ['datetime'], coords = {'datetime':datetimes}, attrs = {'srate':srate/p['down_sampling_factor'], 'time':times})
    ds = xr.Dataset()
    ds['abp_filter'] = da
    return ds

def test_abp_filter(sub):
    print(sub)
    ds = abp_filter(sub, **abp_filter_params)
    print(ds['abp_filter'])

abp_filter_job = jobtools.Job(precomputedir, 'abp_filter', abp_filter_params, abp_filter)
jobtools.register_job(abp_filter_job)

# SLOW ICP RISE DETECTION JOB
def slow_icp_rise_detection(sub, **p):
    cns_reader = CnsReader(data_path / sub)
    stream = cns_reader.streams['ICP']
    raw_icp, datetimes_raw = stream.get_data(apply_gain = True, with_times = True)
    icp_filtered = icp_filter_for_detection_job.get(sub)['icp_filter_for_detection']
    icp_smoothed_for_trough = icp_filter_for_trough_filtering_job.get(sub)['icp_filter_for_trough_filtering']
    datetimes = icp_filtered['datetime'].values
    times = icp_filtered.attrs['time']
    datetimes_smoothed = icp_smoothed_for_trough['datetime'].values
    icp_filtered = icp_filtered.values
    params_detection = p['params_detection']
    detections = detect_slow_icp_rises(raw_icp, icp_filtered, icp_smoothed_for_trough, times, datetimes, datetimes_raw, datetimes_smoothed, params_detection=params_detection)
    for col in detections.columns:
        if 'date' in col:
            detections[col] = detections[col].astype('datetime64[ns]')
    if detections.shape[0] == 0:
        print(f'Warning : No event detected in sub {sub}')
    return xr.Dataset(detections)

def test_slow_icp_rise_detection(sub):
    print(sub)
    ds = slow_icp_rise_detection(sub, **slow_icp_rise_detection_params)
    print(ds.to_dataframe())

slow_icp_rise_detection_job = jobtools.Job(precomputedir, 'slow_icp_rise_detection', slow_icp_rise_detection_params, slow_icp_rise_detection)
jobtools.register_job(slow_icp_rise_detection_job)


# DETECTION_FIG JOB
def detection_fig(sub, **p):
    # CSF things
    has_sub_csf = True if sub in get_csf_subs() else False
    if has_sub_csf:
        ds = icca_csf_job.get(sub)
        da = ds['Vol vidé (E_S)']
        csf_sampling_values = da.values
        csf_sampling_datetimes = da['datetime_Vol vidé (E_S)'].values
    else:
        csf_sampling_values = None
        csf_sampling_datetimes = None
        
    # Mobilisations
    ds = icca_clinical_job.get(sub)
    da = ds['Mobilisation']
    mobilisations_values = da.values
    mobilisations_datetimes = da['datetime_Mobilisation'].values

    icca_events_dict = {'csf_sampling':(csf_sampling_datetimes, csf_sampling_values),
                        'mobilisation':(mobilisations_datetimes, mobilisations_values),
                        }
    
    # PSE
    bug_subs_icca_pse = ['P1']
    if not sub in bug_subs_icca_pse:
        ds_pse = icca_pse_tt_job.get(sub)
    else:
        ds_pse = None

    # Load ICP data
    cns_reader = CnsReader(data_path / sub)
    stream = cns_reader.streams['ICP']
    times = stream.get_times(as_second = True)
    srate = 1 / (np.median(np.diff(times)))
    raw_icp, datetimes = stream.get_data(apply_gain = True, with_times = True)
    icp = iirfilt(raw_icp, srate, highcut = p['highcut_sig'], ftype = 'bessel', order = 3) # smooth
    icp_for_detection = icp_filter_for_detection_job.get(sub)['icp_filter_for_detection']
    icp_smoothed = icp_filter_for_trough_filtering_job.get(sub)['icp_filter_for_trough_filtering']
    datetimes_icp_for_detection = icp_for_detection['datetime'].values
    datetimes_icp_smoothed = icp_smoothed['datetime'].values

    # Load ICP rise detections
    detections = slow_icp_rise_detection_job.get(sub).to_dataframe()
    n_events = detections.shape[0]

    if n_events == 0:
        fig, ax = plt.subplots()
    else:
        fig, axs = plt.subplots(nrows = n_events, figsize = (8, n_events * 3), constrained_layout = True)
        for i in range(n_events):
            if n_events == 1:
                ax = axs
            else:
                ax = axs[i]
            t_start = detections.loc[i,'trough_date'] 
            t_start_sel = t_start - np.timedelta64(p['margin_mins'], 'm')
            t_stop = detections.loc[i,'next_trough_date']
            t_stop_sel = t_stop + np.timedelta64(p['margin_mins'], 'm')
            peak_date = detections.loc[i,'peak_date']
            rise_duration_mins = detections.loc[i,'rise_duration_min']
            decay_duration_mins = detections.loc[i,'decay_duration_min']
            mask = (datetimes > t_start_sel) & (datetimes < t_stop_sel)
            ax.plot(datetimes[mask], icp[mask], color = 'k')
            ax.axvspan(t_start, peak_date, color = 'r', alpha = 0.2, label = 'rise')
            ax.axvspan(peak_date, t_stop, color = 'g', alpha = 0.2, label = 'decay')
            ax.set_title(f'Event {i} - Rise Duration = {round(rise_duration_mins, 2)} mins ; Decay Duration = {round(decay_duration_mins, 2)} mins')
            ax.set_xlabel('Datetime')
            ax.set_ylabel('Intracranial Pressure (mmHg)')
            ax.legend(loc = 'upper left')

            ax_detection = ax.twinx()
            mask_icp_for_detection = (datetimes_icp_for_detection > t_start_sel) & (datetimes_icp_for_detection < t_stop_sel)
            ax_detection.plot(datetimes_icp_for_detection[mask_icp_for_detection], icp_for_detection.values[mask_icp_for_detection], color = 'b')
            ax_detection.set_yticks([])

            ax_smoothed = ax.twinx()
            mask_icp_smoothed = (datetimes_icp_smoothed > t_start_sel) & (datetimes_icp_smoothed < t_stop_sel)
            ax_smoothed.plot(datetimes_icp_smoothed[mask_icp_smoothed], icp_smoothed.values[mask_icp_smoothed], color = 'g', alpha = 0.8)
            ax_smoothed.set_yticks([])

            for icca_event_label, (event_datetimes, event_values) in icca_events_dict.items():
                if event_values is None:
                    continue
                mask_dates = (event_datetimes >= t_start_sel) & (event_datetimes < t_stop_sel)
                local_event_values = event_values[mask_dates]
                local_event_datetimes = event_datetimes[mask_dates]
                if local_event_values.size > 0:
                    for d, v in zip(local_event_datetimes, local_event_values):
                        color = 'y' if icca_event_label == 'csf_sampling' else 'r'
                        ax.axvline(d, color=color , alpha = 0.5)
                        ax.text(x = d, y = np.median(icp[mask]), s = v, rotation = 'vertical', va = 'center', color=color )

            if not ds_pse is None:
                ax2 = ax.twinx()
                counter_tt = 0
                for tt, c in zip(['Propofol','Thiopental','Midazolam'],['g','r','tab:blue']):
                    if not tt in list(ds_pse.data_vars):
                        continue
                    da = ds_pse[tt]
                    pse_values = da.values
                    pse_dates = da[f'datetime_{tt}'].values
                    mask_pse_dates = (pse_dates >= t_start_sel) & (pse_dates < t_stop_sel)
                    local_pse_values = pse_values[mask_pse_dates]
                    local_pse_dates = pse_dates[mask_pse_dates]
                    if local_pse_values.size > 0:
                        counter_tt += 1
                        ax2.scatter(local_pse_dates, local_pse_values, color = c, label = tt)
                if counter_tt != 0:
                    ax2.legend(loc = 'upper right')
    fig.suptitle(f'{sub} - DVE : {has_sub_csf}')
    fig.savefig(base_folder / 'figures' / 'slow_icp_rises_figs' / 'events' / f'{sub}.png' , dpi = 100, bbox_inches = 'tight')
    plt.close(fig)
    return xr.Dataset()

def test_detection_fig(sub):
    print(sub)
    ds = detection_fig(sub, **detection_fig_params)

detection_fig_job = jobtools.Job(precomputedir, 'detection_fig', detection_fig_params, detection_fig)
jobtools.register_job(detection_fig_job)

# LABEL DETECTION JOB 
"""
Predictors :
- Of the patient : gender, age, starting glasgow, ending glasgow, ending mRS, fisher, wfns, territoire lesion, lateralite lesion
- Of the rise / window : glasgow, propofol,	midazolam, thiopental, remifentanil, sufentanil, mobilisation, csf
Metrics of the rise : 
- qEEG : prct suppression healthy side, prct suppression injured side, delta power, alpha power, ADR
- Compliance : PSI, heart spectral peak size, resp spectral peak size, radio hr, P2P1 ratio
"""
def get_patient_metadata_eeg(sub):
    nory_meta = load_table_nory(sub)
    if not nory_meta is None:
        fisher = nory_meta['fisher']
        wfns = nory_meta['wfns']
        lesion_place = nory_meta['lesion_place']
        lesion_side = nory_meta['lesion_side']
        if lesion_side == 'gauche':
            lesion_side = 'left'
        elif lesion_side == 'droit':
            lesion_side = 'right'
        elif lesion_side == 'bilatéral':
            lesion_side = 'bilateral'
        else:
            lesion_side = None
    else:
        fisher = None
        wfns = None
        lesion_place = None
        lesion_side = None

    icu_meta = get_metadata(sub)
    overview = load_overview_data().set_index('Patient')

    try:
        icca_clinical = icca_clinical_job.get(sub)
        glasgow_beginning = icca_clinical['Glasgow, total'].values[0]
    except:
        glasgow_beginning = None

    meta = {'patient':sub, 
            'gender':'M' if icu_meta['sex'] == 'H' else 'F',
            'age':(icu_meta['entree_rea'] - icu_meta['ddn']).total_seconds() / (60 * 60 * 24 * 365),
            'stay_duration_days':overview.loc[sub, 'Duration_Mins'] / (24 * 60),
            'diagnosis':icu_meta['motif'], 
            'glasgow_beginning':glasgow_beginning, 
            'glasgow_end':icu_meta['GCS_sortie'],
            'mrs_end':icu_meta['mRS_sortie'],
            'fisher':fisher, 
            'wfns':wfns,
            'lesion_place':lesion_place,
            'lesion_side':lesion_side,
            }
    return meta

def get_patient_metadata_compliance(sub):
    icu_meta = get_metadata(sub)
    overview = load_overview_data().set_index('Patient')

    nory_meta = load_table_nory(sub)
    if not nory_meta is None:
        fisher = nory_meta['fisher']
        wfns = nory_meta['wfns']
        lesion_place = nory_meta['lesion_place']
        lesion_side = nory_meta['lesion_side']
        if lesion_side == 'gauche':
            lesion_side = 'left'
        elif lesion_side == 'droit':
            lesion_side = 'right'
        elif lesion_side == 'bilatéral':
            lesion_side = 'bilateral'
        else:
            lesion_side = None
    else:
        fisher = None
        wfns = None
        lesion_place = None
        lesion_side = None

    try:
        icca_clinical = icca_clinical_job.get(sub)
        glasgow_beginning = icca_clinical['Glasgow, total'].values[0]
    except:
        glasgow_beginning = None

    meta = {'patient':sub, 
            'gender':'M' if icu_meta['sex'] == 'H' else 'F',
            'age':(icu_meta['entree_rea'] - icu_meta['ddn']).total_seconds() / (60 * 60 * 24 * 365),
            'stay_duration_days':overview.loc[sub, 'Duration_Mins'] / (24 * 60),
            'diagnosis':icu_meta['motif'], 
            'glasgow_beginning':glasgow_beginning, 
            'glasgow_end':icu_meta['GCS_sortie'],
            'mrs_end':icu_meta['mRS_sortie'],
            'fisher':fisher, 
            'wfns':wfns,
            'lesion_place':lesion_place,
            'lesion_side':lesion_side,
            }
    return meta

def get_icca_treatment_info(ds, d1, d2, name_list = ['Norépinéphrine',
               'Midazolam', 'Propofol', 'Kétamine', 'Thiopental', 'Eskétamine', 
               'Sufentanil', 'Rémifentanil', 'Morphine']):
    d1 = np.datetime64(d1)
    d2 = np.datetime64(d2)
    if ds is None:
        available_names = []
    else:      
        available_names = list(ds.keys())
    not_available_names = [name for name in name_list if not name in available_names]
    res_dict = {}
    for name in name_list:
        name_dict = {}
        if name in not_available_names:
            # name_dict['value'] = np.nan
            name_dict['value'] = 0
            name_dict['unit'] = np.nan
        else:
            date_label = f'datetime_{name}'
            da = ds[name]
            unit = da.attrs['unit']
            all_values = da.values
            datetimes = da[date_label].values
            local_mask = (datetimes >= d1) & (datetimes < d2)
            if np.any(local_mask): # if we have an available dose during the window, compute average dose of the ones that are available
                local_values = np.nanmean(all_values[local_mask])
            else: # if no available dose during the window, take last dose
                last_date_ind = np.searchsorted(datetimes, d1) - 1
                last_date = datetimes[last_date_ind]
                n_mins_since_last_date = (d1 - last_date) / np.timedelta64(1 , 'm')
                if n_mins_since_last_date <= 120: # if last take is less than 2 hours ago -> keep dose
                    local_values = all_values[last_date_ind]
                    if last_date_ind >= all_values.size:
                        local_values = all_values[-1]
                    else:
                        local_values = all_values[last_date_ind]
                else: # elif last take higher than 2 hours ago -> no dose
                    # local_values = np.nan
                    local_values = 0
            name_dict['value'] = local_values
            name_dict['unit'] = unit
        res_dict[name] = name_dict
    return res_dict

def get_icca_clinical_info(ds, d1, d2, name_list = ['Dextro (mmol_l)', 'Mobilisation']):
    d1 = np.datetime64(d1)
    d2 = np.datetime64(d2)
    if ds is None:
        res_dict = {'Dextro (mmol_l)':{'value':np.nan , 'unit':np.nan},
                     'Mobilisation':{'value':np.nan , 'unit':np.nan}
        }
    else:
        available_names = list(ds.keys())
        not_available_names = [name for name in name_list if not name in available_names]
        res_dict = {}
        for name in name_list:
            name_dict = {}
            if name in not_available_names:
                name_dict['value'] = np.nan
                name_dict['unit'] = np.nan
            else:
                date_label = f'datetime_{name}'
                da = ds[name]
                unit = da.attrs['unit']
                all_values = da.values
                datetimes = da[date_label].values
                local_mask = (datetimes >= d1) & (datetimes < d2)
                if all_values.dtype != 'float64': # if qualitative variable ...
                    if np.any(local_mask):
                        local_values = 1
                    else:
                        local_values = 0
                else: # dextro
                    if np.any(local_mask): # if we have an available dose during the window, compute average dose of the ones that are available
                        local_values = np.nanmean(all_values[local_mask])
                    else: # if no available dose during the window, take last dose
                        next_date_ind = np.searchsorted(datetimes, d1)
                        last_date_ind = np.searchsorted(datetimes, d1) - 1
                        last_date = datetimes[last_date_ind]
                        next_date = datetimes[last_date_ind]
                        if (d1 - last_date) / np.timedelta64(1 , 'm') < (next_date - d1) / np.timedelta64(1 , 'm'): # if nearest is before d1
                            ind = last_date_ind
                        else:
                            ind = next_date_ind
                        if ind >= all_values.size:
                            local_values = all_values[-1]
                        else:
                            local_values = all_values[ind]
                name_dict['value'] = local_values
                name_dict['unit'] = unit
            res_dict[name] = name_dict
    return res_dict

def get_icca_csf_info(ds, d1, d2):
    if ds is None:
        return {'csf_sampling':{'value':np.nan, 'unit':np.nan}}
    else:
        d1 = np.datetime64(d1)
        d2 = np.datetime64(d2)
        name = 'Vol vidé (E_S)'
        date_label = f'datetime_{name}'
        da = ds[name]
        unit = da.attrs['unit']
        all_values = da.values
        datetimes = da[date_label].values
        local_mask = (datetimes >= d1) & (datetimes < d2)
        if np.any(local_mask): # if we have an available dose during the window, compute average dose of the ones that are available
            local_values = np.nansum(all_values[local_mask])
        else:
            local_values = np.nan
        return {'csf_sampling':{'value':local_values, 'unit':unit}}
    
def loc_compliance_metrics(ds_heart_resp_in_icp, ds_icp_pulse_resp_modulated, ds_abp_pulse_resp_modulated, ds_icp_resp_modulated, ds_abp_resp_modulated, ds_psi, ds_p2p1, icp_detections, abp_detections, ds_prx, ds_raq_icp, ds_raq_abp, d1, d2):
    da_heart_resp_in_icp = ds_heart_resp_in_icp['heart_resp_in_icp']
    dict_res = {}
    for metric in ['heart_in_icp_spectrum','resp_in_icp_spectrum','ratio_heart_resp_in_icp_spectrum']:
        dict_res[metric] = float(da_heart_resp_in_icp.loc[metric,d1:d2].median())
    da_icp_pulse_resp_modulated = ds_icp_pulse_resp_modulated['icp_pulse_resp_modulated']
    dict_res['icp_pulse_resp_modulated'] = float(da_icp_pulse_resp_modulated.loc[d1:d2].median())
    try:
        da_abp_pulse_resp_modulated = ds_abp_pulse_resp_modulated['abp_pulse_resp_modulated']
        dict_res['abp_pulse_resp_modulated'] = float(da_abp_pulse_resp_modulated.loc[d1:d2].median())
    except:
        dict_res['abp_pulse_resp_modulated'] = np.nan
    try:
        da_icp_resp_modulated = ds_icp_resp_modulated['icp_resp_modulated']
        dict_res['icp_resp_modulated'] = float(da_icp_resp_modulated.loc[d1:d2].median())
    except:
        dict_res['icp_resp_modulated'] = np.nan
    try:
        da_abp_resp_modulated = ds_abp_resp_modulated['abp_resp_modulated']
        dict_res['abp_resp_modulated'] = float(da_abp_resp_modulated.loc[d1:d2].median())
    except:
        dict_res['abp_resp_modulated'] = np.nan
    try:
        dict_res['RAQ_2'] = float(ds_raq_icp['raq_icp'].loc[d1:d2].median())
    except:
        dict_res['RAQ_2'] = np.nan
    try:
        dict_res['RAQ_ABP'] = float(ds_raq_abp['raq_abp'].loc[d1:d2].median())
    except:
        dict_res['RAQ_ABP'] = np.nan
    dict_res['P2P1_ratio'] = float(ds_p2p1['ratio_P1P2'].loc[d1:d2].median())
    local_psi = float(ds_psi['psi'].loc[d1:d2].mean())
    dict_res['PSI'] = local_psi if local_psi != 0 else np.nan
    dict_res['icp_pulse_amplitude_mmHg'] = float(icp_detections[(icp_detections['trough_date'] > d1) & (icp_detections['trough_date'] < d2)]['rise_amplitude'].median())
    dict_res['abp_pulse_amplitude_mmHg'] = float(abp_detections[(abp_detections['trough_date'] > d1) & (abp_detections['trough_date'] < d2)]['rise_amplitude'].median())
    try:
        da_prx = ds_prx['prx']
        dict_res['PRx'] = float(da_prx.loc[d1:d2].median())
    except:
        dict_res['PRx'] = np.nan
    return dict_res

def loc_physio_metrics(ecg_peaks, resp_cycles, d1, d2):
    dict_res = {'heart_rate_bpm':np.nan, 'resp_rate_cpm':np.nan}
    if not ecg_peaks is None:
        local_ecg_peaks = ecg_peaks[(ecg_peaks['peak_date'] > d1) & (ecg_peaks['peak_date'] < d2)]
        dict_res['heart_rate_bpm'] = np.nanmedian(60 / np.diff(local_ecg_peaks['peak_time']))
    if not resp_cycles is None:
        local_resp_cycles = resp_cycles[(resp_cycles['inspi_date'] > d1) & (resp_cycles['expi_date'] < d2)]
        dict_res['resp_rate_cpm'] = np.nanmedian(local_resp_cycles['cycle_freq'] * 60)
    return dict_res

def define_windows(start_date, peak_date, stop_date, N_wins_by_phase):
    start_date = np.datetime64(start_date)
    peak_date = np.datetime64(peak_date)
    stop_date = np.datetime64(stop_date)

    duration_rise_mins = (peak_date - start_date) / np.timedelta64(1, 'm')
    duration_rise_wins_mins = duration_rise_mins / N_wins_by_phase
    duration_rise_wins_mins = int(round(duration_rise_wins_mins, 0))

    duration_decay_mins = (stop_date - peak_date) / np.timedelta64(1, 'm')
    duration_decay_wins_mins = duration_decay_mins / N_wins_by_phase
    duration_decay_wins_mins = int(round(duration_decay_wins_mins, 0))

    start_rise_win_dates = np.arange(start_date, peak_date, np.timedelta64(duration_rise_wins_mins, 'm'))
    if start_rise_win_dates.size == N_wins_by_phase + 1:
        start_rise_win_dates = start_rise_win_dates[:-1]
    stop_rise_win_dates = start_rise_win_dates + np.timedelta64(duration_rise_wins_mins, 'm')
    rise_win_dates = [(start,stop) for start,stop in zip(start_rise_win_dates, stop_rise_win_dates)]
    rise_win_labels = [f'rise_{i+1}' for i in range(start_rise_win_dates.size)]
    dict_rise_wins = {label:dates for label,dates in zip(rise_win_labels,rise_win_dates)}

    start_decay_win_dates = np.arange(peak_date, stop_date, np.timedelta64(duration_decay_wins_mins, 'm'))
    if start_decay_win_dates.size == N_wins_by_phase + 1:
        start_decay_win_dates = start_decay_win_dates[:-1]
    stop_decay_win_dates = start_decay_win_dates + np.timedelta64(duration_decay_wins_mins, 'm')
    decay_win_dates = [(start,stop) for start,stop in zip(start_decay_win_dates, stop_decay_win_dates)]
    decay_win_labels = [f'decay_{i+1}' for i in range(start_decay_win_dates.size)]
    dict_decay_wins = {label:dates for label,dates in zip(decay_win_labels,decay_win_dates)}

    dict_baseline_before = {'baseline_before':(start_date - np.timedelta64(duration_rise_wins_mins, 'm'), start_date)}
    dict_baseline_after = {'baseline_after':(stop_date, stop_date + np.timedelta64(duration_decay_wins_mins, 'm'))}
    
    dict_wins = dict_baseline_before | dict_rise_wins | dict_decay_wins | dict_baseline_after
    return dict_wins 

def slow_icp_detection_compliance_features(sub, **p):
    meta_sub = get_patient_metadata_compliance(sub)
    # CSF things
    has_sub_csf = True if sub in get_csf_subs() else False
    if has_sub_csf:
        ds_csf = icca_csf_job.get(sub)
    else:
        ds_csf = None

    # Mobilisations
    try:
        ds_clinical = icca_clinical_job.get(sub)
    except:
        ds_clinical = None
    # PSE
    bug_subs_icca_pse = ['P1','HA1','P73','P86']
    if not sub in bug_subs_icca_pse:
        ds_pse = icca_pse_tt_job.get(sub)
    else:
        ds_pse = None

    try:
        ecg_peaks = detect_ecg_job.get(sub).to_dataframe()
    except:
        ecg_peaks = None
    
    try:
        resp_cycles = detect_resp_job.get(sub).to_dataframe()
    except:
        resp_cycles = None
    
    # Get ICP filtered like during slow icp detections
    icp_filtered = icp_filter_for_detection_job.get(sub)['icp_filter_for_detection']
    datetimes_icp_filtered = icp_filtered['datetime'].values
    icp_filtered = icp_filtered.values

    # Get ABP filtered like during ICP filtered
    abp_filtered = abp_filter_job.get(sub)['abp_filter']
    datetimes_abp_filtered = abp_filtered['datetime'].values
    abp_filtered = abp_filtered.values

    # Whole compliance time series ds
    ds_heart_resp_in_icp = heart_resp_in_icp_job.get(sub)
    ds_icp_pulse_resp_modulated = icp_pulse_resp_modulated_job.get(sub)
    ds_abp_pulse_resp_modulated = abp_pulse_resp_modulated_job.get(sub)
    ds_raq_icp = raq_icp_job.get(sub)
    ds_raq_abp = raq_abp_job.get(sub)
    ds_icp_resp_modulated = icp_resp_modulated_job.get(sub)
    ds_abp_resp_modulated = abp_resp_modulated_job.get(sub)
    ds_psi = psi_job.get(sub)
    ds_p2p1 = ratio_P1P2_job.get(sub)
    icp_detections = detect_icp_job.get(sub).to_dataframe()
    abp_detections = detect_abp_job.get(sub).to_dataframe()

    ds_prx = prx_job.get(sub)

    # Load ICP rise detections
    detections = slow_icp_rise_detection_job.get(sub).to_dataframe()
    remove_features = ['trough_ind', 'peak_ind', 'next_trough_ind', 'trough_time_s','peak_time_s', 'next_trough_time_s']
    detections_features_sel = [col for col in detections.columns if not col in remove_features]
    n_events = detections.shape[0]
    
    if n_events != 0:
        rows = []
        for i in range(n_events):
            start_date = detections.loc[i,'trough_date']
            peak_date = detections.loc[i,'peak_date']
            stop_date = detections.loc[i,'next_trough_date']
            dict_wins = define_windows(start_date, peak_date, stop_date, p['N_wins_by_phase'])
            event_features = detections.loc[i,detections_features_sel].to_dict()
            for win_label, (d1, d2) in dict_wins.items():
                win_duration_mins = (d2-d1) / np.timedelta64(1, 'm')
                csf_win = {k:v['value'] for k,v in get_icca_csf_info(ds_csf, d1, d2).items()}
                clinical_win = {k:v['value'] for k,v in get_icca_clinical_info(ds_clinical, d1, d2).items()}
                pse_win = {k:v['value'] for k,v in get_icca_treatment_info(ds_pse, d1, d2).items()}
                med_icp_win = np.nanmedian(icp_filtered[(datetimes_icp_filtered > d1) & (datetimes_icp_filtered < d2)])
                med_abp_win = np.nanmedian(abp_filtered[(datetimes_abp_filtered > d1) & (datetimes_abp_filtered < d2)])
                med_cpp_win = med_abp_win - med_icp_win
                icp_abp_win = {'median_icp_mmHg':med_icp_win, 'median_abp_mmHg':med_abp_win, 'median_cpp_mmHg':med_cpp_win}
                compliance_win = loc_compliance_metrics(ds_heart_resp_in_icp, ds_icp_pulse_resp_modulated, ds_abp_pulse_resp_modulated, ds_icp_resp_modulated, ds_abp_resp_modulated, ds_psi, ds_p2p1, icp_detections, abp_detections, ds_prx, ds_raq_icp, ds_raq_abp, d1,d2)
                physio_win = loc_physio_metrics(ecg_peaks, resp_cycles, d1, d2)
                row_dict = meta_sub | {'n_event':i} | {'n_event_sub':f'{sub}_{i}'} | event_features | {'win_label':win_label} | {'start_win_date':d1, 'stop_win_date':d2} |{'win_duration_mins':win_duration_mins} | pse_win | clinical_win | csf_win | compliance_win | icp_abp_win | physio_win
                row = pd.DataFrame.from_dict(row_dict, orient = 'index').T
                rows.append(row)
        res = pd.concat(rows)
        res = res.reset_index(drop = True)
        for col in res.columns:
            if not 'date' in col:
                try:
                    res[col] = res[col].astype('float64')
                except:
                    res[col] = res[col]
    else:
        res = pd.DataFrame()

    return xr.Dataset(res)

def test_slow_icp_detection_compliance_features(sub):
    print(sub)
    ds = slow_icp_detection_compliance_features(sub, **slow_icp_detection_compliance_features_params)
    print(ds)
    df = ds.to_dataframe()
    print(df.dtypes)
    print(df)

slow_icp_detection_compliance_features_job = jobtools.Job(precomputedir, 'slow_icp_detection_compliance_features', slow_icp_detection_compliance_features_params, slow_icp_detection_compliance_features)
jobtools.register_job(slow_icp_detection_compliance_features_job)

# WAVEFORM ICP
def waveform_icp_window(sub, **p):

    win_size_s = p['win_size_s']
    n_samples = p['n_samples']
    random_sampling = p['random_sampling']

    cns_reader = pycns.CnsReader(data_path / sub)
    icp_stream = cns_reader.streams['ICP']
    raw_icp, datetimes = icp_stream.get_data(with_times = True, apply_gain = True)
    srate = icp_stream.sample_rate
    window_size_ind = int(srate * win_size_s)
    half_window_size_ind = window_size_ind // 2

    compliance_df = slow_icp_detection_compliance_features_job.get(sub).to_dataframe()
    event_nums = compliance_df['n_event'].unique().tolist()
    # n_event_sample = int(((compliance_df['win_duration_mins'].min()) * 60) - 100)
    n_event_sample = n_samples
    icp_detections = detect_icp_job.get(sub).to_dataframe()

    win_labels = compliance_df['win_label'].unique().tolist()

    dims = ['event','win_label','estimator','time']
    data_init = np.zeros((len(event_nums), len(win_labels), 2, window_size_ind))
    t_vector = range(window_size_ind)/srate
    t_vector = t_vector - np.max(t_vector) / 2
    coords = {'event':event_nums, 'win_label':win_labels, 'estimator':['m','s'], 'time':t_vector}
    waveforms = xr.DataArray(data = data_init, dims = dims, coords = coords)

    for ev in event_nums:
        compliance_df_ev = compliance_df[compliance_df['n_event'] == ev].set_index('win_label')
        for win_label in win_labels:
            d1 = compliance_df_ev.loc[win_label, 'start_win_date']
            d2 = compliance_df_ev.loc[win_label, 'stop_win_date']
            local_icp_detections = icp_detections[(icp_detections['peak_date'] > d1) & (icp_detections['peak_date'] < d2)].reset_index(drop = True)
            if local_icp_detections.shape[0] < n_samples:
                continue
            if random_sampling:
                local_icp_detections = local_icp_detections.sample(n_event_sample).reset_index(drop = True)
            else:
                # local_icp_detections = local_icp_detections.loc[:n_event_sample,:].reset_index(drop = True)
                local_icp_detections = local_icp_detections.iloc[-n_event_sample:,:].reset_index(drop = True)
            epochs = np.zeros((local_icp_detections.shape[0], window_size_ind))
            for i, row in local_icp_detections.iterrows():
                peak_ind = row['peak_ind']
                start_ind = peak_ind - half_window_size_ind
                stop_ind = start_ind + window_size_ind
                icp_epoch = raw_icp[start_ind:stop_ind]
                epochs[i,:] = icp_epoch
            m = np.mean(epochs, axis = 0)
            s = np.std(epochs, axis = 0)
            waveforms.loc[ev, win_label,'m',:] = m
            waveforms.loc[ev, win_label,'s',:] = s
    waveforms.attrs['n_samples'] = n_event_sample

    t = waveforms['time'].values
    min_, max_ = waveforms.min(), waveforms.max()

    nrows = waveforms['event'].size
    ncols = waveforms['win_label'].size
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (3 * ncols, nrows * 4), sharey = True)
    n_samples = waveforms.attrs['n_samples']
    if random_sampling:
        suptitle = f'N pulses randomly sampled : {n_samples}'
    else:
        suptitle = f'N pulses sampled (last of the window) : {n_samples}'
    fig.suptitle(suptitle, y = 1.01)
    fig.subplots_adjust(wspace = 0)
    for r, ev in enumerate(waveforms['event'].values):
        for c, win_label in enumerate(waveforms['win_label'].values):
            ax = axs[r,c]
            m = waveforms.loc[ev, win_label,'m',:].values
            s = waveforms.loc[ev, win_label,'s',:].values
            ax.plot(t, m, color = 'k')
            ax.fill_between(t, m-s, m+s, color = 'k', alpha = 0.2)
            ax.set_title(win_label)
            # ax.set_ylim(5 , 40)
            ax.set_ylim(min_ - min_ / 10, max_ + max_ / 10)
            if c == 0:
                ax.set_ylabel(f'Ev n°{int(ev)}\nICP (mmHg)')
    fig.savefig(base_folder / 'figures' / 'slow_icp_rises_figs' / 'waveform_icp_window' / f'{sub}.png', dpi = 300, bbox_inches = 'tight')
    plt.close(fig)
    return xr.Dataset()

def test_waveform_icp_window(sub):
    print(sub)
    ds = waveform_icp_window(sub, **waveform_icp_window_params)

waveform_icp_window_job = jobtools.Job(precomputedir, 'waveform_icp_window', waveform_icp_window_params, waveform_icp_window)
jobtools.register_job(waveform_icp_window_job)

### FUNCTIONS FOR MANUSCRIPT

def included_population_description(verbose = True, save = False):
    load_patients = get_compliance_subs()
    print(len(load_patients))
    concat = []
    for sub in load_patients:
        meta_sub = get_patient_metadata_compliance(sub)
        concat.append(pd.DataFrame.from_dict(meta_sub, orient = 'index').T)
    df = pd.concat(concat)
    df = df[df['patient'].apply(lambda x:False if 'fin' in x else True)]
    df = df.iloc[:,:8]
    df.columns = ['Patient ID','Gender','Age','Hospital Stay Duration','Reason For Hospitalization','Initial GCS','Final GCS','Final mRS']
    print(df['Reason For Hospitalization'].unique())
    # df['Reason For Hospitalization'] = df['Reason For Hospitalization'].map({'HSA':'SAH','TC':'TBI','autre':'Other', 'autre ':'Other'})
    N = df.shape[0]
    round_ = 2
    text = """Study included {n} patients from 2020 to 2024 ({n_male} males, {n_female} females) aged of {m_age} {s_age} years old.
    They lasted for {m_stay} {s_stay} days in the ICU where they were hospitalized mainly for SAH ({prct_hsa}%), or TBI ({prct_TC}%), or other reasons ({prct_other}).
    Their initial GCS was of {m_gcs_initial} {s_gcs_initial} while their final was of {m_gcs_final} {s_gcs_final}, with an mRS score of {m_mrs} {s_mrs}.
    """.format(n = N, 
               n_male= df['Gender'].value_counts().round(round_)['M'],
               n_female= df['Gender'].value_counts().round(round_)['F'],
               m_age = df['Age'].median().round(round_),
               s_age = iqr_interval(df['Age'].dropna().values, round_),
               m_stay = df['Hospital Stay Duration'].median().round(round_),
               s_stay = iqr_interval(df['Hospital Stay Duration'].dropna().values, round_),
               prct_hsa = round(df['Reason For Hospitalization'].value_counts(normalize = True)['HSA'] * 100, round_),
               prct_TC = round(df['Reason For Hospitalization'].value_counts(normalize = True)['TC'] * 100, round_),
               prct_other = round(df['Reason For Hospitalization'].value_counts(normalize = True)['autre'] * 100, round_),
               m_gcs_initial = df['Initial GCS'].median().round(round_),
               s_gcs_initial = iqr_interval(df['Initial GCS'].dropna().values, round_),
               m_gcs_final = df['Final GCS'].median().round(round_),
               s_gcs_final = iqr_interval(df['Final GCS'].dropna().values, round_),
               m_mrs = df['Final mRS'].median().round(round_),
               s_mrs = iqr_interval(df['Final mRS'].dropna().values, round_),             
               )
    if verbose:
        print(text)
    df_return = df.set_index('Patient ID')
    df_return.insert(0, 'Patient', [f'P{i+1}' for i in range(N)])
    if save:
        df_return.round(round_).to_excel(base_folder / 'figures' / 'slow_icp_rises_figs' / 'manuscript_compliance' / 'included_population_description.xlsx')
    return df_return

def signal_description(verbose = True, save = False):
    round_ = 2
    load_patients = get_compliance_subs()
    dict_durations = {}
    streams = ['ICP','CO2','ABP','ECG_II']
    rows = []
    for sub in load_patients:
        durations = get_patient_durations_by_stream_job.get(sub).to_dataframe().set_index('stream')
        possible_streams = durations.index.tolist()
        row = {'patient':sub}
        for stream in streams:
            if not stream == 'ABP':
                row[stream] = durations.loc[stream,'duration_days']
            else:
                if 'ABP' in possible_streams and 'ART' in possible_streams:
                    row[stream] = max([durations.loc['ABP','duration_days'], durations.loc['ART','duration_days']])
                elif 'ABP' in possible_streams and not 'ART' in possible_streams:
                    row[stream] = durations.loc['ABP','duration_days']
                elif not 'ABP' in possible_streams and 'ART' in possible_streams:
                    row[stream] = durations.loc['ART','duration_days']
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df['patient'].apply(lambda x:False if 'fin' in x else True)].set_index('patient')
    d_str_format = {}
    for col in df.columns:
        d_str_format[col] = med_iqr(df[col].values, round_=round_)
    N = df.shape[0]
    text = "Considering the {N} patients, ICP recordings lasted for {ICP} days, CO2 recordings lasted for {CO2} days, ABP recordings lasted for {ABP} days, ECG recordings lasted for {ECG_II} days, ".format(N=N, ICP = d_str_format['ICP'], CO2 = d_str_format['CO2'], ABP = d_str_format['ABP'], ECG_II = d_str_format['ECG_II'])
    if verbose:
        print(text)
    df.insert(0, 'Patient', [f'P{i+1}' for i in range(N)])
    if save:
        df.round(round_).to_excel(base_folder / 'figures' / 'slow_icp_rises_figs' / 'manuscript_compliance' / 'signal_description.xlsx')
    return df

def slow_icp_rise_description():
    round_ = 2
    detections_slow_icp_rises = []
    for s in get_compliance_files():
        df_sub = slow_icp_rise_detection_job.get(s).to_dataframe()
        df_sub['patient'] = s
        df_sub['total_duration_min'] = df_sub['rise_duration_min'] + df_sub['decay_duration_min']
        df_sub['rise_amplitude_mmHg'] = df_sub['peak_amplitude_mmHg'] - df_sub['trough_amplitude_smoothed_mmHg']
        detections_slow_icp_rises.append(df_sub)
    detections_slow_icp_rises = pd.concat(detections_slow_icp_rises)
    n_by_patient = detections_slow_icp_rises['patient'].value_counts().values

    d_format =  dict(N_slow_icp_rises=detections_slow_icp_rises.shape[0],
                     N_patients_with_detection = n_by_patient.size,    
                     N_initial = 66,
                     n_events=med_iqr(n_by_patient, round_=round_),
    )
    metrics = ['total_duration_min',
               'rise_duration_min','trough_amplitude_smoothed_mmHg','peak_amplitude_mmHg','rise_amplitude_mmHg',
               'decay_duration_min','decay_amplitude_mmHg','next_trough_amplitude_mmHg'
               ]
    
    for col in metrics:
        d_format[col] = med_iqr(detections_slow_icp_rises[col].values, round_=round_)
    
    text = """Automated pipeline allowed for the detection of {N_slow_icp_rises} slow spontaneous ICP rise events. At least one event was detected in {N_patients_with_detection} from the {N_initial} initially included patients. 
    The number of events detected by patient was {n_events}. The total duration of these events was {total_duration_min} minutes. 
    This period was decomposed in a rising phase of {rise_duration_min} minutes that increased the ICP of {rise_amplitude_mmHg} mmHg from {trough_amplitude_smoothed_mmHg} to {peak_amplitude_mmHg} mmHg, 
    followed by a decaying phase of {decay_duration_min} minutes that decreased the ICP of {decay_amplitude_mmHg} to a value of {next_trough_amplitude_mmHg} mmHg.""".format(**d_format)

    print(text)


def windows_description():
    round_ = 2
    df = concat_slow_icp_features('compliance')
    d_format = dict(
        duration_window_rise=med_iqr(df[df['win_label'] == 'rise_1']['win_duration_mins'].values, round_=round_),
        duration_window_decay=med_iqr(df[df['win_label'] == 'decay_1']['win_duration_mins'].values, round_=round_),
    )
    text = """For analysis, rising and decaying phases were divided into 5 equal duration windows (+1 adjacent baseline window)
      of {duration_window_rise} and {duration_window_decay} minutes, respectively.""".format(**d_format)

    print(text)
    

### RUN / CONCAT
def get_nory_keys():
    df_nory = load_table_nory() # load event file
    return list(df_nory['ID_pseudo'].unique())

def get_compliance_files(n_hours_min = 10, remove_subs = ['P9','P19']):
    n_mins = 60*n_hours_min
    subs_abp = [s for s in get_patient_list(['ICP','CO2','ABP'], threshold_duration_mins=n_mins)]
    subs_art = [s for s in get_patient_list(['ICP','CO2','ART'], threshold_duration_mins=n_mins)]
    subs = list(set(subs_abp + subs_art))
    subs =  [s for s in subs if not s in remove_subs]
    return subs

def get_compliance_subs(n_hours_min = 10, remove_subs = ['P9','P19']):
    n_mins = 60*n_hours_min
    subs_abp = [s for s in get_patient_list(['ICP','CO2','ABP'], threshold_duration_mins=n_mins) if not 'fin' in s]
    subs_art = [s for s in get_patient_list(['ICP','CO2','ART'], threshold_duration_mins=n_mins) if not 'fin' in s]
    subs = list(set(subs_abp + subs_art))
    subs =  [s for s in subs if not s in remove_subs]
    return subs

def working_subs():
    nory_subs = get_nory_keys()
    possible_subs = get_patient_list(['ICP','Scalp','CO2'])
    icca_subs = get_icca_subs()
    keep_subs = [s for s in nory_subs if s in possible_subs]
    keep_icca_subs = [s for s in keep_subs if s in icca_subs]
    # remove_subs = ['P14','LA19','PL20','P69','P70','P71'] # no ratioP2P1 precomputed,
    remove_subs = []
    return_subs = [s for s in keep_icca_subs if not s in remove_subs]
    # return_subs = ['MF12','NN7']
    return return_subs

def concat_slow_icp_features(res_mode, save = True):
    if res_mode == 'compliance':
        load_subs = get_compliance_files()
        job = slow_icp_detection_compliance_features_job
    concat = []
    for sub in load_subs:
        try:
            df_sub = job.get(sub).to_dataframe()
        except:
            print(sub, 'not loaded')
            continue
        concat.append(df_sub)
    concat = pd.concat(concat).reset_index(drop = True)
    if 'eeg' in res_mode:
        concat['is_side_of_lesion'] = concat['is_side_of_lesion'].map({1:'injured',0.5:'midline',0:'healthy'})
    if save:
        path = base_folder / 'figures' / 'slow_icp_rises_figs' / 'res_matrix' / f'{res_mode}.xlsx'
        concat.to_excel(path)
    return concat

def compute_all():
    n_hours_min = 10
    # Subs without detection = P1, P16, WJ14
    # jobtools.compute_job_list(icp_filter_for_trough_filtering_job, [(s,) for s in get_patient_list(['ICP'], threshold_duration_mins=60*n_hours_min)], force_recompute=False, engine='loop')
    # jobtools.compute_job_list(icp_filter_for_detection_job, [(s,) for s in get_patient_list(['ICP'], threshold_duration_mins=60*n_hours_min)], force_recompute=False, engine='loop')
    # jobtools.compute_job_list(abp_filter_job, [(s,) for s in list(set(get_patient_list(['ART'], threshold_duration_mins=60*n_hours_min) + get_patient_list(['ABP'], threshold_duration_mins=60*n_hours_min)))], force_recompute=False, engine='loop')
    # jobtools.compute_job_list(slow_icp_rise_detection_job, [(s,) for s in get_patient_list(['ICP'], threshold_duration_mins=60*n_hours_min)], force_recompute=False, engine='loop')
    # remove_subs = ['P86','P8_fin','P19','P73','P96','P87_fin','P9','P18_fin']
    remove_subs = ['P19','P9'] # P19 and P9 no metadata, P95 CO2 is flat
    # jobtools.compute_job_list(detection_fig_job, [(s,) for s in get_patient_list(['ICP'], threshold_duration_mins=60*n_hours_min) if not s in remove_subs], force_recompute=True, engine='loop')
    # jobtools.compute_job_list(detection_fig_job, [(s,) for s in get_patient_list(['ICP'], threshold_duration_mins=60*n_hours_min) if not s in remove_subs], force_recompute=True, engine='joblib', n_jobs = 5)

    jobtools.compute_job_list(slow_icp_detection_compliance_features_job, [(s,) for s in get_compliance_files()], force_recompute=True, engine='loop')

    # jobtools.compute_job_list(waveform_icp_window_job, [(s,) for s in get_compliance_files()], force_recompute=True, engine='joblib', n_jobs = 5)

if __name__ == "__main__":
    # print(working_subs())
    # test_icp_filter_for_detection('MF12')
    # test_icp_filter_for_trough_filtering('MF12')
    # test_abp_filter('MF12')
    # test_slow_icp_rise_detection('P4')
    # test_detection_fig('MF12')
    # test_slow_icp_detection_compliance_features('MF12') # P18_fin
    # test_waveform_icp_window('MF12') # P85, P41, P37, P4, P16, P86, P74, P65, HA1, P73, LD16, P75, WJ14, BJ11, P13, GA9

    compute_all()

    # print(concat_slow_icp_features('compliance', save = True)['patient'].unique().size)

    # MANUSCRIPT
    # included_population_description(save = False)
    # signal_description(save = False)
    # slow_icp_rise_description()
    # windows_description()

    # print(len(get_compliance_files()))
    # print(len(get_compliance_subs()))

