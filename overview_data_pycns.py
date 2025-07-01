import pycns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import jobtools
import pandas as pd
from configuration import *
from tqdm import tqdm
import xarray as xr

def get_patients_list_raw(verbose = False):
    patients = [str(x).split('/')[-1] for x in data_path.iterdir() if x.is_dir()]
    if verbose:
        print(patients)
    return patients

def get_number_of_SD_from_events(subs):
    dict_n_sds = {}
    for sub in subs:
        event_file = data_path / sub / 'Events.xml'
        if event_file.is_file():
            evs = pd.DataFrame(pycns.read_events_xml(event_file, time_zone='Europe/Paris'))
            n_sd = evs['name'].apply(lambda x:1 if 'SD' in x else 0).sum()
        else:
            n_sd = 0
        dict_n_sds[sub] = n_sd
    dict_n_sds
    return dict_n_sds
    

def overview_streams(save):
    from tools import get_patient_dates

    subs = get_patients_list_raw()

    meta_df = pd.read_excel(base_data / 'liste_monito_multi_05_02_25.xlsx').set_index('ID_pseudo')
    cols_meta = ['motif','GCS_sortie','mRS_sortie','mRS_1mois','mRS_3mois','mRS_6mois']

    dict_N_SD = get_number_of_SD_from_events(subs)

    overview_cols = ['pycns_readable','CO2','ECG_II','ICP','ICP_Mean','ABP','ABP_Mean','ART','ART_Mean','EEG','Scalp','N_Scalp_Chans','ECoG','N_ECoG_Chans','Strip_or_Depth','Start_Date','Stop_Date','Duration_Mins','N_SDs']
    overview_cols = overview_cols + cols_meta
    overview_df = pd.DataFrame(index = subs, columns = overview_cols)

    loop_subs = subs
    for sub in tqdm(loop_subs):
        # print(sub)
        try:
            reader = pycns.CnsReader(data_path / sub)
        except:
            print(sub, 'not readable by pycns')
            overview_df.loc[sub, 'pycns_readable'] = 0
        else:
            stream_list = list(reader.streams.keys())

            start,stop = get_patient_dates(sub)

            if start is None or stop is None:
                duration_mins = np.nan
            else:
                duration_us = np.timedelta64(stop - start)
                duration_mins = round(duration_us.astype('timedelta64[m]').astype(float), 2)

            for col in overview_cols:
                if col == 'pycns_readable':
                    overview_df.loc[sub, col] = 1
                elif col == 'Duration_Mins':
                    overview_df.loc[sub, col] = duration_mins
                elif col in ['CO2','ECG_II','ICP','ICP_Mean','ABP','ABP_Mean','ART','ART_Mean','EEG']:
                    overview_df.loc[sub, col] = 1 if col in stream_list else 0
                elif col == 'Scalp' and 'EEG' in stream_list:
                    ch_names = reader.streams['EEG'].channel_names
                    scalp_chans = [ch for ch in ch_names if not 'ECoG' in ch]
                    ecog_chans = [ch for ch in ch_names if 'ECoG' in ch]
                    if not len(scalp_chans) == 0:
                        overview_df.loc[sub, 'Scalp'] = 1
                        overview_df.loc[sub, 'N_Scalp_Chans'] = len(scalp_chans)
                    else:
                        overview_df.loc[sub, 'Scalp'] = 0
                    if not len(ecog_chans) == 0:
                        overview_df.loc[sub, 'ECoG'] = 1
                        overview_df.loc[sub, 'N_ECoG_Chans'] = len(ecog_chans)
                        if len(ecog_chans) > 6:
                            overview_df.loc[sub, 'Strip_or_Depth'] = 'depth'
                        else:
                            overview_df.loc[sub, 'Strip_or_Depth'] = 'strip'
                    else:
                        overview_df.loc[sub, 'ECoG'] = 0
                elif col == 'Start_Date':
                    overview_df.loc[sub, 'Start_Date'] = start
                elif col == 'Stop_Date':
                    overview_df.loc[sub, 'Stop_Date'] = stop
                elif col in cols_meta:
                    if sub in meta_df.index:
                        overview_df.loc[sub, col] = meta_df.loc[sub, col]
                    else:
                        overview_df.loc[sub, col] = None
        
        overview_df.loc[sub, 'N_SDs'] = dict_N_SD[sub]
    overview_df.index.name = 'Patient'
    print(overview_df)
    overview_df = overview_df.sort_values(by = 'Start_Date')
    if save:
        overview_df.to_excel(base_folder / 'overview_data_pycns_05_02_25.xlsx')


# get_patient_durations_by_stream JOB
get_patient_durations_by_stream_params = {}

def get_patient_durations_by_stream(patient, **p):
    event_file = data_path / patient / 'Events.xml'
    if event_file.is_file():
        evs = pd.DataFrame(pycns.read_events_xml(event_file, time_zone='Europe/Paris'))
        n_sd = evs['name'].apply(lambda x:1 if 'SD' in x else 0).sum()
    else:
        n_sd = 0
    raw_folder = data_path / patient
    uppercase = list(string.ascii_uppercase)
    patient_type = 'SD_ICU' if patient[1] in uppercase else 'non_SD_ICU'
    durations = pd.DataFrame(index = [0], columns = ['patient','patient_type','N_SDs','stream','duration_mins','duration_hours','duration_days','n_chans','ecog_type'])
    durations['patient'] = patient
    durations['patient_type'] = patient_type
    try:
        cns_reader = pycns.CnsReader(raw_folder)
    except:
        return durations
    else:
        stream_keys = cns_reader.streams.keys()
        counter = 0
        if len(stream_keys) > 0:
            for stream_name in stream_keys:
                dates = cns_reader.streams[stream_name].get_times()
                start = dates[0]
                stop = dates[-1]
                start = np.datetime64(start, 'us')
                stop = np.datetime64(stop, 'us')
                duration_us = np.timedelta64(stop - start)
                duration_mins = duration_us.astype('timedelta64[m]').astype(int)
                duration_hours = duration_mins / 60
                duration_days = duration_mins / (60 * 24)
                n_chans = None
                ecog_type = None
                row = [patient, patient_type, n_sd, stream_name, duration_mins, duration_hours, duration_days, n_chans, ecog_type]
                durations.loc[counter,:] = row
                counter += 1
                if stream_name == 'EEG': 
                    ch_names  = cns_reader.streams[stream_name].channel_names      
                    ecog_chans = [chan for chan in ch_names if 'ECoG' in chan]
                    scalp_chans = [chan for chan in ch_names if not chan in ecog_chans]
                    eeg_types = ['Scalp','ECoG'] if len(ecog_chans) > 0 and len(scalp_chans) > 0 else ['Scalp']
                    for eeg_type in eeg_types:
                        if eeg_type == 'Scalp':
                            n_chans = len(scalp_chans)
                            ecog_type = None
                        elif eeg_type == 'ECoG':
                            n_chans = len(ecog_chans)
                            if n_chans == 8:
                                ecog_type = 'depth'
                            elif n_chans == 6:
                                ecog_type = 'strip'
                        row = [patient, patient_type, n_sd, eeg_type, duration_mins, duration_hours, duration_days,n_chans,ecog_type]
                        durations.loc[counter,:] = row
                        counter += 1
        durations['N_SDs'] = durations['N_SDs'].astype(int)
        return xr.Dataset(durations)

def test_get_patient_durations_by_stream(patient):
    print(patient)
    ds = get_patient_durations_by_stream(patient, **get_patient_durations_by_stream_params)
    print(ds.to_dataframe())

get_patient_durations_by_stream_job = jobtools.Job(precomputedir, 'get_patient_durations_by_stream', get_patient_durations_by_stream_params, get_patient_durations_by_stream)
jobtools.register_job(get_patient_durations_by_stream_job)
#

# detailed_view_streams JOB
detailed_view_streams_params = {
    'save':True,
    'patient_list':get_patients_list_raw()
}

def detailed_view_streams(key, **p):
    subs = get_patients_list_raw()
    concat = [get_patient_durations_by_stream_job.get(sub).to_dataframe() for sub in subs]
    detailed_view = pd.concat(concat).reset_index(drop = True)
    # print(detailed_view.dtypes)
    return xr.Dataset(detailed_view)

def test_detailed_view_streams():
    ds = detailed_view_streams('concat', **detailed_view_streams_params)
    # ds.to_netcdf(base_folder / 'test.nc')
    print(ds.to_dataframe())

detailed_view_streams_job = jobtools.Job(precomputedir, 'detailed_view_streams', detailed_view_streams_params, detailed_view_streams)
jobtools.register_job(detailed_view_streams_job)
# 

def save_detailed_view_streams():
    detailed_view_streams_job.get('concat').to_dataframe().to_excel(base_folder / 'detailed_view_data_pycns.xlsx')

def load_patient_duration_stream(patient, stream, unit_col = 'duration_hours'):
    detailed_view = detailed_view_streams_job.get('concat').to_dataframe()
    return detailed_view.set_index(['patient','stream']).loc[(patient,stream),unit_col]

# PLOT NON EEG DATA NANS
plot_nan_map_params = {}

def plot_nan_map(sub, **p):
    cns_reader = pycns.CnsReader(data_path / sub)
    stream_names = list(cns_reader.streams.keys())
    stream_names = [s for s in stream_names if not 'EEG' in s]
    sample_rate = 1
    ds = cns_reader.export_to_xarray(stream_names, resample = True, sample_rate = sample_rate)
    da = ds.to_array(dim = 'chan')
    chans = da['chan'].values
    dates = da['times'].values
    nan_map = da.isnull().values
    prct_nans = np.around(nan_map.sum(axis = 1) / nan_map.shape[1], 2)
    yticks = [f'{chan} ({prct_nans[i]} % of NaNs)' for i, chan in enumerate(chans)]

    cmap = LinearSegmentedColormap.from_list('gr', ['green','red'], N=2)

    fig, ax = plt.subplots()
    ax.imshow(nan_map, cmap=cmap, interpolation='none', aspect = 'auto')

    ax.set_title("Nan Plot: Red for NaNs, Green for Non-NaNs")
    ax.set_xlabel("Date")
    ax.set_xticks(ax.get_xticks()[1:-1], labels = dates[ax.get_xticks()[1:-1].astype(int)].astype('datetime64[s]'), rotation = 90)
    ax.set_ylabel("Stream")
    ax.set_yticks(range(len(yticks)) , labels = yticks)

    fig.savefig(base_folder / 'figures' / 'non_eeg_nan_view' / f'{sub}.png', dpi = 500, bbox_inches = 'tight')
    plt.close(fig)
    return xr.Dataset()

def test_plot_nan_map(sub):
    print(sub)
    ds = plot_nan_map(sub, **plot_nan_map_params)

plot_nan_map_job = jobtools.Job(precomputedir, 'plot_nan_map', plot_nan_map_params, plot_nan_map)
jobtools.register_job(plot_nan_map_job)

# TEST GAIN

def test_gain(sub):
    reader = pycns.CnsReader(data_path / sub)
    print('sub : ',sub)
    for stream_name in ['EEG','ABP','ICP','CO2','ECG_II','ICP_Mean','ABP_Mean']:
        print('stream name : ',stream_name)
        for apply_gain in [False, True]:
            stream = reader.streams[stream_name]
            unit = stream.units
            start_ind = 1e5
            stop_ind = 1e5 + 3
            sigs = stream.get_data(isel = slice(start_ind,stop_ind), apply_gain = apply_gain)
            print('Unit', unit)
            print('apply gain : ',apply_gain)
            if stream_name == 'EEG':
                print(sigs[:,0])
            else:
                print(sigs)

def raw_to_true_units(raw_value):
    MinSV,MaxSV,MinMV,MaxMV = -8388608,8388608,-450000,450000 # EEG sample conversion
    Unitized_value = MinMV + ((raw_value - MinSV) * ((MaxMV - MinMV) / (MaxSV - MinSV)))
    return Unitized_value

def get_patient_list(stream_selection = None, patient_type = None, threshold_duration_mins = 120, threshold_N_SDs = 0, verbose = False):
    """
    Function aiming to get a list of patient based on various criteria (stream availability, patient belonging to SD ICU or not, minimal stream duration).
    
    ----------
    Parameters
    ----------
    - stream_selection : None or list or str
        If None, no selection based of this criteria. Select all pycns readable patients if set to 'readable'. Select pycns readable patients with ICP and ABP available if ['ICP','ABP'] for example. Default = None
    - patient_type : None or str
        If None, no selection is applied. If 'SD_ICU' or 'non_SD_ICU', select patients based on patients belonging to this criteria. Default = None
    - threshold_duration_mins : float or int
        If 0, no selection is applied. Else, apply a selection based on streams with durations of at least this parameter. Default = 120 minutes
    - threshold_N_SDs : int
        Filter patient based on number of SD manually detected by Baptiste in events
    - verbose : bool
        If True, print some informations about the amount of patients selected from the total list. Default = False.

    -------
    Returns
    -------
    - sub_list : List of patients
    """
    detailed_view = detailed_view_streams_job.get('concat').to_dataframe()
    all_patients = list(detailed_view['patient'].unique())

    if threshold_N_SDs > 0:
        patient_n_sds_selected = list(detailed_view[detailed_view['N_SDs'] >= threshold_N_SDs]['patient'].unique())

    if patient_type is None:
        sub_list = all_patients
    else:
        detailed_view = detailed_view[detailed_view['patient_type'] == patient_type]

    
    if stream_selection is None:
        sub_list = list(detailed_view['patient'].unique())

    elif stream_selection == 'readable': 
        vcount = detailed_view['patient'].value_counts()
        mask_vcount = vcount > 1
        sub_list = list(mask_vcount.index[mask_vcount])
    else:
        mask = (detailed_view['stream'].isin(stream_selection)) & (detailed_view['duration_mins']>=threshold_duration_mins)
        masked_df = detailed_view[mask]
        vcount = masked_df['patient'].value_counts()
        sub_list = list(vcount.index[vcount == len(stream_selection)])

    if threshold_N_SDs > 0:
        sub_list = [sub for sub in sub_list if sub in patient_n_sds_selected]

    if 'CO2' in stream_selection:
        flat_co2_subs = ['P61','P95']
        sub_list = [s for s in sub_list if not s in flat_co2_subs]

    if verbose:
        n_before_selection = len(all_patients)
        n_after_selection = len(sub_list)
        print(f'Selection based on streams {stream_selection} and patient of type {patient_type} with at least {threshold_duration_mins} minutes of recording and at least {threshold_N_SDs} SDs kept {n_after_selection} patients from {n_before_selection} initially')
    return sub_list

# RUN JOB

def compute_all():
    # run_keys = [(sub,) for sub in get_patient_list('readable')]
    # jobtools.compute_job_list(plot_nan_map_job, run_keys , force_recompute=True, engine='loop')
    # jobtools.compute_job_list(plot_nan_map_job, run_keys , force_recompute=True, engine='joblib', n_jobs = 2)
    # jobtools.compute_job_list(plot_nan_map_job, run_keys , force_recompute=True, engine='slurm',
    #                           slurm_params={'cpus-per-task':'2', 'mem':'20G', },
    #                           module_name='overview_data_pycns')

    # jobtools.compute_job_list(get_patient_durations_by_stream_job, [(sub,) for sub in get_patients_list_raw()] , force_recompute=False, engine='joblib', n_jobs = 10)

    jobtools.compute_job_list(detailed_view_streams_job, [('concat',)] , force_recompute=True, engine = 'loop')

def debug_get_patient_duration_by_stream():
    bug_subs = []
    for sub in get_patients_list_raw():
        try:
            get_patient_durations_by_stream_job.get(sub).to_dataframe()
        except:
            bug_subs.append(sub)
    print(bug_subs)

if __name__ == '__main__':
    # overview_streams(save = True)
    print('P47' in get_patient_list(['ECG_II']))
    # test_gain('HA1')
    # print(get_patient_durations_by_stream('P17'))
    # detailed_view_streams(save = True)

    # test_plot_nan_map('P18')
    # test_get_patient_durations_by_stream('P98')
    # test_detailed_view_streams()

    # compute_all()

    # debug_get_patient_duration_by_stream()

    # save_detailed_view_streams()

    # get_patient_SD_ICU()

    # print(raw_to_true_units(1493730))

    # sels = [['EEG'],['Scalp'],['ECoG'],['Scalp','ECoG'],['ECoG']]
    # mins = [120,120,120,120,60*24*7]
    # for sel, min in zip(sels,mins):
    #     print(get_patient_list(stream_selection = sel, threshold_duration_mins=min, verbose = True))

    # print(get_patient_list(stream_selection = ['Scalp','ECoG'], patient_type='SD_ICU', threshold_duration_mins=120, verbose = True))
    # print(get_patient_list(stream_selection = 'readable', verbose = True))
    # print(get_patient_list(stream_selection = ['ECG_II'], verbose = True))
    # print(get_patient_list(stream_selection = ['ICP'], verbose = True, threshold_duration_mins=60 * 24))
    # print(get_patient_list(stream_selection = ['ICP','CO2'], verbose = True, threshold_duration_mins=60 * 24))
    # print(get_patient_list(stream_selection = ['ICP','CO2','ABP'], verbose = True, threshold_duration_mins=60 * 24))

    # print(get_patient_list(stream_selection = ['ECoG','Scalp'], verbose = True))
    # print(get_patient_list(stream_selection = ['Scalp','ECoG'], verbose = True, threshold_duration_mins=0))

    # patients_sd_icu = get_patient_list(stream_selection = ['Scalp','ECoG'], patient_type='SD_ICU', threshold_duration_mins=120)
    # df = pd.read_excel(base_folder / 'overview_data_pycns.xlsx', index_col = 0)
    # print(df.loc[patients_sd_icu,:]['Duration_Mins'].sum() / (24 * 60))
    # print(df.loc[patients_sd_icu,:]['Duration_Mins'].agg(['mean','std']) / (24*60))
    
    # print([sub for sub in get_patient_list(None, patient_type = 'SD_ICU') if not sub in get_patient_list(['ECoG'], patient_type = 'SD_ICU')])
    
    # print(get_patient_list(stream_selection = 'readable', threshold_duration_mins = 0, threshold_N_SDs = 1, verbose = True))

    # print([sub for sub in get_patient_list(stream_selection = None) if not sub in get_patient_list(stream_selection = 'readable')])

