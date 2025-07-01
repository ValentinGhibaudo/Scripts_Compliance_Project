import pandas as pd
import pycns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr
from configuration import *
from scipy import signal
import physio
import matplotlib.dates as mdates
from pycns import CnsReader
import scipy
from matplotlib.colors import Normalize
import itertools
import seaborn as sns

def compute_nanmedian_mad(data, axis=0):
    """
    Compute median and mad

    Parameters
    ----------
    data: np.array
        An array
    axis: int (default 0)
        The axis 
    Returns
    -------
    med: 
        The median
    mad: 
        The mad
    """
    med = np.nanmedian(data, axis=axis)
    mad = np.nanmedian(np.abs(data - med), axis=axis) / 0.6744897501960817
    return med, mad


def get_stream_index_datetime_edges(sub, stream_name = None):
    raw_folder = data_path / sub
    cns_reader = CnsReader(raw_folder)
    if stream_name is None:
        min_ = min([cns_reader.streams[name].index["datetime"][0] for name in cns_reader.streams.keys()])
        max_ = max([cns_reader.streams[name].index["datetime"][-1] for name in cns_reader.streams.keys()])
    else:
        min_ = cns_reader.streams[stream_name].index["datetime"][0]
        max_ = cns_reader.streams[stream_name].index["datetime"][-1]
    return min_, max_

def filter_events(events_df, pattern, mode = 'without', output_mode = 'df'):
    """
    Filter events dataframe based one a string pattern used to keep or reject it.
    Parameters:
        events_df : pd.DataFrame , containing the events with a column 'name' on which the patterns will be filtered
        pattern : str , set the pattern that will be used to filter the column 'name'
        mode : str , 'with' or 'without' to keep or reject the events corresponding to the pattern, resepectively. Default = 'without' = rejection mode.
        output_mode : str , set the output of the function as being a dataframe already filtered if set to 'df' or a boolean mask if set to 'df'. Default = 'df'
    """
    assert mode in ['with','without'], f"'{mode}' search mode not possible.'mode' parameter should be set by 'with' or 'without'"
    assert output_mode in ['df','mask'], f"'{output_mode}' output_mode not possible.'output_mode' parameter should be set by 'df' or 'mask'"
    if mode == 'without':
        mask = events_df['name'].apply(lambda x:False if pattern in x else True)
    elif mode == 'with':
        mask = events_df['name'].apply(lambda x:True if pattern in x else False)
    if output_mode == 'df':
        return events_df[mask].reset_index(drop = True)
    elif output_mode == 'mask':
        return mask

def iqr_interval(a, round_=2, format = 'str'):
    q25, q75 = np.nanquantile(a, 0.25), np.nanquantile(a, 0.75)
    if format == 'str':
        return f'[{q25.round(round_)}, {q75.round(round_)}]' # "If your audience is international, the comma format may be more intuitive,  it's best to include a space inside the brackets for readability"
    elif format == 'tuple':
        return (q25, q75)
    
def med_iqr(a, round_=2):
    q25, q50, q75 = np.nanquantile(a, 0.25), np.nanquantile(a, 0.50), np.nanquantile(a, 0.75)
    return f'{q50.round(round_)} [{q25.round(round_)}, {q75.round(round_)}]' # "If your audience is international, the comma format may be more intuitive,  it's best to include a space inside the brackets for readability"

def notch_filter(sig, srate, bandcut = (48,52), order = 4, ftype = 'butter', show = False, axis = -1):

    """
    IIR-Filter to notch/cut 50 Hz of signal
    """

    band = [bandcut[0], bandcut[1]]
    Wn = [e / srate * 2 for e in band]
    sos = signal.iirfilter(order, Wn, analog=False, btype='bandstop', ftype=ftype, output='sos')
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if show:
        w, h = signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig

def get_patient_dates(patient):
    raw_folder = data_path / patient
    cns_reader = CnsReader(raw_folder)
    stream_keys = cns_reader.streams.keys()
    if len(stream_keys) == 0:
        print(f'{patient} : no stream available to compute dates of patient journey')
        return None, None
    else:
        start = np.min([cns_reader.streams[name].get_times()[0] for name in stream_keys])
        stop = max([cns_reader.streams[name].get_times()[-1] for name in stream_keys])
        start = np.datetime64(start, 'us')
        stop = np.datetime64(stop, 'us')
        return start,stop

def get_metadata(sub = None):
    """
    Inputs
        sub : str id of patient to get its metadata or None if all metadata. Default is None
    Ouputs 
        pd.DataFrame or pd.Series
    """
    path = '/crnldata/REA_NEURO_MULTI_ICU/liste_monito_multi_04_03_2025.xlsx'
    if sub is None:
        return pd.read_excel(path)
    else:
        return pd.read_excel(path).set_index('ID_pseudo').loc[sub,:]
    
def get_patient_ids():
    path = base_folder / 'tab_base_neuromonito.xlsx'
    return list(pd.read_excel(path)['ID_pseudo'])
    
def get_patients_with_ecog():
    path = base_folder / 'tab_base_neuromonito.xlsx'
    df = pd.read_excel(path)
    return df[df['ECoG'] != 0]['ID_pseudo'].values

def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = -1):

    """
    IIR-Filter of signal
    -------------------
    Inputs : 
    - sig : 1D numpy vector
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig


def plot_frequency_response(srate, lowcut=None, highcut=None, order = 4, ftype = 'butter'):
    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    w, h = signal.sosfreqz(sos,fs=srate)
    fig, ax = plt.subplots()
    ax.plot(w, np.abs(h))
    ax.set_title('Frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    plt.show()

def get_amp(sig, axis = -1):
    analytic_signal = signal.hilbert(sig, axis = axis)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def compute_icp(raw_icp, srate, date_vector = None, lowcut = 0.08, highcut = 10, order = 4, ftype = 'butter', peak_prominence = 15, h_distance_s = 0.5, rise_amplitude_limits = (0,20), amplitude_at_trough_low_limit = -10, verbose = False, show = False):
    icp_filt = iirfilt(raw_icp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    maximums,_ = scipy.signal.find_peaks(icp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    minimums,_ = scipy.signal.find_peaks(-icp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    if minimums[0] > maximums[0]: # first point detected has to be a minimum
        maximums = maximums[1:] # so remove the first maximum if is before first minimum
    if maximums[-1] > minimums[-1]: # last point detected has to be a minimum
        maximums = maximums[:-1] # so remove the last maximum if is after last minimum

    peak_index = maximums
    trough_index = minimums[np.searchsorted(minimums, peak_index) - 1]

    detection = pd.DataFrame()
    detection['trough_ind'] = trough_index
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = trough_index[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peak_index
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['next_trough_ind'] = detection['next_trough_ind'].astype(int)
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']

    detection['amplitude_at_trough'] = raw_icp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_icp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_icp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']

    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']].astype('datetime64[ns]')
        detection['peak_date'] =  date_vector[detection['peak_ind']].astype('datetime64[ns]')
        detection['next_trough_date'] =  date_vector[detection['next_trough_ind']].astype('datetime64[ns]')
    
    #cleaning
    detection_clean = detection.copy()
    detection_clean = detection_clean[(detection_clean['amplitude_at_trough'] >= amplitude_at_trough_low_limit)]
    detection_clean = detection_clean[(detection_clean['rise_amplitude'] >= rise_amplitude_limits[0]) & (detection_clean['rise_amplitude'] <= rise_amplitude_limits[1]) ]

    if verbose:
        print("{n_removed} abp cycles were removed by cleaning".format(n_removed = detection.shape[0] - detection_clean.shape[0]))

    if show:
        t = np.arange(raw_icp.size) / srate
        fig, ax = plt.subplots()
        ax.plot(t, raw_icp)
        ax.scatter(t[detection_clean['trough_ind']], raw_icp[detection_clean['trough_ind']], color = 'r')
        ax.scatter(t[detection_clean['peak_ind']], raw_icp[detection_clean['peak_ind']], color = 'g')
        plt.show()

    return detection_clean.reset_index(drop = True)

def compute_abp(raw_abp, srate, date_vector = None, lowcut = 0.5, highcut = 10, order = 1, ftype = 'bessel', peak_prominence = 15, h_distance_s = 0.5, rise_amplitude_limits = (15,200), amplitude_at_trough_low_limit = 20, show = False, verbose = False):
    assert not np.any(np.isnan(raw_abp)), 'Nans in ABP sig'
    abp_filt = iirfilt(raw_abp, srate, lowcut = lowcut, highcut = highcut, order = order, ftype = ftype)
    maximums,_ = scipy.signal.find_peaks(abp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    minimums,_ = scipy.signal.find_peaks(-abp_filt, distance = int(srate * h_distance_s), prominence = peak_prominence)
    if minimums[0] > maximums[0]: # first point detected has to be a minimum
        maximums = maximums[1:] # so remove the first maximum if is before first minimum
    if maximums[-1] > minimums[-1]: # last point detected has to be a minimum
        maximums = maximums[:-1] # so remove the last maximum if is after last minimum

    peak_index = maximums
    trough_index = minimums[np.searchsorted(minimums, peak_index) - 1]

    detection = pd.DataFrame()
    detection['trough_ind'] = trough_index
    detection['trough_time'] =  detection['trough_ind'] / srate
    next_trough_inds = trough_index[1:]
    next_trough_inds = np.append(next_trough_inds, np.nan)
    detection['next_trough_ind'] = next_trough_inds
    detection['next_trough_time'] =  detection['next_trough_ind'] / srate
    detection['peak_ind'] = peak_index
    detection['peak_time'] =  detection['peak_ind'] / srate
    detection = detection.iloc[:-1,:]
    detection['next_trough_ind'] = detection['next_trough_ind'].astype(int)
    detection['rise_duration'] = detection['peak_time'] - detection['trough_time']
    detection['decay_duration'] = detection['next_trough_time'] - detection['peak_time']
    detection['total_duration'] = detection['rise_duration'] + detection['decay_duration']

    detection['amplitude_at_trough'] = raw_abp[detection['trough_ind']]
    detection['amplitude_at_peak'] = raw_abp[detection['peak_ind']]
    detection['amplitude_at_next_trough'] = raw_abp[detection['next_trough_ind']]

    detection['rise_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_trough']
    detection['decay_amplitude'] = detection['amplitude_at_peak'] - detection['amplitude_at_next_trough']

    if not date_vector is None:
        detection['trough_date'] =  date_vector[detection['trough_ind']].astype('datetime64[ns]')
        detection['peak_date'] =  date_vector[detection['peak_ind']].astype('datetime64[ns]')
        detection['next_trough_date'] =  date_vector[detection['next_trough_ind']].astype('datetime64[ns]')
    
    #cleaning
    detection_clean = detection.copy()
    detection_clean = detection_clean[(detection_clean['amplitude_at_trough'] >= amplitude_at_trough_low_limit)]
    detection_clean = detection_clean[(detection_clean['rise_amplitude'] >= rise_amplitude_limits[0]) & (detection_clean['rise_amplitude'] <= rise_amplitude_limits[1]) ]

    if verbose:
        print("{n_removed} abp cycles were removed by cleaning".format(n_removed = detection.shape[0] - detection_clean.shape[0]))

    if show:
        t = np.arange(raw_abp.size) / srate
        fig, ax = plt.subplots()
        ax.plot(t, raw_abp)
        ax.scatter(t[detection_clean['trough_ind']], raw_abp[detection_clean['trough_ind']], color = 'r')
        ax.scatter(t[detection_clean['peak_ind']], raw_abp[detection_clean['peak_ind']], color = 'g')
        plt.show()

    return detection_clean.reset_index(drop = True)

def interpolate_samples(data, data_times, time_vector, kind = 'linear'):
    f = scipy.interpolate.interp1d(data_times, data, fill_value="extrapolate", kind = kind)
    xnew = time_vector
    ynew = f(xnew)
    return ynew

def attribute_subplots(element_list, nrows, ncols):
    assert nrows * ncols >= len(element_list), f'Not enough subplots planned ({nrows*ncols} subplots but {len(element_list)} elements)'
    subplots_pos = {}
    counter = 0
    for r in range(nrows):
        for c in range(ncols):
            if counter == len(element_list):
                break
            subplots_pos[f'{element_list[counter]}'] = [r,c]
            counter += 1
    return subplots_pos  

def get_mcolors():
    from matplotlib.colors import TABLEAU_COLORS
    return list(TABLEAU_COLORS.keys())

def detect_cross(sig, thresh):
    rises, = np.where((sig[:-1] <=thresh) & (sig[1:] >thresh)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=thresh) & (sig[1:] <thresh)) # detect where sign inversion from + to -
    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay
    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def compute_prx(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    icp = df_sigs['icp'].values
    abp = df_sigs['abp'].values
    dates = df_sigs['dates'].values
    
    wsize_inds = int(srate * wsize_mean_secs)

    starts = np.arange(0, icp.size, wsize_inds)

    icp_down_mean = np.zeros(starts.size)
    abp_down_mean = np.zeros(starts.size)

    for i, start in enumerate(starts):
        stop = start + wsize_inds
        if stop > icp.size:
            break
        icp_down_mean[i] = np.mean(icp[start:stop])
        abp_down_mean[i] = np.mean(abp[start:stop])

    dates = dates[::wsize_inds]
    
    corrs_wsize_secs = wsize_corr_mins * 60
    n_samples_win = int(corrs_wsize_secs / wsize_mean_secs)

    n_samples_between_starts = n_samples_win - int(overlap_corr_prop * n_samples_win)

    start_win_inds = np.arange(0, icp_down_mean.size, n_samples_between_starts)

    prx_r = np.zeros(start_win_inds.size)
    prx_pval = np.zeros(start_win_inds.size)
    for i, start_win_ind in enumerate(start_win_inds):
        stop_win_ind = start_win_ind + n_samples_win
        if stop_win_ind > icp_down_mean.size:
            stop_win_ind = icp_down_mean.size
        actual_win_size = stop_win_ind - start_win_ind
        if actual_win_size > 2: # in case when last window too short to compute correlation ...
            res = scipy.stats.pearsonr(icp_down_mean[start_win_ind:stop_win_ind], abp_down_mean[start_win_ind:stop_win_ind])
            prx_r[i] = res.statistic
            prx_pval[i] = res.pvalue
        else: # ... fill last value with pre-last value
            prx_r[i] = prx_r[i-1]
            prx_pval[i] = prx_pval[i-1]
            
        
    dates = dates[start_win_inds]
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates

def compute_prx_and_keep_nans(cns_reader, wsize_mean_secs = 10, wsize_corr_mins = 5, overlap_corr_prop = 0.8):
    all_streams = cns_reader.streams.keys() # get all stream names

    # check if ABP or ART stream in available streams
    if 'ABP' in all_streams: 
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data' # check if ICP stream in available streams
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names]) # compute srate for upsampling based on the most sampled stream
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate) # load ICP and blood pressure streams with same datetime basis
    icp = ds['ICP'].values # icp : dataset to numpy
    abp = ds[abp_name].values # abp : dataset to numpy
    dates = ds['times'].values # datetimes : dataset to numpy

    wsize_inds = int(srate * wsize_mean_secs) # compute window size in points for the local averaging

    start_mean_inds = np.arange(0, icp.size, wsize_inds) # compute start inds
    stop_mean_inds = start_mean_inds + wsize_inds # stop inds = start inds + window size in points
    stop_mean_inds[-1] = icp.size - 1 # last stop ind is replaced by the size of original signal to not slice too far

    icp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean icp signal
    abp_down_local_mean = np.zeros(stop_mean_inds.size) # initialize local mean abp signal
    dates_local_mean = dates[stop_mean_inds] # compute date vector of local means by slicing original dates by stop inds
    
    for i, start, stop in zip(np.arange(start_mean_inds.size), start_mean_inds, stop_mean_inds): # loop over start and stop inds
        icp_down_local_mean[i] = np.mean(icp[start:stop]) # compute local mean icp (return Nan if Nan in the window)
        abp_down_local_mean[i] = np.mean(abp[start:stop]) # compute local mean abp (return Nan if Nan in the window)

    n_samples_by_corr_win = int(wsize_corr_mins * 60 / wsize_mean_secs) # compute number of points by correlation window (seconds durations of corr win / seconds duration of local mean win) (= 30 if corr win = 5 mins and local mean win = 10 secs)
    n_samples_between_start_prx_inds = n_samples_by_corr_win - int(overlap_corr_prop * n_samples_by_corr_win) # = 6 if overlap = 80% and n_samples_by_corr_win = 30
    start_prx_inds = np.arange(0, icp_down_local_mean.size, n_samples_between_start_prx_inds) # compute start inds of prx

    prx_r = np.zeros(start_prx_inds.size) # initialize prx vector of shape start_prx_inds.size
    prx_pval = np.zeros(start_prx_inds.size) # initialize prx pval vector of shape start_prx_inds.size
    dates_prx = [] # initialize a list to store datetimes of prx computing
    for i, start_win_ind in enumerate(start_prx_inds): # loop over start inds
        stop_win_ind = start_win_ind + n_samples_by_corr_win  # compute stop ind = start ind + n_samples_by_corr_win
        if stop_win_ind >= icp_down_local_mean.size: # if stop win index higher that size of local mean sig ...
            stop_win_ind = icp_down_local_mean.size # ... computing window will end at the last local mean sig point
            dates_prx.append(dates_local_mean[stop_win_ind-1]) # add a datetime corresponding to local mean date vector sliced with current stop ind - 1
        else:
            dates_prx.append(dates_local_mean[stop_win_ind]) # add a datetime corresponding to local mean date vector sliced with current stop ind
        actual_win_size = stop_win_ind - start_win_ind # compute the window size in points
        if actual_win_size > 2: # check if window size has at least two points to correlate ...
            icp_sig_win = icp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean icp sig
            abp_sig_win = abp_down_local_mean[start_win_ind:stop_win_ind] # slice the local mean abp sig
            if np.any(np.isnan(icp_sig_win)) or np.any(np.isnan(abp_sig_win)): # check if nan in the slice of local mean sig and fill with nan if it is the case
                prx_r[i] = np.nan
                prx_pval[i] = np.nan
            elif np.std(icp_sig_win) == 0 or np.std(abp_sig_win) == 0: # if icp or abp is constant
                prx_r[i] = np.nan
                prx_pval[i] = np.nan
            else: # if no nan, compute pearson correlation from scipy
                res = scipy.stats.pearsonr(icp_sig_win, abp_sig_win)
                prx_r[i] = res.statistic
                prx_pval[i] = res.pvalue
        else: # ... else fill with a nan if no two points available to correlate
            prx_r[i] = np.nan
            prx_pval[i] = np.nan
    
    dates_prx = np.array(dates_prx).astype('datetime64')
    # print(np.nanmean(prx_r), np.nanstd(prx_r))
    return prx_r, prx_pval, dates_prx

def compute_homemade_prx(cns_reader, win_size_rolling_mins = 5, highcut_Hz=0.1, ftype = 'bessel', order = 4):
    all_streams = cns_reader.streams.keys()
    if 'ABP' in all_streams:
        abp_name = 'ABP'
    elif 'ART' in all_streams:
        abp_name = 'ART'
    else:
        raise NotImplementedError('No blood pressure stream in data')
    assert 'ICP' in all_streams, 'No ICP stream in data'
    stream_names = ['ICP',abp_name]
    srate = max([cns_reader.streams[stream_name].sample_rate for stream_name in stream_names])
    ds = cns_reader.export_to_xarray(stream_names, start=None, stop=None, resample=True, sample_rate=srate)
    df_sigs = pd.DataFrame()
    df_sigs['icp'] = ds['ICP'].values
    df_sigs['abp'] = ds[abp_name].values
    df_sigs['dates'] = ds['times'].values
    df_sigs = df_sigs.dropna()
    df_sigs['icp'] = iirfilt(df_sigs['icp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    df_sigs['abp'] = iirfilt(df_sigs['abp'], srate, highcut = highcut_Hz, ftype = ftype, order = order)
    down_samp_compute = int(srate / (highcut_Hz * 20))
    down_samp_compute = 1 if down_samp_compute < 1 else down_samp_compute
    new_srate = srate / down_samp_compute
    df_sigs = df_sigs.iloc[::down_samp_compute]
    corr = df_sigs['abp'].rolling(int(win_size_rolling_mins * 60 * srate)).corr(df_sigs['icp'])
    corr.index = df_sigs['dates']
    return corr


def init_da(coords, name = None, values = np.nan):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.full(shape, values)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def get_crest_line(freqs, Sxx, freq_axis = 0):
    argmax_freqs = np.argmax(Sxx, axis = freq_axis)
    fmax_freqs = np.apply_along_axis(lambda i:freqs[i], axis = freq_axis, arr = argmax_freqs)
    return fmax_freqs

def load_overview_data():
    return pd.read_excel(base_folder / 'overview_data_pycns_05_02_25.xlsx')

def compute_bins(series, nbins = 50, n_mads = 6):
    a = series.values
    max_ = np.nanmax(a)
    min_ = np.nanmin(a)
    name = series.name
    if np.any(np.isnan(a)):
        a = a[~np.isnan(a)]
    med, mad = physio.compute_median_mad(a)
    inf = med - n_mads * mad
    sup = med + n_mads * mad
    if sup > max_:
        sup = max_
    if inf < min_:
        inf = min_
    if name != 'PRx':
        bins = np.linspace(inf if inf > 0 else 0, sup, nbins)
    else:
        bins = np.linspace(inf, sup, nbins)
    return bins

def get_significance(p_value):
    if p_value <= 0.001:
        return "***"
    elif p_value <= 0.01:
        return "**"
    elif p_value <= 0.05:
        return "*"
    else:
        return "ns"

def pairplot_homemade(data, kind = 'hist', mapper_clean_name = None, savefile = None, **kwargs):
    p_labels = kwargs.get("p_labels", {})  # Extract p_labels if provided
    p_ticks = kwargs.get("p_ticks", {})  # Extract p_ticks if provided
    metrics = data.columns.tolist()
    if mapper_clean_name is None:
        mapper_clean_name = {k:v for k,v in zip(metrics,metrics)}
    combinations = [i for i in itertools.combinations(metrics, 2)]

    nrows = len(data.columns)
    ncols = nrows

    norm = Normalize(-1, 1)
    cmap = plt.get_cmap('seismic')

    figsize = (nrows * 3, ncols * 3)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize, constrained_layout = True)

    is_done = pd.DataFrame(index = data.columns, columns = data.columns, dtype = 'bool')
    is_done[:] = False
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r,c]
            row_metric = data.columns[r]
            col_metric = data.columns[c]
            if row_metric == col_metric and not is_done.iloc[r,c]:
                bins_x = compute_bins(data[row_metric])
                ax.hist(data[row_metric], bins=bins_x, color = 'k')
                ax.set_xlabel(mapper_clean_name[row_metric], **p_labels)
                ax.set_ylabel('Count', **p_labels)
                ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), **p_ticks)
                ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), **p_ticks)
            else:
                if not is_done.iloc[r,c] and not is_done.iloc[c,r]: 
                    bins_x = compute_bins(data[row_metric])
                    bins_y = compute_bins(data[col_metric])
                    if kind == 'hist':
                        sns.histplot(data = data, x = row_metric, y = col_metric, ax = ax, bins = (bins_x,bins_y), color = 'k', pthresh=0., pmax=1)
                    elif kind == 'scatter':
                        sns.scatterplot(data = data, x = row_metric, y = col_metric, ax = ax)
                    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), **p_ticks)
                    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), **p_ticks)
                else:
                    data_sel = data[[row_metric, col_metric]].dropna()
                    res = scipy.stats.spearmanr(data_sel[row_metric], data_sel[col_metric])
                    r_coef = res.statistic
                    p = res.pvalue * len(combinations)
                    p = 1 if p > 1 else p
                    s = 'R : {r_coef}\np : {p}'.format(r_coef=round(r_coef, 3), p = get_significance(p))
                    ax.text(0.5, 0.5, s, ha = 'center', weight = 'bold', fontsize = 20, va = 'center')
                    ax.set_facecolor(cmap(norm(r_coef)))
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_xlabel(mapper_clean_name[row_metric], **p_labels)
                ax.set_ylabel(mapper_clean_name[col_metric], **p_labels)
            is_done.iloc[r,c] = True
    if not savefile is None:
        fig.savefig(savefile, bbox_inches = 'tight', dpi = 500)
    plt.show()

if __name__ == "__main__":

    # compute_prx_and_keep_nans(CnsReader(data_path / 'MF12'))

    # compute_prx(CnsReader(data_path / 'MF12'))

    # print(iqr_interval([np.random.randn(100)]))
    print(med_iqr([np.random.randn(100)], round_=2))
