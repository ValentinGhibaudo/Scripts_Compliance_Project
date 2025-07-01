global_key = 'all'

load_one_eeg_params = {
    'hours_load':0.1, # duration in hours of eeg stream windows that are loaded
    'apply_gain':True
}

detect_resp_params = {
    'N_cycles_sliding_sd':30, # number of resp cycles in the sliding window computing sd of cycle frequencies
    'threshold_controlled_ventilation_sd_cpm':'auto', # threshold in cycles per minute of SD of sliding cycle freqs to say if ventilation is controlled (True) or assisted (False)
    'N_cycles_sliding_ventilation_bool':100, # number of resp cycles in the sliding window computing averaging boolean classifying of cycles (controlled or not)
}

detect_ecg_params = {}

ihr_params = {
    'detect_ecg_params':detect_ecg_params,
    'limits_bpm':(20,200),
    'interpolation_kind':'linear',
    'srate_interp':8,
}

irr_params = {
    'detect_resp_params':detect_resp_params,
    'interpolation_kind':'linear',
    'srate_interp':2,
}

rsa_params = {
    'detect_resp_params':detect_resp_params,
    'detect_ecg_params':detect_ecg_params
}

irsa_params = {
    'rsa_params':rsa_params,
    'interpolation_kind':'linear',
    'srate_interp':2,
}

detect_abp_params = {
    'lowcut':0.3,
    'highcut':10,
    'order':1,
    'ftype':'bessel',
    'peak_prominence' : 15,
    'h_distance_s' : 0.3,
    'rise_amplitude_limits' : (15,250),
    'amplitude_at_trough_low_limit' : 20,
}

detect_icp_params = {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter',
    'peak_prominence' : 0.5,
    'h_distance_s' : 0.3,
    'rise_amplitude_limits' : (0,20),
    'amplitude_at_trough_low_limit' : -10,
}

prx_params = {
    'wsize_mean_secs':10, 
    'wsize_corr_mins':5, 
    'overlap_corr_prop':0.8,
}

psi_params = {}

crps_params = {
    'detect_resp_params':detect_resp_params,
    'detect_ecg_params':detect_ecg_params
}

ratio_P1P2_params = {
    'down_sample':False,
    'win_compute_duration_hours':1
}

heart_resp_in_icp_params = {
    'spectrogram_win_size_secs':60,
    'resp_fband':(0.12,0.6),
    'heart_fband':(0.8,2.5),
    'rolling_N_time_spectrogram':5,
}

heart_rate_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'ihr_params':ihr_params,
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100,
    }

abp_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'highcut':0.5, 
    'order':4, 
    'ftype':'butter',
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}

icp_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'highcut':0.5, 
    'order':4, 
    'ftype':'butter',
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}

icp_resp_modulated_params = {
    'icp_by_resp_cycle_params':icp_by_resp_cycle_params
}

abp_resp_modulated_params = {
    'abp_by_resp_cycle_params':abp_by_resp_cycle_params
}

icp_pulse_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'detect_icp_params':detect_icp_params, 
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}

abp_pulse_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'detect_icp_params':detect_icp_params, 
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}

icp_pulse_resp_modulated_params = {
    'icp_pulse_by_resp_cycle_params':icp_pulse_by_resp_cycle_params,
    'unit_type':'absolute', # absolute (ptp) or relative (abs(ptp/mean)*100)
}

abp_pulse_resp_modulated_params = {
    'abp_pulse_by_resp_cycle_params':abp_pulse_by_resp_cycle_params,
    'unit_type':'absolute', # absolute (ptp) or relative (abs(ptp/mean)*100)
}

raq_icp_params = {
    'icp_resp_modulated_params':icp_resp_modulated_params,
    'icp_pulse_resp_modulated_params':icp_pulse_resp_modulated_params,
    'n_mad_clean':10,
}

raq_abp_params = {
    'abp_resp_modulated_params':abp_resp_modulated_params,
    'abp_pulse_resp_modulated_params':abp_pulse_resp_modulated_params,
    'n_mad_clean':10,
}

win_size_rolling_secs = 60
ihr_roll_params = {
    'ihr_params':ihr_params,
    'win_size_rolling_secs':win_size_rolling_secs,
}

irr_roll_params = {
    'irr_params':irr_params,
    'win_size_rolling_secs':win_size_rolling_secs,
}

irsa_roll_params = {
    'irsa_params':irsa_params,
    'win_size_rolling_secs':win_size_rolling_secs,
}



