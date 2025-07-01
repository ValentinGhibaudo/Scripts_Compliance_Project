This repository contains scripts aiming to analyze data acquired with MobergAnalytics amplifier at Neuro intensive care unit of HCL by Baptiste Balança.

Developped by Valentin Ghibaudo

Most of them use pip installable tools except for the computing of Pulse Shape Index (DOI: 10.1109/JBHI.2021.3088629) and P2/P1 ratio (DOI: 10.3390/s23187834) available through request to their own developers, and the pycns toolbox available here : https://github.com/samuelgarcia/pycns

This project is coded through entangled jobs that are automatically run if not yet computed. It means that running the script "icp_slow_rises_jobs.py" will automatically recruite all jobs the the right order (i.e. use the pipeline) and then compute a table of metrics used for statistics in the file slow_icp_rises_compliance.R and in the notebook figures_tables_compliance_paper.ipynb.


# DEFINED JOBS : 

* multi_projects_jobs.py = jobs useful for several projects
    - detect_resp_job : detect all resp cycles of a patient
    - detect_ecg_job : detect all ecg peaks of a patient
    - ihr_job : compute instaneous heart rate of a patient
    - irr_job : compute instaneous respi rate of a patient
    - rsa_job : compute all RSA cycles of a patient
    - irsa_job : compute instaneous RSA of a patient
    - detect_abp_job : detect all ABP cycles of a patient
    - detect_icp_job : detect all ICP cycles of a patient
    - prx_job : compute all PRx of a patient
    - psi_job : compute all Pulse Shape Index of a patient
    - crps_job : compute a vector of strength of Cardio Respi Phase Synchronization for the hwole journey of a patient
    - ratio_P1P2_job : compute P2/P1 ratio time series with ICP
    - heart_resp_in_icp_job : compute time series of heart and resp amplitude components in ICP frequency domain
    - heart_rate_by_resp_cycle_job : compute resp cyclic deformation of heart rate series
    - abp_by_resp_cycle_job : compute resp cyclic deformation of abp smoothed series
    - icp_by_resp_cycle_job : compute resp cyclic deformation of icp smoothed series
    - icp_resp_modulated_job : compute time series of how (peak to peak amplitude by cycle) ICP signal is modulated by respiratory cycle (ICP time domain)
    - abp_resp_modulated_job : compute time series of how (peak to peak amplitude by cycle) ABP signal is modulated by respiratory cycle (ABP time domain)
    - icp_pulse_resp_modulated_job : compute time series of how ICP pulse is modulated by respiratory cycle (ICP pulse time domain)
    - abp_pulse_resp_modulated_job : compute time series of how ABP pulse is modulated by respiratory cycle (ABP pulse time domain)
    - raq_icp_job : compute cycle by cycle RAQ from ICP
    - raq_abp_job : compute cycle by cycle RAQ from ABP
    - ihr_roll_job : compute a rolling median and MAD from ihr trace from ihr_job (ihr = instantaenous heart rate)
    - irr_roll_job : compute a rolling median and MAD from irr trace from irr_job (ihr = instantaenous respi rate)
    - irsa_roll_job : compute a rolling median from irsa trace from irsa_job (irsa = instantaenous respiratory sinus arrhythmia)

* icca_jobs.py
    - icca_bio_job : process icca bio data
    - icca_clinical_job : process icca clinical data
    - icca_pse_tt_job : process icca pse treatments data
    - icca_medication_tt_job : process icca medication treatments data
    - icca_csf_job : process icca csf sampling data

* icp_slow_rises_jobs.py
    - icp_filter_for_detection_job : filter ICP for future detections of events (lowpass < 30 minutes)
    - icp_filter_for_trough_filtering_job : 2nd filter of ICP less lowpass to filter the trough values already high
    - abp_filter_job : filter ABP for the computing CPP
    - slow_icp_rise_detection_job : detect slow icp rises (automatization of nory detections)
    - detection_fig_job : plot detections of slow_icp_rise_detection_job
    - slow_icp_detection_compliance_features_job : label detection from slow_icp_rise_detection_job with compliance features
    - slow_icp_detection_eeg_monopolar_features_job : label detection from slow_icp_rise_detection_job with monopolar montage qEEG features
    - slow_icp_detection_eeg_bipolar_features_job : label detection from slow_icp_rise_detection_job with bipolar montage qEEG features
    - waveform_icp_window_job : one fig by patient, one ax by event, to plot ICP average waveforms according to the window

* overview_data_pycns.py
    - plot_nan_map_job : plot map of nan values from non eeg signals of a patient
    - get_patient_durations_by_stream_job : compute detailed information on all pycns streams of a patient 
    - detailed_view_streams_job : concat get_patient_durations_by_stream_job of all patients
