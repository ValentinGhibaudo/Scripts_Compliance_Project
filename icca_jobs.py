import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import jobtools
from configuration import *
from icca_tools import *

# PARAMS

icca_biology_params = {
}
icca_clinical_params = {
}
icca_pse_tt_params = {
    'use_human_weight':True
}
icca_medication_tt_params = {
}
icca_csf_params = {
}

# ICCA BIO
def icca_bio(sub, **p):
    ds = load_biology_in_dataset(sub)
    return ds

def test_icca_bio(sub):
    print(sub)
    ds = icca_bio(sub, **icca_biology_params)
    print(ds)

icca_bio_job = jobtools.Job(precomputedir, 'icca_bio', icca_biology_params, icca_bio)
jobtools.register_job(icca_bio_job)

# ICCA CLINICAL
def icca_clinical(sub, **p):
    ds = load_clinical_in_dataset(sub)
    return ds

def test_icca_clinical(sub):
    print(sub)
    ds = icca_clinical(sub, **icca_clinical_params)
    print(ds)

icca_clinical_job = jobtools.Job(precomputedir, 'icca_clinical', icca_clinical_params, icca_clinical)
jobtools.register_job(icca_clinical_job)

# ICCA PSE TT
def icca_pse_tt(sub, **p):
    ds = load_PSE_treatment_in_dataset(sub, use_human_weight=p['use_human_weight'])
    return ds

def test_icca_pse_tt(sub):
    print(sub)
    ds = icca_pse_tt(sub, **icca_pse_tt_params)
    print(ds)

icca_pse_tt_job = jobtools.Job(precomputedir, 'icca_pse_tt', icca_pse_tt_params, icca_pse_tt)
jobtools.register_job(icca_pse_tt_job)

# ICCA MEDICATION TT
def icca_medication_tt(sub, **p):
    ds = load_medication_treatment_in_dataset(sub)
    return ds

def test_icca_medication_tt(sub):
    print(sub)
    ds = icca_medication_tt(sub, **icca_medication_tt_params)
    print(ds)

icca_medication_tt_job = jobtools.Job(precomputedir, 'icca_medication_tt', icca_medication_tt_params, icca_medication_tt)
jobtools.register_job(icca_medication_tt_job)

# ICCA CSF 
def icca_csf(sub, **p):
    ds = load_csf_in_dataset(sub)
    return ds

def test_icca_csf(sub):
    print(sub)
    ds = icca_csf(sub, **icca_csf_params)
    print(ds)

icca_csf_job = jobtools.Job(precomputedir, 'icca_csf', icca_csf_params, icca_csf)
jobtools.register_job(icca_csf_job)

def get_patient_list_icca_treatments(mode, tt_name = None, verbose = False):
    """
    Function aiming to get a list of patient based on various icca treatments criteria    
    ----------
    Parameters
    ----------
    - mode : str
        Should be 'pse' (for pousse-seringue electrique) or 'medication' for one times treatments
    - tt_name : list or None
        List of one or several treatment names used to filter (boolean AND) patients than do have ALL these treatments. If None, return all patients for the selected mode.
    - verbose : bool
        If True, print some available treatments for the mode (pse or medication)
    -------
    Returns
    -------
    - sub_list : List of patients
    """
    icca_subs = get_icca_subs()
    icca_subs_tt = [s for s in icca_subs if (icca_path / s / f'{s}_ICCA_treatment_anonymous.xlsx').exists()]
    pse_bug_subs = ['P1','P73','HA1']
    assert mode in ['pse','medication'], "mode should be 'pse' or 'medication'"
    dict_tt = {}
    all_tt_names = []
    if mode == 'pse':
        job = icca_pse_tt_job
    elif mode == 'medication':
        job = icca_medication_tt_job
    for sub in icca_subs_tt:
        if mode == 'pse' and sub in pse_bug_subs:
            continue
        ds = job.get(sub)
        tt_names = list(ds.keys())
        dict_tt[sub] = tt_names
        all_tt_names.extend(tt_names)
        
    if verbose:
        print(list(set(all_tt_names)))
        
    if tt_name is None:
        return_subs = list(dict_tt.keys())
    else:
        assert isinstance(tt_name,list), "tt_name argument should be of type 'list' even if composed of one element"
        return_subs = []
        for tt in tt_name:
            assert tt in all_tt_names, f'"{tt}" not available'
            return_subs.extend([s for s in dict_tt.keys() if tt in dict_tt[s]])
        return_subs = [s for s in return_subs if return_subs.count(s) == len(tt_name)]
        return_subs = list(set(return_subs))
    assert len(return_subs) > 0, 'no patient available for this condition of treatments'
    return return_subs

def compute_all():
    icca_subs = get_icca_subs()
    sub_keys = [(sub,) for sub in icca_subs]
    # jobtools.compute_job_list(icca_bio_job, sub_keys, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(icca_clinical_job, sub_keys, force_recompute=False, engine='loop')

    no_pse_subs = ['P1','HA1','P73'] # P73 = no flow param for a pse treatment
    pse_tt_keys = [(sub,) for sub in icca_subs if not sub in no_pse_subs]
    jobtools.compute_job_list(icca_pse_tt_job, pse_tt_keys , force_recompute=False, engine='loop')
    # jobtools.compute_job_list(icca_medication_tt_job, sub_keys , force_recompute=False, engine='loop')

    # csf_subs = get_csf_subs()
    # csf_keys = [(sub,) for sub in csf_subs]
    # jobtools.compute_job_list(icca_csf_job, csf_keys,  force_recompute=False, engine='loop')

if __name__ == "__main__":
    test_icca_bio('MF12')
    # test_icca_clinical('P98')
    # test_icca_pse_tt('GA9') # P1 (no PSE TT), P73 (no flow param for a pse treatment), HA1 (no PSE tt)
    # test_icca_medication_tt('P18')
    # test_icca_csf('PL20')

    # compute_all()

