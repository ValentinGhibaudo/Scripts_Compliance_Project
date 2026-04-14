import sys,os
import getpass
from pathlib import Path

base_folder = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
base_data = '/crnldata/REA_NEURO_MULTI_ICU/'

base_folder = Path(base_folder)
base_data = Path(base_data)
data_path = base_data / 'raw_data'
icca_path = base_data / 'data_ICCA'

base_mnt_data = Path('/mnt/data/NEURO_REA_MONITORAGE/')
precomputedir = base_mnt_data / 'precompute' # mnt/data/
metadata_file = base_data / "liste_monito_multi_13_02_2026.xlsx"


if __name__ == '__main__':
    print(base_folder, base_folder.exists())
    print(data_path, data_path.exists())
    print(precomputedir, precomputedir.exists())
    print(icca_path, precomputedir.exists())
