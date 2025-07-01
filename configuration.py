import sys,os
import getpass
from pathlib import Path

base_folder = '/crnldata/tiger/baptiste.balanca/Neuro_rea_monitorage/'
base_data = '/crnldata/REA_NEURO_MULTI_ICU/'

base_folder = Path(base_folder)
base_data = Path(base_data)
data_path = base_data / 'raw_data'
icca_path = base_data / 'data_ICCA'

precomputedir = base_folder / 'precompute_del'


if __name__ == '__main__':
    print(base_folder, base_folder.exists())
    print(data_path, data_path.exists())
    print(precomputedir, precomputedir.exists())
    print(icca_path, precomputedir.exists())
