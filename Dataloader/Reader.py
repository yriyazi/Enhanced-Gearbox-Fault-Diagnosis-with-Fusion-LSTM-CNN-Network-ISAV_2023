import os
import csv
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, List, TypeAlias

DataSetType: TypeAlias = List[Tuple[str, List[List[str]]]]
def read_csv_files(folder_path:str)->DataSetType:
    csv_files: DataSetType = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    csv_files.append((file_path, data))
    return csv_files
    
def array_iefier(csv_files: DataSetType):
    _set: dict[str, NDArray[np.float64]] = {}
    for i in range(10):
        name = str(csv_files[i][0].split('\\')[-1].split('.')[0])
        _set[name] = None

        float_list = [] 
        for _row in csv_files[i][1][1:]:
            float_list .append([float(element) for element in _row])

        _set[name] = np.array(float_list)
    return _set

def _unifier(_set,_min):
    for i in _set.keys():
        _set[i] = _set[i][:_min,:]
    return _set

def dataser_reader():
    _min = 88320

    csv_files_Faulty  = read_csv_files(r'Dataset\Gear Orginal\BrokenTooth')
    csv_files_Healthy = read_csv_files(r'Dataset\Gear Orginal\Healthy')

    _Faulty = array_iefier(csv_files_Faulty)
    _Health = array_iefier(csv_files_Healthy)

    _Faulty = _unifier(_Faulty,_min)
    _Health = _unifier(_Health,_min)

    label_F = []
    for label in _Faulty.keys():
        label_F.append(label)

    label_H = []
    for label in _Health.keys():
        label_H.append(label)

    columns = ['Data', 'Faulty']
    df      = pd.DataFrame( columns=columns)

    for index in range(0,10):
        df_temp_F = pd.DataFrame( columns=columns)
        df_temp_H = pd.DataFrame( columns=columns)

        df_temp_F['Data']     = _Faulty[label_F[index]][:,0]
        df_temp_F['Faulty']   = np.ones((_min))
        df = pd.concat([df, df_temp_F], ignore_index=True)
        del df_temp_F

        df_temp_H['Data']     = _Health[label_H[index]][:,0]
        df_temp_H['Faulty']   = np.zeros((_min))
        df = pd.concat([df, df_temp_H], ignore_index=True)
        del df_temp_H

    return df
