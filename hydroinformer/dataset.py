import os
import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class DynamicDataset(Dataset):
    """
    Carrega o dataset do camels com dados de vazao usgs horarios para forcantes dinamicas
    A base lida esta em formato netcdf
    Escrito para trabalhar apenas no modo 'MS' -> multivariado para predizer univariado
    """
    def __init__(self, root_path='./', flag='train', size=None,
                 features='MS', data_path_dynamic = './dataset/dynamic.nc',
                 target='q_obs', scale=True, inverse=False, timeenc=0, freq='h', cols=None):

        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]


        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path_dynamic
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = xr.load_dataset(os.path.join(self.root_path, self.data_path)).to_dataframe()     
        cols = df_raw.columns.tolist()
        cols = cols[2:] # retira gauge_id e a data

        # Verifica qual dataset usar e filtra pela data

        if self.flag == 'train':
            self.start = pd.to_datetime('01/10/1990', dayfirst=True)
            self.end = pd.to_datetime('30/09/2003', dayfirst=True)
            df_data = self.filter_period(df_raw, self.start, self.end)
        elif self.flag == 'test':
            self.start = pd.to_datetime('01/10/2003', dayfirst=True)
            self.end = pd.to_datetime('30/09/2008', dayfirst=True)
            df_data = self.filter_period(df_raw, self.start, self.end)     
        elif self.flag == 'val':
            self.start = pd.to_datetime('01/10/2008', dayfirst=True)
            self.end = pd.to_datetime('30/09/2018', dayfirst=True)
            df_data  = self.filter_period(df_raw, self.start, self.end)
        
        df_filtered = df_data.copy() # versao do df_raw com slice do periodo desejado
        df_data = df_data[cols]      # versao do df_data retirando as colunas gauge_id e date

        # Faz scaling e armazena em 'data'        
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_filtered[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data
        self.data_stamp = data_stamp
        self.df_stamp = df_stamp

       
    def __getitem__(self, index):
        index_n = self.check_index(index)
        s_begin = index_n
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]    
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def check_index(self, index):
        """
        Para um dado 'index', garante que os dados estejam contidos numa mesma bacia,
        pois as bacias foram construidas em um mesmo dataset com append e isso reinicia as datas ao 
        longo do dataset
        """
        index_date = self.df_stamp.iloc[index]
        length_selected = int((self.end - index_date) / pd.Timedelta(hours = 1))
        if self.seq_len > length_selected:
            new_index = index - self.seq_len
            new_index = self.check_index(new_index)
        else:
            new_index = index
        
        return new_index

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def filter_period(self, df_raw, start_date, end_date):
        """
        Filtra conforme o período seja treino, teste ou validação
        """
        df_filtered = df_raw[df_raw['date'] > start_date]
        df_filtered = df_filtered[df_filtered['date'] < end_date]

        return df_filtered


class DynamicStaticDataset(Dataset):
    """
    Carrega o dataset do camels com dados de vazao usgs horarios para forcantes dinamicas
    A base lida esta em formato netcdf
    Escrito para trabalhar apenas no modo 'MS' -> multivariado para predizer univariado
    """
    def __init__(self, root_path='./dataset/', flag='train', size=None,
                 features='MS', data_path_dynamic = 'dynamic.nc',
                 target='q_obs', scale=True, inverse=False, timeenc=0, freq='h', cols=None, data_path_static = 'static.csv'):

        self.flag = flag
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]


        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path_dynamic = data_path_dynamic
        self.data_path_static = data_path_static
        self.__read_data__()

    def __read_data__(self):
        # Lidando com dados estaticos
        self.scaler_static = StandardScaler()
        
        df_raw_static = pd.read_csv(os.path.join(self.root_path, self.data_path_static))
        cols = df_raw_static.columns.tolist()
        cols = cols[2:]
        self.scaler_static.fit(df_raw_static[cols])
        data_static = self.scaler_static.transform(df_raw_static[cols])
        df_scaled = pd.DataFrame(data_static, columns = cols)
        self.df_static = pd.concat([df_raw_static.gauge_id, df_scaled], axis = 1)
        
        # Lidando com dados dinamicos
        df_raw_dynamic = xr.load_dataset(os.path.join(self.root_path, self.data_path_dynamic)).to_dataframe()     
        df_raw_dynamic.gauge_id.astype(int)
        self.scaler_dynamic = StandardScaler()
        cols = df_raw_dynamic.columns.tolist()
        cols = cols[2:] # retira gauge_id e a data

        # Verifica qual dataset usar e filtra pela data
        if self.flag == 'train':
            self.start = pd.to_datetime('01/10/1990', dayfirst=True)
            self.end = pd.to_datetime('30/09/2003', dayfirst=True)
            df_data = self.filter_period(df_raw_dynamic, self.start, self.end)
        elif self.flag == 'test':
            self.start = pd.to_datetime('01/10/2003', dayfirst=True)
            self.end = pd.to_datetime('30/09/2008', dayfirst=True)
            df_data = self.filter_period(df_raw_dynamic, self.start, self.end)     
        elif self.flag == 'val':
            self.start = pd.to_datetime('01/10/2008', dayfirst=True)
            self.end = pd.to_datetime('30/09/2018', dayfirst=True)
            df_data  = self.filter_period(df_raw_dynamic, self.start, self.end)
        
        df_filtered = df_data.copy() # versao do df_raw_dynamic com slice do periodo desejado
        df_data = df_data[cols]      # versao do df_data retirando as colunas gauge_id e date

        # Faz scaling e armazena em 'data'        
        if self.scale:
            self.scaler_dynamic.fit(df_data.values)
            data_dynamic = self.scaler_dynamic.transform(df_data.values)
        else:
            data_dynamic = df_data.values
        
        df_stamp = df_filtered[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data_dynamic
        if self.inverse:
            self.data_y = df_data.values
        else:
            self.data_y = data_dynamic
        self.data_stamp = data_stamp
        self.df_stamp = df_stamp
        self.df_filtered = df_filtered

       
    def __getitem__(self, index):
        index_n = self.check_index(index)
        s_begin = index_n
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]    
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        gauge_id = self.df_filtered.iloc[index_n].gauge_id
        static_attr = self.df_static[self.df_static.gauge_id == int(gauge_id)].drop('gauge_id', axis=1).values
        return seq_x, seq_y, seq_x_mark, seq_y_mark, static_attr
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def check_index(self, index):
        """
        Para um dado 'index', garante que os dados estejam contidos numa mesma bacia,
        pois as bacias foram construidas em um mesmo dataset com append e isso reinicia as datas ao 
        longo do dataset
        """
        index_date = self.df_stamp.iloc[index]
        length_selected = int((self.end - index_date) / pd.Timedelta(hours = 1))
        if self.seq_len > length_selected:
            new_index = index - self.seq_len
            new_index = self.check_index(new_index)
        else:
            new_index = index
        
        return new_index

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def filter_period(self, df_raw, start_date, end_date):
        """
        Filtra conforme o período seja treino, teste ou validação
        """
        df_filtered = df_raw[df_raw['date'] > start_date]
        df_filtered = df_filtered[df_filtered['date'] < end_date]

        return df_filtered        


    
    