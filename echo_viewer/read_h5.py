import os
import h5py
import numpy as np

class H5Loader:
    def __init__(self, file_path):
        self.file_path = file_path
        # self.find_dataset_indices()
        # self.load_tissue()
        

    def load_file(self):
        with h5py.File(self.file_path, "r") as f:
            print("Keys: %s" % f.keys())
            TDI_key = list(f.keys())[3]
            tissue_key = list(f.keys())[5]
            data_TDI   = list(f[TDI_key])
            data_tissue = list(f[tissue_key])
            print(data_tissue)

    def load_tissue(self):
        with h5py.File(self.file_path, "r") as f:
            tissue_key = list(f.keys())[self.signal_indices['tissue_index']]
            data_tissue = list(f[tissue_key])
            tissue_array = np.array(data_tissue)
            tissue_result = np.flip(np.transpose(tissue_array.astype(float), (1, 0, 2)), 0)

            # velocity_key = list(f.keys())[self.signal_indices['flow_index']]
            #
            # USData_velocity = np.flip(np.transpose(flow_velocity.astype(float) * v_nyquist, (1, 0, 2)), 1)
            # velocity_key = list(f.keys())[self.signal_indices['flow_index']]
            pass

    def find_dataset_indices(self):
        with h5py.File(self.file_path, 'r') as f:
            signal_names = [ds for ds in f.keys()]

        self.signal_indices = {
            'tissue_index': signal_names.index('Tissue') if 'Tissue' in signal_names else None,
            'mmode_index': signal_names.index('MMode') if 'MMode' in signal_names else None,
            'flow_index': signal_names.index('FlowVelocity') if 'FlowVelocity' in signal_names else None,
            'doppler_index': signal_names.index('Doppler') if 'Doppler' in signal_names else None,
            'IQ_index': signal_names.index('IQ') if 'IQ' in signal_names else None,
            'ecg_index': signal_names.index('Ecg') if 'Ecg' in signal_names else None,
        }
        pass




