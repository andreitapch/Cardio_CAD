import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from  pattern_recognition.utils.parameters_dics import segment_name_dict

class InterpolateCurves:
    def __init__(self, dir_path, cardiac_seg, cardiac_phases, reference_label):
        self.control_strain_interp, self.abnormal_sr = [], []
        self.abnormal_strain_interp, self.control_sr = [], []
        self.strain_concatenated, self.sr_concatenated = [], []
        self.interpolated_sr, self.interpolated_strain = [], []
        self.length = {}
        self.idx_max, self.max_phases = [], []
        self.ev_time_sorted, self.label_sorted = [], []

        self.sr_phases, self.strain_phases, self.l_phase = [], [], []
        self.sr_sorted, self.strain_sorted, self.ev_length_sorted = [], [], []

        self.segments = cardiac_seg
        self.phases = cardiac_phases

        list_controls = self.load_data(dir_path, 'control')
        list_abnormal = self.load_data(dir_path, 'abnormal')

        self.sort_st_data(list_controls, reference_label, 'control')
        self.sort_st_data(list_abnormal, reference_label, 'abnormal')
        self.split_st_to_phases('control')

        self.sr_phases_interp = [[] for _ in range(self.phases)]
        self.strain_phases_interp = [[] for _ in range(self.phases)]

        self.interpolate_data()
        self.concatenate_data('control', 'abnormal')

    @staticmethod
    def load_data(root_folder, pat_type):
        if pat_type == 'control':
            pat_type = 'normal'

        folder_path = Path(root_folder) / pat_type.capitalize()
        return glob.glob(os.path.join(folder_path, '**/*.mat'), recursive=True)

    # def sort_st_data(self, file_list, ref_label, pat_type):
    #     keys = ['Label', 'ev_length', 'strain', 'event_time', 'sr']
    #     self.length[pat_type] = len(file_list)
    #
    #     for index_list, fList in enumerate(file_list):
    #         label, ev_length, strain, event_time, sr = [], [], [], [], []
    #         event_list = [label, ev_length, strain, event_time, sr]
    #         load_data = loadmat(fList)
    #         mat_data = load_data['Raw_Curves']
    #         raw_data = mat_data[0][0]
    #
    #         for idx_label in range(len(raw_data['Label'])):
    #             for key, list_ in zip(keys, event_list):
    #                 list_.append(raw_data[key][idx_label][0])
    #         pass

    import pandas as pd

    def sort_st_data(self, file_list, ref_label, pat_type):
        keys = ['Label', 'ev_length', 'strain', 'event_time', 'sr']
        self.length[pat_type] = len(file_list)
        data = []

        for index_list, fList in enumerate(file_list):
            load_data = loadmat(fList)
            mat_data = load_data['Raw_Curves']
            raw_data = mat_data[0][0]

            for idx_label in range(len(raw_data['Label'])):
                row = {}
                for key in keys:
                    row[key] = raw_data[key][idx_label][0]
                row['segment'] = segment_name_dict[str(idx_label)]
                data.append(row)

        df = pd.DataFrame(data)
        return df

    def split_st_to_phases(self, pat_type):
        self.strain_phases = [[] for _ in range(self.phases)]
        self.sr_phases = [[] for _ in range(self.phases)]
        self.l_phase = [[] for _ in range(self.phases)]

        for strain, sr, ev_time in zip(self.strain_sorted, self.sr_sorted, self.ev_time_sorted):
            idx = 0
            for strain_idx, sr_idx in zip(strain, sr):
                for j in range(len(strain_idx)):
                    self.strain_phases[0].append(strain_idx[j][:ev_time[idx][0][0]])
                    self.sr_phases[0].append(sr_idx[j][:ev_time[idx][0][0]])
                    self.l_phase[0].append(len(strain_idx[j][:ev_time[idx][0][0]]))

                    for idx_phases in range(len(ev_time) - 1):
                        phases_start = ev_time[idx][idx_phases][0]
                        phases_end = ev_time[idx][idx_phases + 1][0]
                        self.strain_phases[idx_phases + 1].append(strain_idx[j][phases_start:phases_end])
                        self.sr_phases[idx_phases + 1].append(sr_idx[j][phases_start:phases_end])
                        self.l_phase[idx_phases + 1].append(len(strain_idx[j][phases_start:phases_end]))
                idx += 1
        for i in range(len(self.l_phase)):
            self.max_phases.append(max(self.l_phase[i][0:(self.length[pat_type] * self.segments)]))
            self.idx_max.append(self.l_phase[i].index(self.max_phases[i]))

    def concatenate_data(self, normal_pat, abnormal_pat):
        for i in range(len(self.strain_phases[0])):
            strain = np.concatenate((
                self.strain_phases[0][i], self.strain_phases[1][i], self.strain_phases[2][i], self.strain_phases[3][i],
                self.strain_phases[4][i], self.strain_phases[5][i]), axis=None)
            self.strain_concatenated.append(strain)
            sr = np.concatenate(
                (self.sr_phases[0][i], self.sr_phases[1][i], self.sr_phases[2][i], self.sr_phases[3][i],
                 self.sr_phases[4][i], self.sr_phases[5][i]),
                axis=None)
            self.sr_concatenated.append(sr)

        self.interpolated_strain = np.concatenate((self.strain_phases_interp[:][:]), axis=1).tolist()
        self.interpolated_sr = np.concatenate((self.sr_phases_interp[:][:]), axis=1).tolist()


    def interpolate_data(self):
        for idx_strain in range(len(self.strain_phases)):
            for idx_seg in range(len(self.strain_phases[idx_strain])):
                x1 = np.linspace(0, (self.l_phase[idx_strain][idx_seg] - 1), self.max_phases[idx_strain])
                xp1 = np.arange(0, self.l_phase[idx_strain][idx_seg])
                int_strain_phase = np.interp(x1, xp1, (self.strain_phases[idx_strain][idx_seg]))
                self.strain_phases_interp[idx_strain].append(int_strain_phase)
                int_sr_phase1 = np.interp(x1, xp1, self.sr_phases[idx_strain][idx_seg])
                self.sr_phases_interp[idx_strain].append(int_sr_phase1)



