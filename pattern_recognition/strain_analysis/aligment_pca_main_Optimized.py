import os
import glob
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
from itertools import zip_longest
from sklearn.decomposition import PCA


class Alignment_PCA:
    def __init__(self, dir_path, cardiac_seg, cardiac_phases, ref_label):
        self.length = {}
        self.idx_max, self.max_phases = [], []
        self.ev_time_sorted, self.label_sorted = [], []

        self.sr_phases, self.strain_phases, self.l_phase = [], [], []
        self.sr_sorted, self.strain_sorted, self.ev_length_sorted = [], [], []

        self.segments = cardiac_seg
        self.phases = cardiac_phases

        list_controls = self.load_data(dir_path, 'normal')
        list_abnormal = self.load_data(dir_path, 'abnormal')

        self.sort_st_data(list_controls, ref_label, 'normal')
        self.sort_st_data(list_abnormal, ref_label, 'abnormal')
        self.split_st_to_phases('normal')

        self.sr_phases_interp = [[] for _ in range(self.phases)]
        self.strain_phases_interp = [[] for _ in range(self.phases)]

        self.interpolate_data()
        # control = self.concatenate_data(control_sorted)
        # abnormal = self.concatenate_data(control_abnormal)

        normal_strain_curve, abnormal_strain_curve = 0, 0
        # self.PCA(normal_strain_curve, abnormal_strain_curve)

    @staticmethod
    def load_data(root_folder, pat_type):
        folder_path = Path(root_folder) / pat_type.capitalize()
        return glob.glob(os.path.join(folder_path, '**/*.mat'), recursive=True)

    def sort_st_data(self, file_list, ref_label, pat_type):
        keys = ['Label', 'ev_length', 'strain', 'event_time', 'sr']
        self.length[pat_type] = len(file_list)

        for index_list, fList in enumerate(file_list):
            ev_time_ref, strain_ref, sr_ref, ev_length_ref, label_ref = [], [], [], [], []
            label, ev_length, strain, ev_time, sr = [], [], [], [], []
            event_list = [label, ev_length, strain, ev_time, sr]
            load_data = loadmat(fList)
            mat_data = load_data['Raw_Curves']
            raw_data = mat_data[0][0]

            for idx_label in range(len(raw_data['Label'])):
                for key, list_ in zip(keys, event_list):
                    list_.append(raw_data[key][idx_label][0])

            for j in range(len(label)):
                for i in range(len(label)):
                    if label[i][0] == ref_label[j]:
                        ev_time_ref.append(ev_time[i])
                        strain_ref.append(strain[i])
                        sr_ref.append(sr[i])
                        ev_length_ref.append(ev_length[i])
                        label_ref.append(label[i])

            self.ev_time_sorted.append(ev_time_ref)
            self.strain_sorted.append(strain_ref)
            self.sr_sorted.append(sr_ref)
            self.ev_length_sorted.append(ev_length_ref)
            self.label_sorted.append(label_ref)

    def split_st_to_phases(self, pat_type):
        self.strain_phases = [[] for _ in range(self.phases)]
        self.sr_phases = [[] for _ in range(self.phases)]
        self.l_phase = [[] for _ in range(self.phases)]

        for strain, sr, ev_time in zip(self.strain_sorted, self.sr_sorted, self.ev_time_sorted):
            idx = 0
            for strain_idx, sr_idx in zip(strain, sr):
                for j in range(len(strain_idx)):
                    print('idx', idx)
                    print('j', j)
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

    def interpolate_data(self):
        for idx_strain in range(len(self.strain_phases)):
            for idx_seg in range(len(self.strain_phases[idx_strain])):
                x1 = np.linspace(0, (self.l_phase[idx_strain][idx_seg] - 1), self.max_phases[idx_strain])
                xp1 = np.arange(0, self.l_phase[idx_strain][idx_seg])
                int_strain_phase1 = np.interp(x1, xp1, (self.strain_phases[idx_strain][idx_seg]))
                self.strain_phases_interp[idx_strain].append(int_strain_phase1)
                int_sr_phase1 = np.interp(x1, xp1, self.sr_phases[idx_strain][idx_seg])
                self.sr_phases_interp[idx_strain].append(int_sr_phase1)

    @staticmethod
    def concatenate_data(sorted_list):
        a = 0
        return a

    def plot_data(self, data, title, ylabel, xlabel):

        pass

    def align_curves(self):

        pass

    @staticmethod
    def PCA(strain_curve):
        pca = PCA()
        principalComponents = pca.fit_transform(strain_curve[0:18][:])
        pcas = principalComponents.transpose((1, 0))
        fig = plt.figure
        plt.plot(pcas)
        plt.show()

        pass


root_path = os.path.dirname('D:/PHD/Experiments/CAD/Data-strain/')
segments = 18
phases = 6
ref_label = [1, 2, 3, 4, 5, 6]

Alignment_PCA(root_path, segments, phases, ref_label)
