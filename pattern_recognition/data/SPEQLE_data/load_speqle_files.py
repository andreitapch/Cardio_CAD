import os
import re
import glob
import numpy as np
import pandas as pd
import natsort as ns
from tqdm import tqdm
from scipy.io import loadmat

from pattern_recognition.utils.utils import interpolate_array
from pattern_recognition.utils.parameters_dics import SPEQLE_parameters
from pattern_recognition.utils.exceptions import EmptyDataFrameException
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertTracesToDB as insert_to_db
from pattern_recognition.data.SPEQLE_data.read_speqle_event import ReadSpeqleEvent as read_event
from pattern_recognition.data.SPEQLE_data.read_speqle_curves import ReadSpeqleStrainInfo as read_curves


def interp_for_gls(curvesDf_strain, curvesDf_sr, ecgDf):
    interp_strain, interp_sr, interp_ecg = [], [], []
    interp_strain_df = curvesDf_strain.copy()
    interp_sr_df = curvesDf_sr.copy()
    interp_ecg_df = ecgDf.copy()

    max_systole = np.max(ecgDf['esFrame'])
    max_diastole = np.max(ecgDf['ed2Frame'])
    len_curves = len(curvesDf_strain)
    for idx_segment, segment in enumerate(ecgDf['segment'].unique()):
        idx_init_split = ecgDf['esFrame'][idx_segment]
        for idx_strain in range(len_curves):
            mean_strain = curvesDf_strain['mean_strain'][idx_strain]
            mean_sr = curvesDf_sr['mean_sr'][idx_strain]
            interp_strain.append(interpolate_array(mean_strain, idx_init_split, max_systole, max_diastole))
            interp_sr.append(interpolate_array(mean_sr, idx_init_split, max_systole, max_diastole))
        mean_ecg = ecgDf['ecg'][0]
        interp_ecg.append(interpolate_array(mean_ecg, idx_init_split, max_systole, max_diastole))

    interp_strain_df['mean_strain'] = interp_strain
    interp_sr_df['mean_strain'] = interp_sr
    interp_ecg_df['mean_strain'] = interp_ecg


class SpeqleFileLoader:
    def __init__(self, studyType):
        self.studyType = studyType
        self.mean_curves = None
        self.eventsDf = pd.DataFrame()
        self.curvesDf_strain = pd.DataFrame()
        self.curvesDf_sr = pd.DataFrame()
        self.ecgDf = pd.DataFrame()

        self.file_path = SPEQLE_parameters['file_path']
        self.segments_name = SPEQLE_parameters['segm_names']
        self.bl = SPEQLE_parameters['bl']
        self.event_time_name = SPEQLE_parameters['event_time_fname']
        self.final_analysis_name = SPEQLE_parameters['final_analysis_fname']
        self.user = SPEQLE_parameters['analysed_by'][0]
        self.analysis_folder = SPEQLE_parameters['analysis_folder']
        self.mean_curves_fname = SPEQLE_parameters['mean_curves_fname']
        self.mean_ecg = SPEQLE_parameters['mean_ecg_vname']

        self.subfolder_path = ns.natsorted(os.listdir(self.file_path))
        self.file_pattern = re.compile(r'DOPPLER_(\d+)_DOPPLERANON')

        self.load_files()
        if (not self.eventsDf.empty and not self.curvesDf_strain.empty and not self.curvesDf_sr.empty
                and not self.ecgDf.empty):
            insert_to_db(self.eventsDf, self.curvesDf_strain, self.curvesDf_sr, self.ecgDf)

    def load_files(self):

        for sub_folder in tqdm(self.subfolder_path, total=len(self.subfolder_path)):
            try:
                if sub_folder == 'DOPPLER_5024_DOPPLERANON':
                    print(sub_folder)
                concat_eventsDf = pd.DataFrame()
                concat_curvesDf_strain = pd.DataFrame()
                concat_curvesDf_sr = pd.DataFrame()
                concat_ecgDf = pd.DataFrame()

                patient_id_match = self.file_pattern.match(sub_folder)
                patient_id = patient_id_match.group(1)

                # Timing Events
                event_path = os.path.join(self.file_path, sub_folder, self.bl, self.user + self.event_time_name)

                if not os.path.exists(event_path):
                    continue

                pass
                # Timing Events
                event_mat_file = loadmat(str(event_path))
                event_data = event_mat_file['event_time']
                event = read_event(event_data)
                folder_names = event.ordered_dict['view']['name']
                analysis_folders = event.ordered_dict['view']['analysis']
                segments_available = event.ordered_dict['view']['segment']

                for idx_segment, (segment, fname, fanalysis) in enumerate(zip(segments_available, folder_names,
                                                                             analysis_folders)):
                    matching_folder = glob.glob(os.path.join(self.file_path, sub_folder, self.bl, fname))
                    if not matching_folder:
                        continue

                    mean_curves_files = glob.glob(
                        os.path.join(matching_folder[0], fanalysis, '*' + self.mean_curves_fname))
                    if not mean_curves_files:
                        continue

                    mean_curve_mat = loadmat(str(mean_curves_files[0]))
                    mean_data = mean_curve_mat['mean_curves'][0]
                    curve_length = len(mean_data[self.mean_ecg][0][0])

                    traces = read_curves(mean_data, patient_id, segment, event, idx_segment, curve_length, self.studyType)

                    # Concatenate traces per subject
                    concat_eventsDf = pd.concat([concat_eventsDf, traces.events_df], ignore_index=True)
                    concat_curvesDf_strain = pd.concat([concat_curvesDf_strain, traces.curves_strain_df],
                                                       ignore_index=True)
                    concat_curvesDf_sr = pd.concat([concat_curvesDf_sr, traces.curves_sr_df], ignore_index=True)
                    concat_ecgDf = pd.concat([concat_ecgDf, traces.ecg_df], ignore_index=True)

                self.eventsDf = pd.concat([self.eventsDf, concat_eventsDf], ignore_index=True)
                self.curvesDf_strain = pd.concat([self.curvesDf_strain, concat_curvesDf_strain], ignore_index=True)
                self.curvesDf_sr = pd.concat([self.curvesDf_sr, concat_curvesDf_sr], ignore_index=True)
                self.ecgDf = pd.concat([self.ecgDf, concat_ecgDf], ignore_index=True)

            except EmptyDataFrameException:
                continue
