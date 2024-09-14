import re
import numpy as np
import pandas as pd
from pattern_recognition.utils.parameters_dics import SPEQLE_parameters
from pattern_recognition.utils.exceptions import EmptyDataFrameException


class ReadSpeqleEvent:
    def __init__(self, event_time):
        self.event_indices = []
        self.mean_pos, self.event_length = [], []
        self.dfs, self.ordered_dict = {}, {}
        self.split_struct_to_array(event_time)
        self.filter_dfs()
        self.order_dfs(SPEQLE_parameters['segm_names'])
        self.event_time_loader()

    def split_struct_to_array(self, struct_split):
        for field in struct_split.dtype.names:
            data_dict = {}
            subheaders = struct_split[field][0][0].dtype.names
            for subheader in subheaders:
                data_dict[subheader] = []
                for n in range(len(struct_split[field][0][0][subheader])):
                    data_dict[subheader].append(struct_split[field][0][0][subheader][n][0])
            self.dfs[field] = pd.DataFrame(data_dict)

    def order_dfs(self, seg_order):
        self.ordered_dict = {}
        for field in self.dfs.keys():
            self.dfs[field]['segment'] = pd.Categorical(self.dfs[field]['segment'], categories=seg_order, ordered=True)
            ordered_df = self.dfs[field].sort_values('segment').reset_index(drop=True)
            self.ordered_dict[field] = ordered_df

    def filter_dfs(self):
        pattern = '|'.join(re.escape(name) for name in SPEQLE_parameters['segm_names'])
        if not self.dfs['view'].empty or self.dfs['timing'].emptys:
            self.dfs['view'][['name', 'analysis']] = self.dfs['view'][['name', 'analysis']].astype(str)
            self.dfs['view']['name'] = self.dfs['view']['name'].str.replace('[\'', '').str.replace('\']', '')
            self.dfs['view']['analysis'] = self.dfs['view']['analysis'].str.replace('[\'', '').str.replace('\']', '')

            match = self.dfs['view']['name'].str.contains(pattern, case=False, regex=True)
            self.dfs['view'] = self.dfs['view'][match]
            self.dfs['timing'] = self.dfs['timing'][match]
            self.dfs['view']['name'] = self.dfs['view']['name'].replace(['_WA', '_mid', '_bas'], '', regex=True)
            temp_df = self.dfs['view']['name'].str.split('.').str[0].to_frame('name_without_dot')
            duplicates = temp_df.duplicated(keep=False)
            ends_with_dot_x = self.dfs['view']['name'].str.contains('.\d$')
            original_index = self.dfs['view'].index
            self.dfs['view'] = self.dfs['view'][~(duplicates & ends_with_dot_x)]
            new_index = self.dfs['view'].index
            removed_rows_index = original_index.difference(new_index)

            self.dfs['timing'] = self.dfs['timing'].drop(removed_rows_index, errors='ignore')

            if self.dfs['view'].empty or self.dfs['timing'].empty:
                raise EmptyDataFrameException()

            self.dfs['timing'] = self.dfs['timing'].reset_index(drop=True)
            self.dfs['view'] = self.dfs['view'].reset_index(drop=True)

            temp_df = self.dfs['view']['name'].str.split('_', expand=True)
            self.dfs['view']['segment'] = temp_df.iloc[:, -1].str.split('.', expand=True)[0].str.lower()
            self.dfs['view']['cardiac_view'] = temp_df.iloc[:, 3].replace({'A2C': '2ch', 'A4C': '4ch', 'A3C': 'Aplax'})
            self.dfs['timing']['segment'] = self.dfs['view']['segment']
            self.dfs['timing']['cardiac_view'] = self.dfs['view']['cardiac_view']
            self.dfs['timing'].drop_duplicates(subset=['segment'], inplace=True)
            self.dfs['timing'].reset_index(drop=True, inplace=True)
            self.dfs['view'].drop_duplicates(subset=['segment'], inplace=True)
            self.dfs['view'].reset_index(drop=True, inplace=True)
            pass

    def event_time_loader(self):
        timing = self.ordered_dict[SPEQLE_parameters['timing_vname']]
        # timing.drop_duplicates(subset=['segment'], inplace=True)
        # timing.reset_index(drop=True, inplace=True)
        event_lengths = SPEQLE_parameters['event_lengths_vname']
        segment_df = SPEQLE_parameters['segment_vname']

        seg_len = len(timing[segment_df])
        self.event_length = [0] * seg_len
        self.event_indices = [[0 for _ in range(5)] for _ in range(6)]
        for idx_seg in range(seg_len):
            if len(timing[event_lengths][idx_seg].shape) == 3:
                seg_ev_length = timing[event_lengths][idx_seg][:, :, 0]
            elif len(timing[event_lengths][idx_seg].shape) == 2:
                seg_ev_length = timing[event_lengths][idx_seg]
            else:
                print("Unexpected dimensions for timing[event_lengths][idx_seg]")
            seg_ev_length = seg_ev_length.tolist()
            self.event_length[idx_seg] = np.max(seg_ev_length, axis=1).tolist()
            self.event_indices[idx_seg][0] = self.event_length[idx_seg][0] + 1
            for counter in range(1, len(self.event_indices[idx_seg])):
                self.event_indices[idx_seg][counter] = self.event_indices[idx_seg][counter - 1] + \
                                                       self.event_length[idx_seg][counter]
            pass
