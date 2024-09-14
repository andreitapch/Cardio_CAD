parameters = {
    'save_plot_dir': 'D:\PHD\Development\Python\Cardio_CAD\pattern_recognition\plots',
    'sourceData': 'db'  # xlsx,...
}

stParameters = {
    'stPath': 'D:/PHD/Experiments/CAD/DOPPLER-CIP/analysis 2D speckle tracking (baseline)',
    'stAnalysis_all': 'analyses_ALL',
    'stAnalysis_followup': 'analyses_ALL_FOLLOWUP',

    'imagePath': None,
    'stFilePattern': 'SL_SEG_TRACE.txt'
    # 'stFilePattern': 'FULL_TRACE_SEG.CSV',  # depends on if going to use calculated strain or coord  FIX IMPROVE CALL

}

SPEQLE_parameters = {  # Read from SPEQLE Traces
    'file_path': 'F:/DOPPLER_CIP/SPEQLE/speqle_data/rest_echo/',
    'analysis_folder': 'analysed1_',
    'bl': 'BL',
    'analysed_by': ['Ruta', 'Oana'],
    'segm_names': ['antlat', 'ant', 'antsept', 'infsept', 'inf', 'inflat'],
    'sub_segment_names': ['api', 'mid', 'bas''AntLat'],
    'event_time_fname': '_event_timings.mat',
    'mean_curves_fname': '_meancurves.mat',
    'final_analysis_fname': 'FINAL_ANALYSIS_',
    'event_time_vname': 'event_time',
    'event_length_vname': 'event_length',
    'curve_segment_fname': 'curve_segment',
    'mean_curves_vname': 'mean_curves',
    'mean_ecg_vname': 'mean_ecg',
    'mean_time_vname': 'mean_time',
    'mean_strain_vname': 'mean_strain',
    'mean_sr_vname': 'mean_sr',
    'patient_vname': 'patient_id',
    'segment_vname': 'segment',
    'curve_length_vname': 'curve_length',
    'segment_wall_vname': 'segment_wall',
    'event_lengths_vname': 'event_lengths',
    'time_points_vname': 'time_points',
    'cardiac_view_vname': 'cardiac_view',
    'id_position_fname': 'id_position',
    'view_vname': 'view',
    'timing_vname': 'timing',
    'event_vname': 'event_name',
    'end_systole': 3
}
cardiac_events_order = ['IVC', 'Eject', 'IVR', 'EDF', 'ADF']

Cardiac_events = {
    'EMC': {'Name': 'EMC', 'Description': 'Electromechanical coupling'},
    'IVC': {'Name': 'IVC', 'Description': 'Isovolumic contraction'},
    'Eject': {'Name': 'Eject', 'Description': 'Ejection'},
    'IVR': {'Name': 'IVR', 'Description': 'Isovolumic relaxation'},
    'EDF': {'Name': 'EDF', 'Description': 'Early diastolic filling'},
    'ADF': {'Name': 'ADF', 'Description': 'Atrial diastolic filling'}
}

cardiac_phases = {
    'MVC': {'Name': 'MVC', 'Description': 'Mitral Valve Closes'},
    'AVO': {'Name': 'AVO', 'Description': 'Aortic Valve Opens'},
    'AVC': {'Name': 'AVC', 'Description': 'Aortic Valve Closes'},
    'MVO': {'Name': 'MVO', 'Description': 'Mitral Valve Closes'},
    'ED': {'Name': 'ED', 'Description': 'End of Diastole'},
    'ES': {'Name': 'ES', 'Description': 'End of Systole'}
}

cardiac_wall_order = {
    'Anterior': [1, 7, 13],
    'Anteroseptal': [2, 8, 14],
    'Inferoseptal': [3, 9, 15],
    'Inferior': [15, 10, 4],
    'Inferolateral': [4, 10, 16],
    'Lateral': [5, 11, 17]
}
# segment_names_dic = {
#     '1': {'wall': '','CV': '2CH', 'seg': 'basAnt', 'Name': 'Basal Anterior', 'group': '1'},
#     '2': {'wall': '','CV': 'APLAX', 'seg': 'basAntSept', 'Name': 'Basal Anteroseptal', 'group': '2'},
#     '3': {'wall': '','CV': '4CH', 'seg': 'basSept', 'Name': 'Basal InferoSeptal', 'group': '3'},
#     '4': {'wall': '','CV': '2CH', 'seg': 'basInf', 'Name': 'Basal Inferior', 'group': '4'},
#     '5': {'wall': '','CV': 'APLAX', 'seg': 'basPost', 'Name': 'Basal Inferolateral', 'group': '5'},
#     '6': {'wall': '','CV': '4CH', 'seg': 'basLat', 'Name': 'Basal Lateral', 'group': '6'},
#     '7': {'wall': '','CV': '2CH', 'seg': 'midAnt', 'Name': 'Mid Anterior', 'group': '1'},
#     '8': {'wall': '','CV': 'APLAX', 'seg': 'midAntSept', 'Name': 'Mid Anteroseptal', 'group': '2'},
#     '9': {'wall': '','CV': '4CH', 'seg': 'midSept', 'Name': 'Mid InferoSeptal', 'group': '3'},
#     '10': {'wall': '','CV': '2CH', 'seg': 'midInf', 'Name': 'Mid Inferior', 'group': '4'},
#     '11': {'wall': '','CV': 'APLAX', 'seg': 'midPost', 'Name': 'Mid Inferolateral', 'group': '5'},
#     '12': {'wall': '','CV': '4CH', 'seg': 'midLat', 'Name': 'Mid Anterolateral', 'group': '6'},
#     '13': {'wall': '','CV': '2CH', 'seg': 'apAnt', 'Name': 'Apical Anterior', 'group': '1'},
#     '14': {'wall': '','CV': 'APLAX', 'seg': 'apAntSept', 'Name': 'Apical Anteroseptal', 'group': '2'},
#     '15': {'wall': '','CV': '4CH', 'seg': 'apSept', 'Name': 'Apical InferoSeptal', 'group': '3'},
#     '16': {'wall': '','CV': '2CH', 'seg': 'apInf', 'Name': 'Apical Inferior', 'group': '4'},
#     '17': {'wall': '','CV': 'APLAX', 'seg': 'apPost', 'Name': 'Apical Inferolateral', 'group': '5'},
#     '18': {'wall': '','CV': '4CH', 'seg': 'apLat', 'Name': 'Apical Anterolateral', 'group': '6'}
# }

dbConParameters = {
    'driver': 'ODBC Driver 17 for SQL Server',  # SQL Server
    'server': 'ANDYPC\SQLEXPRESS',
    'database': 'DOPPLER_CIP',
    'tConnection': 'yes',
    'username': None,
    'password': None
}

dbTableNames = {
    'speqle_cardiacEvent_T': 'speqle_cardiacEvent',
    'speqle_strainValues_T': 'speqle_strainValues',
    'speqle_strainValuesInterp_T': 'speqle_strainValuesInterp',
    'speqle_segmentalSr_T': 'speqle_segmentalSr',
    'speqle_segmentalSrInterp_T': 'speqle_segmentalSrInterp',
    'speqle_segmentalStrain_T': 'speqle_segmentalStrain',
    'speqle_segmentalStrainInterp_T': 'speqle_segmentalStrainInterp',
    'calculated_reconstructedTraces_T': 'calculated_reconstructedTraces',
    'calculated_speqle_pca_T': 'calculated_speqle_pca',
    'calculated_pca_results_T': 'calculated_pca_results',
    'st_strainValues_T': 'st_strainValues',
    'st_strainValuesInterp_T': 'st_strainValuesInterp',
    'st_segStrainCurves_T': 'st_segStrainCurves',
    'st_segStrainCurvesInterp_T': 'st_segStrainCurvesInterp',
    'config_tracingSourceName_T': 'config_tracingSourceName',
    'config_cardiacSegments_T': 'config_cardiacSegments',
    'config_repeatabilityCoefficient_T': 'config_repeatabilityCoefficient',
    'config_gender_T': 'config_gender',
    'clinical_Data_Biomarkers_BL_T': 'Clinical_Data_Biomarkers_BL',
    'clinical_Data_Biomarkers_FU_T': 'Clinical_Data_Biomarkers_FU',
    'clinical_Data_BL_T': 'Clinical_Data_BL',
    'clinical_Data_FU_T': 'Clinical_Data_FU',
    'doppler_remodeling_EDV_T': 'doppler_remodeling_EDV',
    'doppler_remodeling_EF_T': 'doppler_remodeling_EF',
    'doppler_remodeling_CMR_T': 'doppler_remodeling_CMR',
    'imgData_2DEcho_rest_BL_T': 'imgData_2DEcho_rest_BL',
    'imgData_2DEcho_rest_FU_T': 'imgData_2DEcho_rest_FU',
    'imgData_2DEcho_speckle_BL_T': 'imgData_2DEcho_speckle_BL',
    'imgData_3DEcho_rest_BL_T': 'imgData_3DEcho_rest_BL',
    'imgData_3DEcho_rest_FU_T': 'imgData_3DEcho_rest_FU',
    'imgData_CMR_rest_BL_T': 'imgData_CMR_rest_BL',
    'imgData_CMR_rest_FU_T': 'imgData_CMR_rest_FU'

}
dbParameters = {
    'studyNr': 'study_nr',
    'view': 'cardView',
    'cardiac_views': ['2ch', '4ch', 'Aplax'],
    'gls_vName': 'gls',
    'ecg_vName': 'ecg',
    'esFrame_vName': 'esFrame',
    'ed1Frame_vName': 'ed1Frame',
    'ed2Frame_vName': 'ed2Frame',
    'ed2Time_vName': 'ed2Time',
    'esTime_vName': 'esTime',
    'ed1Time_vName': 'ed1Time',
    'studyType_vName': 'studyType',
}

param_grid = {
    'hidden_size': [64, 128, 256],
    'num_blocks': [2, 3, 4],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [32, 64, 128]
}
