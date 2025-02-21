# def select_ste_strain_traces_view():
#     query = "SELECT study_nr, cardiacView, segment_wall, ed1Frame, esFrame, ed2Frame, gls, ecg, ed2Time, esTime, " \
#             "ed1Time, timePoints, curve_length, curve_segment, id_position, strain_trace " \
#             "FROM ste_straintracesView ORDER BY cardiacView,curve_segment, study_nr "
#     return query
#
# def select_phi_strain_traces_view():
#     query = "SELECT study_nr, cardiacView, segment_wall, ed1Frame, esFrame, ed2Frame, gls, ecg, ed2Time, esTime, " \
#             "ed1Time, timePoints, curve_length, curve_segment, id_position, strain_trace " \
#             "FROM phi_straintracesView ORDER BY cardiacView,curve_segment, study_nr "
#     return query

def select_total_remodeling_data():
    query = (
        "SELECT Patient_ID, study_nr, remodeling_CMR as 'RM CMR', remodeling_3DEcho as 'RM 3D Echo', remodeling_A2CH as "
        "'RM 2CH', remodeling_A4CH as 'RM 4CH', Total_Remodeling as 'Total RM' "
        "FROM total_Remodeling_view "
        "WHERE Total_Remodeling IS NOT NULL "
        "ORDER BY Total_Remodeling DESC")
    return query


def select_total_doppler_remodeling_data():
    query = (
        "SELECT Patient_ID, RMEDV_MRI  as 'DRM CMR', RMEDV_E3D as 'DRM 3D Echo', RMEDV_A2CH as 'DRM 2CH', RMEDV_A4CH "
        "as 'DRM 4CH' , RMEDV_PARA  as 'DRM PARA' , rmEDV  as 'Total DRM' FROM doppler_remodeling_EDV "
        "WHERE rmEDV IS NOT NULL "
        "ORDER BY rmEDV DESC")
    return query


# def select_studyNr_perNr_segment():
#     query = (
#         "SELECT count as 'Nbr of segments', study_nrs as 'Study nbrs'  "
#         "FROM studyNr_perNr_segment_view"
#     )
#     return query

#
# def select_studyNr_with18():
#     query = (
#         "SELECT study_nrs FROM studyNr_perNr_segment_view where count = 18"
#     )
#     return query


# def select_count_segment_speqle():
#     query = (
#         "SELECT count_segments as 'Nbr of segments' , count_study_nr as 'qty of patients' "
#         "FROM speqle_count_segment_view"
#     )
#     return query


# def select_patient_study_nrs():
#     query = (
#         "select patientID, study_nr, origin "
#         "from patient_study_18seg_view"
#     )
#     return query


# def select_Dremodeling_18seg():
#     query = ("SELECT Patient_ID, RMEDV_MRI as 'DRM CMR', RMEDV_E3D as 'DRM 3D Echo',"
#              " RMEDV_A2CH as 'DRM 2CH', RMEDV_A4CH as 'DRM 4CH', RMEDV_PARA as 'DRM PARA', "
#              "rmEDV as 'Total DRM' FROM doppler_remodeling_EDV "
#              "WHERE Patient_ID IN (SELECT patientID FROM patient_study_18seg_view) AND rmEDV IS NOT NULL "
#              "ORDER BY rmEDV DESC")
#     return query
#
#
# def select_Cremodeling_18seg():
#     query = ("SELECT Patient_ID, study_nr, remodeling_CMR as 'RM CMR', remodeling_3DEcho as 'RM 3D Echo', "
#              "remodeling_A2CH as 'RM 2CH', remodeling_A4CH as 'RM 4CH', Total_Remodeling as 'Total RM' "
#              "FROM total_Remodeling_view "
#              "WHERE Patient_ID IN (SELECT patientID FROM patient_study_18seg_view) AND Total_Remodeling IS NOT NULL "
#              "ORDER BY Total_Remodeling DESC")
#     return query


# def select_merged_remoded_18seg_view():
#     query = ("SELECT Patient_ID, study_nr as 'Study Nr', remodeling_CMR as 'RM CMR', remodeling_3DEcho as 'RM 3DEcho', "
#              "remodeling_A2CH as 'RM 2CH', remodeling_A4CH as 'RM 4CH', Total_Remodeling as 'Total RM', "
#              "RMEDV_MRI as 'DRM CMR', RMEDV_E3D as 'DRM 3DEcho', RMEDV_A2CH as 'DRM 2CH', RMEDV_A4CH as 'DRM 4CH', "
#              "RMEDV_PARA as 'DRM PARA', rmEDV as 'Total DRM' "
#              "FROM merged_remodeling_18seg_view "
#              "ORDER BY Total_Remodeling DESC, rmEDV DESC")
#     return query


def select_delete_all_data(tableName):
    query = f"IF (EXISTS (SELECT * FROM {tableName})) DELETE FROM {tableName}"
    return query


def select_all_data(table_name):
    query = f"(SELECT * FROM {table_name})"
    return query


def select_all_data_where(table_name, column_name, operator, value):
    if operator == '=':
        query = f"SELECT * FROM {table_name} WHERE {column_name} {operator} '{value}'"
    elif operator == 'in':
        value_list = ', '.join(f"'{v}'" for v in value)
        query = f"SELECT * FROM {table_name} WHERE {column_name} {operator} ({value_list})"
    elif operator == 'like':
        query = f"SELECT * FROM {table_name} WHERE {column_name} LIKE '%{value}%'"
    else:
        query = f"SELECT * FROM {table_name}"
    return query

