import os
import numpy as np
import plotly.io as pio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Directories
root_dir = 'F:\PHD\Experiments\Journal_2023\Results'
folder_name = 'Martinos'  # Martinos , MartinosClutter
model = ''
results_path = os.path.join(root_dir, folder_name, model)
file_name = 'GLS-Peak.xlsx'

print('Loading ' + file_name)
xls = pd.ExcelFile(os.path.join(results_path, file_name))
num_sheets = len(xls.sheet_names)

for i, sheet_name in enumerate(xls.sheet_names):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    column_names = df.columns[2:]

    gt_median = np.median(df['GT'])
    diff = np.abs(df[df.columns[3:]].median() - gt_median)
    closest_col = df.columns[np.argmin(diff) + 3]

    fig = go.Figure()

    # add a box for each column
    for col in column_names:
        if col == 'GT':
            color = 'lightseagreen'
        # elif col == closest_col:
        #     color = 'dodgerblue'
        else:
            color = px.colors.sequential.Plasma[0]  # default color

        fig.add_trace(go.Box(y=df[col], name=col, marker_color=color, boxpoints='all'))

    fig.update_layout(title_text=sheet_name, showlegend=False)

    save_dir = os.path.join(results_path, 'boxPlot')
    os.makedirs(save_dir, exist_ok=True)
    file_Path = os.path.join(os.path.join(save_dir, sheet_name + '.png'))
    pio.write_image(fig, file_Path)
