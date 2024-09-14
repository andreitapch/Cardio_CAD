import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

import pandas as pd
import plotly.express as px

# Directories
root_dir = 'F:\PHD\Experiments\Journal_2023\Results'
folder_name = 'Martinos'  # Martinos , MartinosClutter
model = ''
results_path = os.path.join(root_dir, folder_name, model)


model = ''
file_name = 'GLS-Peak.xlsx'
# pio.renderers.default = 'png'

def mean_squared_error(gt, prediction):
    return np.mean((gt - prediction) ** 2)


def plot_comparison(pred, gt, mitt, file_name, sheet_name):
    mse_gtPred = mean_squared_error(gt, pred)
    mse_gtMitt = mean_squared_error(gt, mitt)

    # Plotting
    plt.plot(gt, label='Ground Truth', marker='o')
    plt.plot(mitt, label='MITT', marker='x')
    plt.plot(pred, label='PWC-Net', marker='s')

    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Comparison of Predictions')
    plt.legend(loc='upper left')

    # Display the MSE values as text on the plot
    plt.text(0.5, max(gt) + 0.5, f'MSE for GT-PWC-Net: {mse_gtPred:.2f}', fontsize=10)
    plt.text(0.5, max(gt), f'MSE for GT-MITT: {mse_gtMitt:.2f}', fontsize=10)

    plt.show()

print('Loading ' + file_name)
xls = pd.ExcelFile(os.path.join(results_path, file_name))
num_sheets = len(xls.sheet_names)

for i, sheet_name in enumerate(xls.sheet_names):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    column_names = df.columns[2:]

    fig = px.box(df, y=column_names.tolist(), color_discrete_sequence=px.colors.sequential.Plasma,
                 points="all")
    fig.update_traces(marker_color='lightseagreen', selector=dict(name='gt'))
    fig.update_traces(marker_color='indianred', selector=dict(name='mitt'))
    fig.update_layout(title_text=sheet_name)
    fig.show()
    save_dir = os.path.join(results_path, 'boxPlot')
    os.makedirs(save_dir, exist_ok=True)

    file_Path = os.path.join(os.path.join(save_dir, sheet_name + '.png'))
    pio.write_image(fig,file_Path)

    # gt = np.array(df['GT'])
    # pred = np.array(df['Pred'])
    # mitt = np.array(df['MITT'])

    # file_name = xlsx_file.replace('.xlsx', '').replace('GLS-', '').replace('_gt_pred', '')
    # plot_comparison(pred, gt, mitt, file_name, sheet_name)

    a = 1
