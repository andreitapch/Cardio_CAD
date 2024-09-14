import os
import pandas as pd
import pyCompare

# Directories
root_dir = 'F:\PHD\Experiments\Journal_2023\Results'
folder_name = 'Martinos'  # Martinos , MartinosClutter
model = ''
results_path = os.path.join(root_dir, folder_name, model)

## Variables
isPlotBlandAlt = True

xlxs_files = [f for f in os.listdir(results_path) if f.endswith('.xlsx')]


def plot_bland_altman(data1, data2, file_name, sheet_name, isMitt):
    bland_altman_path = os.path.join(results_path, 'Bland-Altman',isMitt)
    os.makedirs(bland_altman_path, exist_ok=True)
    save_path = os.path.join(bland_altman_path, sheet_name + file_name + '.png')
    pyCompare.blandAltman(data1, data2, title=(isMitt + sheet_name + ':' + file_name), savePath=save_path)
    # pyCompare.blandAltman(data1, data2, title=('MITT' + ':' + file_name), savePath=save_path)


if not xlxs_files:
    print("No Excel files found in the specified directory.")
else:
    for xlsx_file in xlxs_files:
        xls = pd.ExcelFile(os.path.join(results_path, xlsx_file))
        num_sheets = len(xls.sheet_names)

        for i, sheet_name in enumerate(xls.sheet_names):
            df = pd.DataFrame()
            df = pd.read_excel(xls, sheet_name=sheet_name)

            gt = df['GT']
            pred = df['Pred']
            # pred = df['MITT']
            isMitt =''
            file_name = xlsx_file.replace('.xlsx', '').replace('GLS-', '').replace('_gt_pred', '')
            plot_bland_altman(gt, pred, file_name, sheet_name,isMitt)
            break
