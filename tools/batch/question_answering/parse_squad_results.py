import os
import json
import pandas as pd
import glob
import math
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Convert the tf GPT-2 Model to Gluon.')
parser.add_argument('--dir', type=str, required=True,
                    help='The basic directory to analyze the results.')
parser.add_argument('--save_path', type=str, default=None, help='The path to save the results.')
args = parser.parse_args()

if args.save_path is None:
    args.save_path = os.path.basename(args.dir) + '.csv'

base_dir = args.dir
prefix = 'test_squad2_'

dat_l = []
datetime_parser = '%Y-%m-%d %H:%M:%S,%f'

for folder in sorted(os.listdir(base_dir)):
    if folder.startswith(prefix):
        model_name = folder[len(prefix):]
        log_path_l = glob.glob(os.path.join(base_dir, folder, 'fintune*/finetune*.log'))
        param_path_l = sorted(glob.glob(os.path.join(base_dir, folder, 'fintune*/*.params')))
        if len(param_path_l) == 0 or len(log_path_l) == 0:
            best_f1_threshold = math.nan
            best_exact_threshold = math.nan
            best_f1 = math.nan
            best_em = math.nan
            time_spent_in_hours = math.nan
        else:
            log_path = log_path_l[0]
            result_file = glob.glob(os.path.join(base_dir, folder, 'fintune*/best_results.json'))[0]
            with open(result_file, 'r') as in_f:
                result_dat = json.load(in_f)
            if 'best_f1_thresh' in result_dat:
                best_f1_threshold = result_dat['best_f1_thresh']
            else:
                best_f1_threshold = math.nan
            if 'best_exact_thresh' in result_dat:
                best_exact_threshold = result_dat['best_exact_thresh']
            else:
                best_exact_threshold = math.nan
            best_f1 = result_dat['best_f1']
            best_em = result_dat['best_exact']
            with open(log_path, 'r') as in_f:
                log_lines = in_f.readlines()
                start_time_str = ' '.join(log_lines[0].split()[0:2])
                end_time_str = ' '.join(log_lines[-1].split()[0:2])
                start_time = datetime.strptime(start_time_str, datetime_parser)
                end_time = datetime.strptime(end_time_str, datetime_parser)
                time_spent = end_time - start_time
                time_spent_in_hours = time_spent.total_seconds() / 3600
        dat_l.append({'name': model_name,
                      'best_f1': best_f1,
                      'best_em': best_em,
                      'best_f1_thresh': best_f1_threshold,
                      'best_em_thresh': best_exact_threshold,
                      'time_spent_in_hours': time_spent_in_hours})
df = pd.DataFrame(dat_l)
print(df)
print('Saving to {}'.format(args.save_path))
df.to_csv(args.save_path)
