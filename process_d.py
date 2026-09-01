"""Computes d' for each of the dispersions in the Sept30 storm.

Writes to output/dprime_{i}.txt and plots/dprime/dprime_plot_{i}.png for all i 
between 0 up to # selections - 1.
"""
import glob
import datetime
import os
import sys

import pandas as pd
import pylab as plt
import numpy as np
import tqdm

import lib_dasilva2026
import lib_transitdist

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('storm_name')
args = parser.parse_args()


df = pd.read_csv(f'output/{args.storm_name}_filtered.csv', parse_dates=['start_time', 'end_time'], index_col=0)
df = df.reset_index()

omni_data = lib_dasilva2026.load_omni(glob.glob(f"data/{args.storm_name}/omni/*/*.cdf"))

xline_files = []
xline_files.extend(glob.glob(f"data/{args.storm_name}/KH_Xline_Data/**/*.txt"))
xline_files.extend(glob.glob(f"data/{args.storm_name}/KH_Xline_Data/*.txt"))
xline_times = []

for fname in xline_files:
    fname = os.path.basename(fname)
    parsed_date = datetime.datetime.strptime(fname.split('-')[0], "%Y%m%d_%H%M")
    xline_times.append(parsed_date)

xline_times = np.array(xline_times)

print(xline_times)

for i, row in tqdm.tqdm(list(df.iterrows())):
    print('#' * 60)
    print(f'# Working on Row {i} out of {len(df) - 1}')
    print('#' * 60)
    out_name = f'output/{args.storm_name}/dprime_{i}.txt'
    b_out_name = f'output/{args.storm_name}/b{i}.txt'
    xline_out_name = f'output/{args.storm_name}/xline{i}.txt'
    
    #if os.path.exists(out_name):
    #    continue

    if i < 16:
        continue
    
    stime = row.start_time.to_pydatetime()
    xline_time = xline_times[np.argmin(np.abs(xline_times - stime))]
    time_str = xline_time.strftime("%Y%m%d_%H%M")
    xline_files = []
    xline_files.extend(glob.glob(f"data/{args.storm_name}/KH_Xline_Data/**/{time_str}*.txt"))
    xline_files.extend(glob.glob(f"data/{args.storm_name}/KH_Xline_Data/{time_str}*.txt"))
    xline_file = xline_files[0]

    # -----------------------------------------------------------
    dprime, B, xline_row = lib_transitdist.calc_transit_dist(
        stime, xline_file, row.ead_file,
        lon_nbrhood_size=180,
        plot=False,
        return_all=True,
    )

    os.makedirs(f'plots/dprime/{args.storm_name}', exist_ok=True)
    plt.savefig(f'plots/dprime/{args.storm_name}/dprime_plot_{i}.png', dpi=300)
    plt.close(plt.gcf())
    
    with open(out_name, 'w') as fh:
        fh.write(str(dprime))
    
    with open(b_out_name, 'w') as fh:
        fh.write(str(B))
        
    with open(xline_out_name, 'w') as fh:
        fh.write(str(xline_row))
