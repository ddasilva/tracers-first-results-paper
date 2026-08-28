"""Computes d' for each of the dispersions in the Sept30 storm.

Writes to output/dprime_{i}.txt and plots/dprime/dprime_plot_{i}.png for all i 
between 0 up to # selections - 1.
"""
import glob
from datetime import datetime
import os
import sys

import pandas as pd
import pylab as plt
import numpy as np
import tqdm

import lib_dasilva2026
import lib_transitdist


df = pd.DataFrame({
    'start_time': [
        datetime(2025, 9, 30, 6, 38),
        datetime(2025, 9, 30, 8, 13),
        datetime(2025, 9, 30, 9, 48),
    ],
    'ead_file': [
        'data/Sept30_Storm/ead/ts2_def_ead_20250930_v0.10.1.cdf',
        'data/Sept30_Storm/ead/ts2_def_ead_20250930_v0.10.1.cdf',
        'data/Sept30_Storm/ead/ts2_def_ead_20250930_v0.10.1.cdf',
    ],
    'xline_file': [
        'data/Sept30_Storm/KH_Xline_Data/30/20250930_0638-TRACERS_xline.txt',
        'data/Sept30_Storm/KH_Xline_Data/30/20250930_0813-TRACERS_xline.txt',
        'data/Sept30_Storm/KH_Xline_Data/30/20250930_0948-TRACERS_xline.txt',
    ],
})


# df = pd.DataFrame({
#     'start_time': [
#         datetime.datetime(2025,12,21,16,46),
#         datetime.datetime(2025,12,22,15,18),
#         datetime.datetime(2026,2,28,3,9)        
#     ],
#     'ead_file': [
#         'data/shirsh/ts2_def_ead_20251221_v0.10.0.cdf',
#         'data/shirsh/ts2_def_ead_20251222_v0.10.0.cdf',
#         'data/shirsh/ts2_def_ead_20260228_v0.10.0.cdf',
#     ],
#     'xline_file': [
#         'data/shirsh/xline_shirsh/20251221_1646-TRACERS_xline.txt',
#         'data/shirsh/xline_shirsh/20251222_1518-TRACERS_xline.txt',
#         'data/shirsh/xline_shirsh/20260228_0309-TRACERS_xline.txt',
#     ]
# })


for i, row in tqdm.tqdm(list(df.iterrows())):
    print('#' * 60)
    print(f'# Working on Row {i} out of {len(df) - 1}')
    print('#' * 60)
    os.makedirs('output/shirsh', exist_ok=True)
    out_name = f'output/shirsh/dprime_{i}.txt'
    b_out_name = f'output/shirsh/b{i}.txt'
    xline_out_name = f'output/shirsh/xline{i}.txt'
    
    #if os.path.exists(out_name):
    #    continue

    stime = row.start_time.to_pydatetime()
    xline_file = row.xline_file

    # Ten attempts with increasingly larger windows. Eventually we will
    # reach a neighbor size equal to len(df_xline), which is an exhaustive
    # search and cannot fail.
    for j in range(1, 10):
        try:
            dprime, B, xline_row = lib_transitdist.calc_transit_dist(
                stime, xline_file, row.ead_file, plot=False,
                lon_nbrhood_size=30*j,
                return_all=True,
            )
            break
        except RuntimeError as e:
            print(e)

    with open(out_name, 'w') as fh:
        fh.write(str(dprime))

    with open(b_out_name, 'w') as fh:
        fh.write(str(B))
        
    with open(xline_out_name, 'w') as fh:
        fh.write(str(xline_row))
