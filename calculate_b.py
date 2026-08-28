import PyGeopack as gp
from datetime import datetime
import lib_transitdist
import numpy as np

start_times = [
    datetime(2025, 9, 30, 6, 38),
    datetime(2025, 9, 30, 8, 13),
    datetime(2025, 9, 30, 9, 48),
]
pos = [
    (6.770000, 4.750000, 5.480000),
    (0.610000, 6.750000, 11.310000),
    (-0.630000, 9.750000, 10.000000),
]

from rbinvariantslib import models

for i, time in enumerate(start_times):
    params = models.get_tsyganenko_params(time)

    gp_date = int(time.strftime("%Y%m%d"))
    gp_ut = int(time.strftime("%H")) + time.minute / 60

    
    out = gp.ModelField(
        *pos[i],
        Date=gp_date,
        ut=gp_ut,
        Model='T96',
        CoordIn='GSM',
        CoordOut='GSM',
        WithinMPOnly=False,
        **params
    )

    out = np.array(out)
    print("B=", out)
    B = np.linalg.norm(out)

    print("|B|=", B)

