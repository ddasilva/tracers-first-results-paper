import pandas as pd
import pylab as plt
import numpy as np
import glob
from cdasws import CdasWs
import matplotlib.dates as mdates

def make_storm_vis(storm, start_time, end_time, title, no_cpcp=False):
    df = pd.read_csv(f'output/{storm}_filtered.csv', parse_dates=['start_time', 'end_time'], index_col=0)

    # Load recon rates
    # --------------------------------------------------------------------
    recon_rates = {}
    
    for i in range(len(df)):
        recon_rates[i] = pd.read_csv(f'output/{storm}/recon_event{i}.csv')

    # Load OMNI data
    # --------------------------------------------------------------------
    variables = [
       "BX_GSE",
       "BY_GSM",
       "BZ_GSM",
       "Pressure",
       "SYM_H",
    ]
    cdas = CdasWs()
    status, data_xarray = cdas.get_data("OMNI_HRO_1MIN", variables, start_time, end_time)
    omni_epoch = data_xarray['Epoch']
    omni_Bx = data_xarray['BX_GSE']
    omni_By = data_xarray['BY_GSM']
    omni_Bz = data_xarray['BZ_GSM']
    omni_dynpres = data_xarray['Pressure']
    symh = data_xarray['SYM_H']
    
    fill_value = 9999
    omni_Bx[omni_Bx>fill_value] = np.nan
    omni_By[omni_By>fill_value] = np.nan
    omni_Bz[omni_Bz>fill_value] = np.nan
    
    fill_value = 99
    omni_dynpres[omni_dynpres>=fill_value] = np.nan

    # Load cpcp
    # -------------------------------------------------------------------
    if not no_cpcp: 
        cpcp_files = glob.glob(f'data/{storm}/daSilva-cpcp/*.txt')
        dfs_tmp = []
        
        for cpcp_file in cpcp_files:
            df_cpcp = pd.read_csv(cpcp_file, sep='\t')
            df_cpcp.columns = df_cpcp.columns.str.strip()
            df_cpcp['DATE'] = pd.to_datetime(df_cpcp['DATE'])
            dfs_tmp.append(df_cpcp)
        
        df_cpcp = pd.concat(dfs_tmp)
        df_cpcp = df_cpcp.sort_values('DATE')
        df_cpcp = df_cpcp[(df_cpcp.DATE > omni_epoch[0]) & (df_cpcp.DATE < omni_epoch[-1])]

    # Make Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    
    for i in range(len(df)):
        xs = [df.iloc[i].start_time] * len(recon_rates[i])
        ys = recon_rates[i].recon_rate
        axes[0].plot(xs, ys, 'k.')
    
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel('Reconnection Rates\n(mV/m)')
    axes[0].grid(linestyle='dashed')

    if not no_cpcp:
        ax = axes[0].twinx()
        ax.plot(df_cpcp.DATE, df_cpcp.CPCP / 1000, alpha=0.5)
        ax.set_ylabel('SuperDARN\nCross Polar\nCap Potential (kV)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(ax.get_ylim()))
        
    axes[1].plot(omni_epoch, omni_Bx, label='Bx')
    axes[1].plot(omni_epoch, omni_By, label='By')
    axes[1].plot(omni_epoch, omni_Bz, label='Bz')
    axes[1].plot(omni_epoch, np.sqrt(omni_Bx**2 + omni_By**2 + omni_Bz**2), label='|B|', color='k')
    axes[1].legend(ncol=1, bbox_to_anchor=(1.1, .5), loc='center right')
    axes[1].grid(linestyle='dashed')
    axes[1].set_ylabel('OMNI IMF (nT)')

    axes[2].plot(omni_epoch, omni_dynpres, color='y')
    axes[2].set_ylabel('$P_{dyn}$ (nPa)')
    axes[2].grid(linestyle='dashed')

    axes[3].grid(linestyle='dashed')
    axes[3].plot(omni_epoch, symh, color='k')
    axes[3].set_ylabel('SYM-H (nT)')

    axes[3].xaxis.set_major_locator(mdates.DayLocator(interval=2))

    fig.suptitle(f'Dispersion Reconnection Rates During {title}', y=0.95)
    fig.savefig(f'plots/{storm}_recon_during_storm.png', dpi=300, bbox_inches='tight')