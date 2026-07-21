import pandas as pd
import pylab as plt
import numpy as np
import glob
from cdasws import CdasWs
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.interpolate import make_smoothing_spline
from matplotlib.dates import date2num
from datetime import timedelta

def make_storm_vis(storm, start_time, end_time, title, no_cpcp=False, hl={}):
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
    top_x = []
    top_y = []
    
    for i in range(len(df)):
        xs = [df.iloc[i].start_time] * len(recon_rates[i])
        ys = recon_rates[i].recon_rate

        if i in hl: 
            color, label = hl[i]
            axes[0].plot(xs, ys, '.', color=color, label=label)
        else:
            axes[0].plot(xs, ys, 'k.')
        
        top_x.append(df.iloc[i].start_time)
        m = recon_rates[i].recon_rate < 5
        top_y.append(np.max(recon_rates[i].recon_rate[m]))

    #axes[0].plot(top_x, top_y, color='k')
    if hl:
        axes[0].legend()
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel('Reconnection Rates\n(mV/m)')
    axes[0].grid(linestyle='dashed')

    if not no_cpcp:
        ax = axes[0].twinx()
        ax.plot(df_cpcp.DATE, df_cpcp.CPCP / 1000, alpha=0.5)
        ax.set_ylabel('SuperDARN\nCross Polar\nCap Potential (kV)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(ax.get_ylim()))        
        spl = make_smoothing_spline(date2num(df_cpcp.DATE), df_cpcp.CPCP / 1000, lam=1e1)
        ax.plot(df_cpcp.DATE, spl(date2num(df_cpcp.DATE)), color='C0')

        dt = timedelta(hours=1)
        
        from scipy.stats import linregress
        top_x = np.array(top_x)
        #x = (spl(date2num(top_x + dt)) - spl(date2num(top_x - dt))) / 2
        x = spl(date2num(top_x))
        y = top_y

        r2 = [np.nan, np.nan, np.nan]
        
        for i in range(3, (len(df))):
            reg = linregress(x[:i], y[:i])
            print(i, top_x[i], reg.rvalue**2)
            r2.append(reg.rvalue**2)
            
    axes[1].plot(omni_epoch, omni_Bx, label='Bx', linewidth=0.5)
    axes[1].plot(omni_epoch, omni_By, label='By', linewidth=0.5)
    axes[1].plot(omni_epoch, omni_Bz, label='Bz', linewidth=0.5)
    axes[1].plot(omni_epoch, np.sqrt(omni_Bx**2 + omni_By**2 + omni_Bz**2), label='|B|', color='k', linewidth=0.5)
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

    print(f'SYM-H Minimum:', symh.min())

    fig.suptitle(f'Dispersion Reconnection Rates During {title}', y=0.95)
    fig.savefig(f'plots/{storm}_recon_during_storm.png', dpi=300, bbox_inches='tight')

    if not no_cpcp and len(r2) > 3:
        plt.figure()
    
        fig, axes = plt.subplot_mosaic(
            [["dispersion", "scatter"], ["r2", "scatter"], ["symh", "scatter"]], figsize=(18, 9),
        )

        # Plot Dispersion
        ax = axes["dispersion"]
        ax.set_ylim(0, 5)
        ax.set_ylabel('Reconnection Rates\n(mV/m)')
        ax.grid(linestyle='dashed')
    
        for i in range(len(df)):
            xs = [df.iloc[i].start_time] * len(recon_rates[i])
            ys = recon_rates[i].recon_rate
    
            if i in hl: 
                color, label = hl[i]
                ax.plot(xs, ys, '.', color=color, label=label)
            else:
                ax.plot(xs, ys, 'k.')

        # Plot CPCP
        ax = ax.twinx()
        ax.plot(df_cpcp.DATE, df_cpcp.CPCP / 1000, alpha=0.5)
        ax.set_ylabel('SuperDARN\nCross Polar\nCap Potential (kV)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_ylim(0, max(ax.get_ylim()))        
        spl = make_smoothing_spline(date2num(df_cpcp.DATE), df_cpcp.CPCP / 1000, lam=1e1)
        ax.plot(df_cpcp.DATE, spl(date2num(df_cpcp.DATE)), color='C0')

        # Cumulative R2
        ax = axes["r2"]
        ax.grid(linestyle='dashed')
        i = np.nanargmax(r2)
        print(i)

        ax.text(df.start_time.iloc[0], sum(ax.get_ylim())/4, 'Correlation Less Meaningful\nfor $\\leq$3 Points') 
        
        ax.plot(df.start_time[:i+1], r2[:i+1], 'o-', color='#000000', label='Onset and Main Phase')
        ax.plot(df.start_time, r2, 'o-', color='#cccccc', label='Recovery Period')
        ax.plot(df.start_time[:i+1], r2[:i+1], 'o-', color='#000000')
        ax.set_ylabel('Culumative Correlation\n Cofficient ($r^2$)')
        ax.legend()

        # SymH
        ax = axes['symh']
        ax.grid(linestyle='dashed')
        ax.plot(omni_epoch, symh, color='k')
        ax.set_ylabel('SYM-H (nT)')
            
        # Scatterplot
        ax = axes['scatter']
        ax.grid(linestyle='dashed')

        #x = (spl(date2num(top_x + dt)) - spl(date2num(top_x - dt))) / 2
        x = spl(date2num(top_x))

        y = top_y
        
        ax.plot(x[:i+1], y[:i+1], 'ko', color='#000000', label='Onset and Main Phase')
        ax.plot(x[i+1:], y[i+1:], 'o', color='#cccccc', label='Recovery Period')
        reg = linregress(x[:i], y[:i])

        xvals = np.linspace(x.min() - 10, x.max() + 10,100)
        yvals = reg.slope * xvals + reg.intercept
        ax.plot(xvals, yvals, '-', color='k', label='Regression Fit (Onset and Main Phase Points)')
        ax.legend()

        ax.set_ylabel('Maximum Reconnection Rate per Dispersion (mV/m)')
        ax.set_xlabel('Smoothed SuperDARN Cross Polar Cap Potential (kV)')
        
        axes['r2'].set_xlim(axes['dispersion'].get_xlim())
        axes['symh'].set_xlim(axes['dispersion'].get_xlim())

        fig.suptitle('Reconnection Rate and SuperDARN CPCP Correlation Analysis', y=0.92, fontsize=14)
        fig.savefig(f'plots/{storm}_r2_analysis.png', dpi=300, bbox_inches='tight')
        