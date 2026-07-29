from spacepy import pycdf
import lib_dasilva2026
from datetime import timedelta
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import num2date
import importlib
import pylab as plt
import numpy as np
import pandas as pd
from datetime import timedelta


def make_event_vis(row, ace_file, scoring_file, d, title_label, dispersion_stime, dispersion_etime, alpha=0, ascending=True, offset_timedelta=timedelta(seconds=0), eic_legend_loc='upper right'):
    # Load ACE data -----------------------------
    if ace_file is None:
        ace_data = False
    else:
        ace_data = True
        cdf = pycdf.CDF(ace_file)
        ace_flux = cdf[f"ts2_l2_ace_def"][:]
        ace_spect = ace_flux.sum(axis=-1)
        ace_energies = cdf[f"ts2_l2_ace_energy"][:]
        ace_time = cdf["Epoch"][:]
        cdf.close()

    # Load data
    data = lib_dasilva2026.load_data(
        aci_file=row.aci_file,
        ead_file=row.ead_file,
    )
    subset_data = data.subset(row.start_time, row.end_time)
    dispersion_subset = subset_data.subset(dispersion_stime, dispersion_etime)
    Eic = dispersion_subset.find_Eic(Eic_frac=0.1)
    
    recon_rate, err_low, err_high = dispersion_subset.calculate_recon_rate(
        alpha=alpha,
        d=d,
        Eic_frac=0.1,
        ascending=ascending,
    )
    
    # Do plotting -----------------------
    if ace_data:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    else:
        fig, (ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    
    im = ax1.pcolor(
        subset_data.time,
        subset_data.energies[lib_dasilva2026.CHAN_CUTOFF:],
        subset_data.spect.T[lib_dasilva2026.CHAN_CUTOFF:],
        norm=LogNorm(vmin=lib_dasilva2026.SPECT_VMIN, vmax=lib_dasilva2026.SPECT_VMAX),
        cmap='jet',
    )
    ax1.plot(dispersion_subset.time, Eic, 'b*-', label='Eic: Low Energy Cutoff')
    ax1.set_yscale('log')
    ax1.set_ylabel('Ion Energy (eV)')
    ax1.legend(loc=eic_legend_loc, framealpha=1)

    if ace_data:
        mask = (ace_time> row.start_time) & (ace_time < row.end_time)
        im_e = ax2.pcolor(
            ace_time[mask],
            ace_energies,
            ace_spect[mask].T,
            norm=LogNorm(vmin=1e5, vmax=1e11),
            cmap='jet',
        )
        ax2.set_yscale('log')
        ax2.set_ylabel('Electron Energy (eV)')

    # Plot reconnection rate
    ax3.plot(dispersion_subset.time, recon_rate.value, label='Lockwood Reconnection Rate')
    ax3.fill_between(dispersion_subset.time, err_low.value, err_high.value, alpha=0.25)
    ax3.set_ylabel('Reconnection Rate\n(mV/m)')
    ax3.grid(color='#ccc', linestyle='dashed')
    ax3.legend()
    ax3.set_ylim(0, 5)
    
    # PLot scoring function
    df_scoring = pd.read_csv(scoring_file, parse_dates=['time'])
    label = r"D(t) : Scoring Function | "
    label += f"Total Score: {row.score:.2f}"

    mask = (df_scoring.time > dispersion_stime) & (df_scoring.time < dispersion_etime) 
    ax4.fill_between(df_scoring.time[mask], 0, df_scoring.D[mask], label=label)
    ax4.grid(color='#ccc', linestyle='dashed')
    ax4.set_ylim(-0.25, 0.25)
    ax4.legend()
    ax4.set_ylabel('D(t)')

    # Colorbar tweaking
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical').set_label(r'Summed Omni Flux')

    if ace_data:
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_e, cax=cax, orientation='vertical').set_label(r'Summed Omni Flux')
    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Make ticklabels
    new_xticklabels = []
    
    for xtick in ax4.get_xticks():
        time = num2date(xtick).replace(tzinfo=None)
        i = np.argmin(np.abs(subset_data.time - time))
        new_xticklabel = (
            time.strftime('%H:%M:%S') + '\n'
            + '%.1f' % subset_data.mlat[i] + '\n'
            + '%.1f' % subset_data.mlt[i] + '\n'
        )
        new_xticklabels.append(new_xticklabel)
    
    ax4.set_xticklabels(new_xticklabels)
    ax4.text(-0.15, -0.4, "Time\nMLAT\nMLT", transform=ax4.transAxes)
    ax4.set_xlim(row.start_time + offset_timedelta, row.end_time)
    
    time_str = (
              f"{subset_data.time[0].strftime('%Y-%m-%d')},  "
              f"{subset_data.time[0].strftime('%H:%M:%S')} - {subset_data.time[-1].strftime('%H:%M:%S')} UT"
    )
    imf_str = f"IMF = <{row.Bx:.1f}, {row.By:.1f}, {row.Bz:.1f}> nT"
    y = .97 if ace_data else .99
    fig.suptitle(f'{title_label}\n{time_str}\n{imf_str}', y=y)
    
    return fig