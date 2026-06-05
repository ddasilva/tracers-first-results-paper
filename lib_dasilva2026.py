from dataclasses import dataclass
from datetime import timedelta

from astropy import units
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num, num2date
import numpy as np
from spacepy import pycdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cdasws import CdasWs
import cdasws
import lib_lockwood1992
import matplotlib.dates as mdates
import pandas as pd

EIC_FRAC = 0.1
CHAN_CUTOFF = 10

SPECT_VMIN = 1e3
SPECT_VMAX = 1e9

@dataclass
class TRACERSData:
    time: np.ndarray
    energies: np.ndarray
    flux: np.ndarray
    spect: np.ndarray
    mlat: np.array
    mlt: np.array
    
    def subset(self, stime, etime):
        i = np.searchsorted(self.time, stime)
        j = np.searchsorted(self.time, etime)
        
        return TRACERSData(
            time=self.time[i:j],
            energies=self.energies,
            flux=self.flux[i:j],
            spect=self.spect[i:j],
            mlat=self.mlat[i:j],
            mlt=self.mlt[i:j],
        )

    def plot_spect(self, fig=None, ax=None, Eic=False, Eic_times=None, Eic_frac=EIC_FRAC, cmap=None):
        if ax is None:
            fig = plt.figure(figsize=(12, 4))
            ax = plt.gca()
            
        im = ax.pcolor(
            self.time,
            self.energies[CHAN_CUTOFF:],
            self.spect.T[CHAN_CUTOFF:],
            norm=LogNorm(vmin=SPECT_VMIN, vmax=SPECT_VMAX),
            cmap=cmap,
        )           

        ax.set_yscale('log')
        ax.set_ylabel('Energy (eV)')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical').set_label(r'Precipitating ($|\alpha_{pa}| < 90^\circ$)' + '\nDifferential Energy Flux')

        if Eic and Eic_times:
            dispersion_subset = self.subset(Eic_times[0], Eic_times[1])
            Eic = dispersion_subset.find_Eic(Eic_frac=Eic_frac)
            ax.plot(dispersion_subset.time, Eic, 'b*-', label='Eic: Low Energy Cutoff')
            ax.legend(loc='upper right', framealpha=1)
                
    def find_Eic(self, smooth=True, Eic_frac=EIC_FRAC, window_size=5):
        Eic = np.zeros(self.time.size)
        Eic[:] = np.nan
        
        for i in range(Eic.size):            
            idx_peak_energy = np.argmax(self.spect[i, CHAN_CUTOFF:]) + CHAN_CUTOFF
            j = idx_peak_energy
            
            while j >  CHAN_CUTOFF:
                if self.spect[i, j] < Eic_frac * self.spect[i, idx_peak_energy]:
                    Eic[i] = self.energies[j]
                    break
                j -= 1

        if smooth:
            Eic = self.smooth_Eic(Eic, window_size)
        
        return Eic 

    def smooth_Eic(self, Eic, window_size=5):
        """Smooth Eic with a mask of points to include in moving average.
        
        Args
          Eic: array
          window_size: integer, must be odd
        Returns
          Smoothed Eic array
        """            
        assert (window_size is None) or (window_size % 2 == 1),\
            'Window size must be odd'
    
        Eic_clean = Eic.copy()
        
        for i in range(Eic.size):
            total = 0.0
            count = 0
            
            for di in range(-window_size//2, window_size//2 + 1):
                if i + di > 0 and i + di < Eic.size:
                    total += Eic[i + di]
                    count += 1
            
            if count > 0:  # else left as nan
                Eic_clean[i] = total / count
                    
        return Eic_clean

    def calculate_recon_rate(self, alpha, d, Eic_frac=EIC_FRAC, ascending=True):
        """Calculate the reconnection rate

        Returns
            recon_rate, err_low, err_high
        """
        Eic = self.find_Eic(Eic_frac=Eic_frac)
                    
        Eic_units = Eic * units.eV
        
        Ey_sat, Ey_mp = lib_lockwood1992.estimate_reconn_rate(
            self.time,
            Eic_units,
            self.mlat,
            alpha,
            d,
            return_error=True,
            ignore_uncertain=True,
            ascending=ascending,
        )

        return (
            Ey_mp['nominal'],
            Ey_mp['err_low'],
            Ey_mp['err_high'],
        )
        
    def plot_recon_rate(self, stime, etime, alpha=0, d=16, Eic_frac=EIC_FRAC, data_file=None, cmap=None, ascending=True):
        # Subset dispersion data and calculate reconnection rate
        dispersion_subset = self.subset(stime, etime)
        Eic = dispersion_subset.find_Eic(Eic_frac=Eic_frac)

        recon_rate, err_low, err_high = dispersion_subset.calculate_recon_rate(
            alpha=alpha,
            d=d,
            Eic_frac=Eic_frac,
            ascending=ascending,
        )

        # Save data if specified -----------
        if data_file is not None:
            df = pd.DataFrame({
                'time': dispersion_subset.time,
                'err_low': err_low,
                'err_high': err_high,
                'recon_rate': recon_rate,
            })
            df.to_csv(data_file, na_rep='NaN', index=0)
            print(f'Wrote to {data_file}')            
            print('RR Max:', np.nanmax(recon_rate))
            
        # Do plotting -----------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        im = ax1.pcolor(
            self.time,
            self.energies[CHAN_CUTOFF:],
            self.spect.T[CHAN_CUTOFF:],
            norm=LogNorm(vmin=SPECT_VMIN, vmax=SPECT_VMAX),
            cmap=cmap,
        )
        ax1.plot(dispersion_subset.time, Eic, 'b*-', label='Eic: Low Energy Cutoff')
        ax1.set_yscale('log')
        ax1.set_ylabel('Energy (eV)')
        ax1.legend(loc='upper right', framealpha=1)
        
        ax2.plot(dispersion_subset.time, recon_rate.value, label='Lockwood Reconnection Rate')
        ax2.fill_between(dispersion_subset.time, err_low.value, err_high.value, alpha=0.25)
        #ax2.set_ylim(0, 2)
        ax2.set_ylabel('Reconnection Rate\n(mV/m)')
        ax2.grid(color='#ccc', linestyle='dashed')
        plt.legend()
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical').set_label(r'Precipitating ($|\alpha_{pa}| < 90^\circ$)' + '\nDifferential Energy Flux')
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        new_xticklabels = []

        for xtick in ax2.get_xticks():
            time = num2date(xtick).replace(tzinfo=None)
            i = np.argmin(np.abs(self.time - time))
            new_xticklabel = (
                time.strftime('%H:%M:%S') + '\n'
                + 'MLAT %.1f' % self.mlat[i]
            )
            new_xticklabels.append(new_xticklabel)

        ax2.set_xticklabels(new_xticklabels)
                
        return fig, (ax1, ax2)

    def plot_omni(self, padding=timedelta(minutes=5), data_file=None):
        # Download data on the fly ----------------------------------------------
        interval = cdasws.timeinterval.TimeInterval(self.time[0] - padding, self.time[-1] + padding)
        cdas = CdasWs()
        dataset = 'OMNI_HRO_1MIN'
        var_names = cdas.get_variable_names(dataset)    
        status, data = cdas.get_data(dataset, var_names, interval)

        mask = data['BX_GSE'] > 999
        mask |= data['BY_GSM'] > 999
        mask |= data['BZ_GSM'] > 999

        # Write data file if requested ----------------------
        if data_file is not None:
            df = pd.DataFrame({
                'time': data['Epoch'][~mask],
                'Bx': data['BX_GSE'][~mask],
                'By': data['BY_GSM'][~mask],
                'Bz': data['BZ_GSM'][~mask],
            })
            df.to_csv(data_file, na_rep='NaN', index=0)
            print(f'Wrote to {data_file}')            

        # Plot data -----------------------------------
        plt.figure(figsize=(12, 4))
        plt.plot(data['Epoch'][~mask], data['BX_GSE'][~mask], 'o-', label='Bx')
        plt.plot(data['Epoch'][~mask], data['BY_GSM'][~mask], 'o-', label='By')
        plt.plot(data['Epoch'][~mask], data['BZ_GSM'][~mask], 'o-', label='Bz')
        plt.axvspan(self.time[0], self.time[-1], alpha=0.25)
        plt.grid(color='#ccc', linestyle='dashed')
        plt.title('OMNI IMF Around Event')
        plt.legend()
        plt.ylabel('nT')
        plt.ylim(-25, 25)
        
        date_format = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(date_format)

        i = np.searchsorted(data['Epoch'], self.time[0])
        print('Bx', data['BX_GSE'][i])
        print('By', data['BY_GSM'][i])
        print('Bz', data['BZ_GSM'][i])
        print('Vsw', data['flow_speed'][i])
        print('n', data['proton_density'][i])
        print('Pdyn', data['Pressure'][i])

def load_data(aci_file, ead_file=None, mlat=None, omni_spect=False):
    if 'ts1' in aci_file:
        key = 'ts1'
    else:
        key = 'ts2'
    
    # Load data from ACI file
    cdf = pycdf.CDF(aci_file)

    flux = cdf[f'{key}_l2_aci_tscs_def'][:]

    if omni_spect:
        spect = flux.sum(axis=-1)
    else:
        spect = cdf[f'{key}_l2_aci_tscs_def'][:, :, 0:9].sum(axis=-1)
    
    energies = cdf[f'{key}_l2_aci_energy'][:]
    time = cdf['Epoch'][:]

    cdf.close()

    # Load MLat from ACI file
    if mlat is not None:
        mlat = mlat * np.ones(time.size)
        mlt = np.nan * np.ones(time.size)
        
    elif ead_file:
        cdf = pycdf.CDF(ead_file)
    
        ead_time = cdf['Epoch'][:]
        ead_mlat = cdf[f'{key}_ead_mlat'][:]
        ead_mlt = cdf[f'{key}_ead_mlt'][:]

        cdf.close()
    
        mlat = np.interp(
            x=date2num(time),
            xp=date2num(ead_time),
            fp=ead_mlat
        )
        mlt = np.interp(
            x=date2num(time),
            xp=date2num(ead_time),
            fp=ead_mlt
        )

    else:
        raise RuntimeError()
    
    
    # Return TRACERSData instance
    return TRACERSData(
        time=time,
        energies=energies,
        flux=flux,
        spect=spect,
        mlat=mlat,
        mlt=mlt,
    )

    

def load_omni(omniweb_files, silent=False):
    """Read OMNIWeb files into a single dictionary.

    Args
      omniweb_files: string path to cdf files
    Returns
      dictionary mapping parameters to file
    """
    # Read OMNIWeb Data
    # ------------------------------------------------------------------------------------
    time_items = []
    Bx_items = []
    By_items = []
    Bz_items = []
    n_items = []
    P_items = []
    v_items = []
    
    for omniweb_file in sorted(omniweb_files):
        # Open file
        if not silent:
            print(f"Loading {omniweb_file}")
        omniweb_cdf = pycdf.CDF(omniweb_file)
        epochs = omniweb_cdf["Epoch"][:]
        #print(omniweb_cdf.keys())
        # Read the data
        time_items.append(np.array([time.replace(tzinfo=None) for time in epochs]))

        Bx_items.append(omniweb_cdf["BX_GSE"][:])
        By_items.append(omniweb_cdf["BY_GSM"][:])
        Bz_items.append(omniweb_cdf["BZ_GSM"][:])
        n_items.append(omniweb_cdf["proton_density"][:])
        P_items.append(omniweb_cdf["Pressure"][:])
        v_items.append(omniweb_cdf['flow_speed'][:])
        #print(omniweb_cdf.keys())
        
    # Merge arrays list of items
    omniweb_data = {}
    omniweb_data["time"] = np.concatenate(time_items)
    omniweb_data["time_d2n"] = date2num(omniweb_data["time"])
    omniweb_data["Bx"] = np.concatenate(Bx_items)
    omniweb_data["By"] = np.concatenate(By_items)
    omniweb_data["Bz"] = np.concatenate(Bz_items)
    omniweb_data["n"] = np.concatenate(n_items)
    omniweb_data["P"] = np.concatenate(P_items)
    omniweb_data["v"] = np.concatenate(v_items)

    # Interpolate over fill values
    for key in omniweb_data:
        if "time" in key:
            continue

        mask = omniweb_data[key] > 999

        omniweb_data[key][mask] = np.interp(
            omniweb_data["time_d2n"][mask],
            omniweb_data["time_d2n"][~mask],
            omniweb_data[key][~mask],
        )

    return omniweb_data
