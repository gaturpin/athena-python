import sys
import os
sys.path.insert(0, '/Users/georgeturpin/binary_athena/athena/vis/python/')
import athena_read
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from scipy.integrate import simpson

def CleanData(df):
    """ Cleans dataframe, removes any extra data the second restart overlaps with. """

    # Find points where value of next orbit is lower than previous, indicating restart
    restart_points = df[df['orbit'] < df['orbit'].shift(1)].index
    indices_to_drop = []

    # If restart found, clean and return
    if restart_points > 0:
        for restart_point in restart_points:

            initial_orbit_value = df.at[restart_point, 'orbit']
            pre_restart_point = df.index[df['orbit'] > initial_orbit_value][0]
            indices_to_drop.extend(range(pre_restart_point, restart_point))

        df_cleaned = df.drop(indices_to_drop).reset_index(drop=True)
        return df_cleaned

    # Else no restart found, data already clean
    else:
        return df

def KineticEnergy(vx, vy, m):
    """ Calculates kinetic energy of binary """
    vel_mag = np.sqrt(vx*vx + vy*vy)
    kin_energy = vel_mag**2/2
    return kin_energy

def GravEnergy(df):
    """ Calculates gravitational energy of binary """
    d = np.sqrt((df.x1-df.x2)*(df.x1-df.x2) + (df.y1-df.y2)*(df.y1-df.y2))
    return -(df.m1+df.m2)/d

def SpecificEnergy(df):
    """ Calculates specific energy of binary """
    vx = df.vx1 - df.vx2
    vy = df.vy1 - df.vy2
    return KineticEnergy(vx, vy, df.m1) + GravEnergy(df)

def SpecificAngularMomentum(df):
    """ Calculates specific angular momentum of binary """
    vx, vy = df.vx1 - df.vx2, df.vy1 - df.vy2        
    x = df.x1 - df.x2
    y = df.y1 - df.y2
    L = x*vy - y*vx
    return L

def SMA(df):
    """ Calculates semi-major axis of binary """
    return (df.m1 + df.m2)/(2*-Specific_Energy(df))

def Eccentricity(df):
    """ Calculates eccentricity of binary """
    return np.sqrt(1 + 2*(Specific_Energy(df)*SpecificAngularMomentum(df)**2)/((df.m1 + df.m2)**2))

def DensityProfile(data):
    """ Calculates 1D surface density profile from athena dataframe (.athdf file) 
    returns binned radius, 1d density profile """
    
    density = data['rho'].flatten()
    xx, yy = np.meshgrid(data['x1v'], data['x2v'], indexing= 'xy')
    x = xx.flatten()
    y = yy.flatten()
    radius = np.sqrt(x*x + y*y)

    mean_density = binned_statistic(radius, density, statistic='mean',bins = 1000, range=(0,50))
    density_profile = mean_density.statistic

    bin_edges = mean_density.bin_edges
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centres, density_profile


def DensityProfileAveraged(dir_loc, file_name, file_ids):
    """ Calcuates the mean 1D density profile over a specific file number range
    dir_loc - string, location of directory with athena .athdf outputs
    file_name - file name of run, typically binary_wind
    file_ids - size 2, [file number start, file number end] 
    returns radius, mean 1d density profile """
    
    start, end = file_ids
    for i in range(start, end+1, 1):
        data = athena_read.athdf(dir_loc + file_name + '.prim.' + str(i).zfill(5) + '.athdf')
        radius, density_1d = DensityProfile(data)
        if i==start:
            density_sum = density_1d
        else:
            density_sum += density_1d

    return radius, density_sum/(end-start)



def DenProfStarCentred(data, csv, rmax, bins):
    """ Calculates the 1d surface density profile centred on each binary star
    takes the athena .athdf file and post-process csv with star positions on"""

    
    df_single = pd.DataFrame()
    orbit = data['Time']/(np.pi*2)
    idx = np.where((csv['orbit'] < orbit+0.01) & (orbit < csv['orbit']))[0]
    x, y = np.zeros(2), np.zeros(2)
    
    x[0], x[1], = np.mean(csv['x1'][idx]), np.mean(csv['x2'][idx])
    y[0], y[1], = np.mean(csv['y1'][idx]), np.mean(csv['y2'][idx])
    xx, yy = np.meshgrid(data['x1v'], data['x2v'], indexing= 'xy')
    density = data['rho'].flatten()
    
    for star in range(2):
        x_centred, y_centred = xx-x[star], yy-y[star]
        radius_centred = np.sqrt(x_centred*x_centred + y_centred*y_centred).flatten()
        
        mean_density = binned_statistic(radius_centred, density, statistic='mean',bins = bins, range=(0,rmax))
        density_profile = mean_density.statistic

        bin_edges = mean_density.bin_edges
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        star_df = pd.DataFrame({
            'radius': bin_centres,
            f'star_density_{star}': density_profile
        })
        
        if df_single.empty:
            df_single = star_df
        else:
            df_single = pd.merge(df_single, star_df, on='radius', how='outer')
    return df_single


def DenProfStarMean(dir_loc, file_name, file_ids, average_over, csv, rmax=50, bins=1000):
    """ Calculates the 1d surface density profile with respect to both stars position.
    Used to calculate the sizes of the mini-discs/circumprimary discs around each binary component.
    dir_loc - string, location of directory with athena .athdf outputs
    file_name - file name of run, typically binary_wind
    file_ids - size 2, [file number start, file number end] 
    average_over - specify how many orbits/files to calculate a mean value over
    csv - pandas dataframe of csv file with star positions in.
    rmax - radius to calculate to, 0 -> rmax. Default: whole disc domain (0-50)
    bins - bins to average over. Default: 1000 bins
    Returns nothing but saves single and averaged dataframes.
    """
    # Create required dirs if they don't already exist
    os.makedirs('rad-prof', exist_ok=True)
    os.makedirs('rad-prof/all', exist_ok=True)
    start_file, end_file = file_ids

    # Loop over files in steps of specified value to average over
    for start in range(start_file, end_file, average_over):
        end = start+average_over
        df_sum = pd.DataFrame()

        #Sum over individual files within the to-average range
        for i in range(start, end+1, 1):       
            data = athena_read.athdf(dir_loc + file_name + '.prim.' + str(i).zfill(5) + '.athdf')

            # Get radial profiles for both stars and save as a df
            df_single = DenProfStarCentred(data, csv, rmax, bins)
            # Save to file
            df_single.to_csv('rad-prof/all/den-1d.' + str(i) + '.csv')

            # Sum single df ready to average
            if df_sum.empty:
                df_sum = df_single.copy()
            else:
                for star in range(2):
                    df_sum[f'star_density_{star}'] += df_single[f'star_density_{star}']
    


        # Calculate the mean values in the average_over range
        df_mean = df_sum.copy()
        for star in range(2):
            df_mean[f'star_density_{star}'] /= (end - start + 1)
            
        # rename columns   
        df_mean.rename(columns={
            'star_density_0': 'mean_den_1',
            'star_density_1': 'mean_den_2'
        }, inplace=True)
        
        # Save mean radial profiles to file
        df_mean.to_csv('rad-prof/mean-den-1d.' + str(start) + '-' + str(end) + '.csv')


def MinidiscParams(df):
    """ Calculates density max, size and total mass of mini-discs.
    Takes dataframe of 1d surface density profile centred on stars (output of DenProfStarCentred()).
    Returns peak density, minimum (~size of mini-disc), and total integrated mass.
    """
    
    first_peak, first_min = {}, {}
    total_mass = {}
    for star in range(2):
        den = df[f'mean_den_{star+1}']
        rad = df.radius
        
        peaks, _ = find_peaks(den)
        minima, _ = find_peaks(-den)
        
        first_peak[star] = [rad[peaks[0]], den[peaks[0]]]
        first_min[star] = [rad[minima[0]], den[minima[0]]]
        
        mask = rad <= first_min[star][0]
        total_mass[star] = simpson(y=den[mask], x=rad[mask])
    return list(first_peak.values()), list(first_min.values()), list(total_mass.values())
    