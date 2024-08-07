import os
import glob
import I2MC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Import data from Tobii TX300
# =============================================================================

def tobii_TX300(fname, res=[1920,1080]):

    # Load all data
    raw_df = pd.read_csv(fname, delimiter=',')
    df = pd.DataFrame()
    
    # Extract required data
    df['time'] = raw_df['time']
    df['L_X'] = raw_df['L_X']
    df['L_Y'] = raw_df['L_Y']
    df['R_X'] = raw_df['R_X']
    df['R_Y'] = raw_df['R_Y']
    
    
    ###
    # Sometimes we have weird peaks where one sample is (very) far outside the
    # monitor. Here, count as missing any data that is more than one monitor
    # distance outside the monitor.
    
    # Left eye
    lMiss1 = (df['L_X'] < -res[0]) | (df['L_X']>2*res[0])
    lMiss2 = (df['L_Y'] < -res[1]) | (df['L_Y']>2*res[1])
    lMiss  = lMiss1 | lMiss2 | (raw_df['L_V'] > 1)
    df.loc[lMiss,'L_X'] = np.NAN
    df.loc[lMiss,'L_Y'] = np.NAN
    
    # Right eye
    rMiss1 = (df['R_X'] < -res[0]) | (df['R_X']>2*res[0])
    rMiss2 = (df['R_Y'] < -res[1]) | (df['R_Y']>2*res[1])
    rMiss  = rMiss1 | rMiss2 | (raw_df['R_V'] > 1)
    df.loc[rMiss,'R_X'] = np.NAN
    df.loc[rMiss,'R_Y'] = np.NAN

    return(df)



#%% Preparation

# Find the files
current_path = os.path.dirname(os.path.realpath(__file__))
data_files = glob.glob(os.path.join(current_path,'Files','DATA','RAW','**','*.csv'), recursive = True) # find all the files

# define the output folder
output_folder = os.path.join(current_path,'Files','DATA','i2mc_output') # define folder path\name

# Create the outputfolder
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# NECESSARY VARIABLES

opt = {}
# General variables for eye-tracking data
opt['xres']         = 1920.0                # maximum value of horizontal resolution in pixels
opt['yres']         = 1080.0                # maximum value of vertical resolution in pixels
opt['missingx']     = np.NAN                # missing value for horizontal position in eye-tracking data (example data uses -xres). used throughout the algorithm as signal for data loss
opt['missingy']     = np.NAN                # missing value for vertical position in eye-tracking data (example data uses -yres). used throughout algorithm as signal for data loss
opt['freq']         = 300.0                 # sampling frequency of data (check that this value matches with values actually obtained from measurement!)

# Variables for the calculation of visual angle
# These values are used to calculate noise measures (RMS and BCEA) of
# fixations. The may be left as is, but don't use the noise measures then.
# If either or both are empty, the noise measures are provided in pixels
# instead of degrees.
opt['scrSz']        = [50.9174, 28.6411]    # screen size in cm
opt['disttoscreen'] = 65.0                  # distance to screen in cm.

# Options of example script
do_plot_data = True # if set to True, plot of fixation detection for each trial will be saved as png-file in output folder.
# the figures works best for short trials (up to around 20 seconds)

# =============================================================================
# OPTIONAL VARIABLES
# The settings below may be used to adopt the default settings of the
# algorithm. Do this only if you know what you're doing.

# # STEFFEN INTERPOLATION
opt['windowtimeInterp']     = 0.1                           # max duration (s) of missing values for interpolation to occur
opt['edgeSampInterp']       = 2                             # amount of data (number of samples) at edges needed for interpolation
opt['maxdisp']              = opt['xres']*0.2*np.sqrt(2)    # maximum displacement during missing for interpolation to be possible

# # K-MEANS CLUSTERING
opt['windowtime']           = 0.2                           # time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
opt['steptime']             = 0.02                          # time window shift (s) for each iteration. Use zero for sample by sample processing
opt['maxerrors']            = 100                           # maximum number of errors allowed in k-means clustering procedure before proceeding to next file
opt['downsamples']          = [2, 5, 10]
opt['downsampFilter']       = False                         # use chebychev filter when downsampling? Its what matlab's downsampling functions do, but could cause trouble (ringing) with the hard edges in eye-movement data

# # FIXATION DETERMINATION
opt['cutoffstd']            = 2.0                           # number of standard deviations above mean k-means weights will be used as fixation cutoff
opt['onoffsetThresh']       = 3.0                           # number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward at fixation ends to refine their placement and stop algorithm from eating into saccades
opt['maxMergeDist']         = 30.0                          # maximum Euclidean distance in pixels between fixations for merging
opt['maxMergeTime']         = 30.0                          # maximum time in ms between fixations for merging
opt['minFixDur']            = 40.0                          # minimum fixation duration after merging, fixations with shorter duration are removed from output

#%% Run I2MC

for file_idx, file in enumerate(data_files):
    print('Processing file {} of {}'.format(file_idx + 1, len(data_files)))

    # Extract the name form the file path
    name = os.path.splitext(os.path.basename(file))[0]
    
    # Create the folder the specific subject
    subj_folder = os.path.join(output_folder, name)
    if not os.path.isdir(subj_folder):
       os.mkdir(subj_folder)
       
    # Import data
    data = tobii_TX300(file, [opt['xres'], opt['yres']])

    # Run I2MC on the data
    fix,_,_ = I2MC.I2MC(data,opt)

    ## Create a plot of the result and save them
    if do_plot_data:
        # pre-allocate name for saving file
        save_plot = os.path.join(subj_folder, name+'.png')
        f = I2MC.plot.data_and_fixations(data, fix, fix_as_line=True, res=[opt['xres'], opt['yres']])
        # save figure and close
        f.savefig(save_plot)
        plt.close(f)

    # Write data to file after make it a dataframe
    fix['participant'] = name
    fix_df = pd.DataFrame(fix)
    save_file = os.path.join(subj_folder, name+'.csv')
    fix_df.to_csv(save_file)

print('The extraction of fixations is done!')