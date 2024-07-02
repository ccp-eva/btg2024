import zipfile
import natsort
import numpy as np
import os
import time
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
import pdb

print('utils.py loaded')

def get_bpm_from_peaks(signal, fps=30, freq_min=40, freq_max=240, height=0):
    bpm_array = freq_min*np.ones(len(signal))
    min_distance = int(60/freq_max*fps)
    peaks, _ = find_peaks(signal, distance=min_distance, height=height)
    if len(peaks)>0:
        freq_values = fps/(peaks[1:] - peaks[:-1])*60
        peaks_copy = np.copy(peaks)
        peaks_copy[0] = 0
        peaks_copy[-1] = len(signal)
        for idx, freq_value in enumerate(freq_values):
            start = peaks_copy[idx]
            end = peaks_copy[idx+1]
            bpm_array[start:end] = max(freq_value, freq_min)
    return bpm_array, peaks

def ti_extraction(zip_file, save_folder):
    '''
    input:
        - zip_file (str) zip file to extract ti from
        - save_folder (str) location to save the np-array ti
    output:
        - np_files (list of str) paths to np array containing ti
    '''
    # Create the list of ASC files to be processed / robust to txt asc and zip files
    zip_file = zipfile.ZipFile(zip_file, 'r')
    ti_files = [f.filename for f in zip_file.infolist() if (f.filename.endswith(('.asc','.txt','.npy'))) and not os.path.basename(f.filename).split('/')[-1].startswith('.')]
    list_saved_files = os.listdir(save_folder)
    # Check if np-arrays were already saved
    if len(ti_files) == len(list_saved_files):
        print('TI extraction was already done and located in %s' % (save_folder))
        np_files = [os.path.join(save_folder,f) for f in list_saved_files]
        # natural sorting more robust
        np_files = natsort.natsorted(np_files)
        # np_files.sort()
    else:
        start_time = time.time()
        np_files = []
        # natural sorting more robust
        ti_files = natsort.natsorted(ti_files)
        # ti_files.sort()
        for idx, asc_file in enumerate(ti_files):
            if idx % 100 == 0:
                print('ASC values extraction of %s (%d/%d)' % (zip_file,idx,len(ti_files)))
            # Read asc file, calibrate and convert to BGR
            ti_image = ti_to_array(zip_file.read(asc_file))
            np_file = os.path.join(save_folder, os.path.basename(asc_file)+'.npy')
            np.save(np_file, ti_image)
            np_files.append(np_file)
        print('ASC values extraction of %s (%d/%d) completed in %ds' % (zip_file, idx, len(ti_files), time.time()-start_time))
    # Close zip file
    zip_file.close()
    return np_files

def ti_to_array(sourceFileName):
    """
    Read ti from source file and return a ti in numpy array.
    Robust to txt and asc files.
    Must be tried with different characters combination.
    """
    if isinstance(sourceFileName, bytes):
        content = str(sourceFileName).replace(',','.').replace('\\t',' ').replace('\\r','').replace('\'','').split('\\n')
    elif isinstance(sourceFileName, str):
        if sourceFileName.endswith('.npy'):
            return np.load(sourceFileName)
        with open(sourceFileName, errors='replace') as f:
            content = f.read().replace(',','.').replace('\t',' ').split('\n')
    else:
        raise Exception('File format %s not supported' % (sourceFileName))
    list_of_values = []
    for line in content:
        line_array = np.fromstring(line, dtype=float, sep=' ')
        # Check if the line is not empty and has at least 100 values
        if len(line_array) < 100:
            continue
        list_of_values.append(line_array)
    # Check the consitence of the data
    if np.std([len(line) for line in list_of_values]) > 0:
        raise Exception('Data in file %s is not consistent' % (sourceFileName))
    return np.array(list_of_values)

def generate_colormap_with_legend(min_val=0, max_val=1, width=50, height=720, fontsize=12):
    # Create an image with the jet colormap
    colormap = np.linspace(1, 0, height).astype(np.float32)
    colormap = np.repeat(colormap[:, np.newaxis], width, axis=1)
    colormap_jet = cv2.applyColorMap((colormap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Calculate the interval for the legend based on height and fontsize
    num_steps = height // (fontsize * 3)  # Roughly 3 times the fontsize for each label
    step_value = (max_val - min_val) / num_steps
    # Find a step value with fewer digits
    rounded_step_value = round(step_value, -int(np.floor(np.log10(step_value))))
    
    # Draw the legend on the colormap
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(num_steps + 1):
        value = min_val + i * rounded_step_value
        # Ensure the value does not exceed max_val
        value = min(value, max_val)
        text = f"{value:.2f}"
        position = (5, int(height - (i * (height / num_steps))))
        cv2.putText(colormap_jet, text, position, font, fontsize / 30, (255, 255, 255), 1, cv2.LINE_AA)
    return colormap_jet