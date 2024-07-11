from utils import *

def ti_to_image(sourceframe, minValue=20, maxValue=40):
    ## TODO: Implement this function
    outputframe = sourceframe.copy()
    ## Do not modify below here
    # Apply colormap
    outputframe = cv2.applyColorMap(sourceframe, cv2.COLORMAP_JET)
    # Generate colormap with legend
    colormap_with_legend = generate_colormap_with_legend(minValue, maxValue, width=outputframe.shape[1]//15, height=outputframe.shape[0], fontsize=15)
    # Concatenate the colormap with the output frame
    return np.concatenate((outputframe, colormap_with_legend), axis=1)

def main():
    # Load data
    ti_zip_file = os.path.join("data","1-example.zip")
    save_folder = os.path.join("data_npy",os.path.basename(ti_zip_file).split('.')[0])
    # Create save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    np_files = ti_extraction(ti_zip_file, save_folder)

    ## TODO:
    ## 1. Loop over the np_files
    ## 2. Convert each numpy file (temperature) to an image using ti_to_image function (to fill)
    ## 3. Display the image using matplotlib
    ## 4. Select a ROI using cv2.selectROI
    ## 5. Track the min temperature in the ROI using np.min
    ## 6. Plot the min temperatures using matplotlib
    ## 7. Calculate the frequency from the min temperatures using the function get_bpm_from_peaks
    ## 8. Plot the temperature with peaks and the frequency
    ## 9. Improve the frequency calculation using a Gaussian filter and plot the results


if __name__ == "__main__":
    main()