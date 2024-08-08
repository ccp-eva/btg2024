from utils import *

def ti_to_image(sourceframe, minValue=25, maxValue=38):
    """
    Apply thermal calibration to a frame and scale the values between 0 and 255 uint8 type for image processing.
    The inverse option allows invertion of the scale.
    """
    # Apply calibration
    outputframe = (sourceframe - minValue)/(maxValue - minValue)
    # Scale to 0-255
    outputframe = outputframe*255
    # Clip values
    outputframe[outputframe < 0] = 0
    outputframe[outputframe > 255] = 255
    # Convert to uint8
    outputframe = np.array(outputframe, dtype='uint8')
    outputframe = cv2.applyColorMap(outputframe, cv2.COLORMAP_JET)
    colormap_with_legend = generate_colormap_with_legend(minValue, maxValue, width=outputframe.shape[1]//15, height=outputframe.shape[0], fontsize=15)
    return np.concatenate((outputframe, colormap_with_legend), axis=1)

def main():
    # Load data
    ti_zip_file = os.path.join("data","1-example.zip")
    fps = 30
    save_folder = os.path.join("data_npy",os.path.basename(ti_zip_file).split('.')[0])
    # Create save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    np_files = ti_extraction(ti_zip_file, save_folder)

    ## TODO:
    ## Loop over the np_files and convert each to an image using np.load and ti_to_image functions
    ## Display the image using cv2.imshow
    ## Select a ROI using cv2.selectROI
    ## Track the min temperature in the ROI using np.min
    ## Plot the min temperatures using matplotlib
    ## Calculate the frequency from the min temperatures
    ## Plot the temperature with peaks and the frequency
    ## Improve the frequency calculation using a Gaussian filter and plot the results

    # Variable to store min temperatures
    min_temps = []

    # Loop over the np_files
    for idx, np_file in enumerate(np_files):
        # Load thermal image
        ti_image = np.load(np_file)

        # Convert to image
        ti_vis = ti_to_image(ti_image)
        
        # First loop we select a ROI
        if idx == 0:
            roi = cv2.selectROI('Thermal Image', ti_vis)
            cv2.destroyAllWindows()
            print('ROI selected: ', roi)
        
        # Draw ROI
        cv2.rectangle(ti_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (255,255,255), 2)

        # Draw min temperature in ROI
        min_temp = np.min(ti_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
        cv2.putText(ti_vis, 'Min Temp: %.2f' % (min_temp), (roi[0], roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


        # Track min temperature in ROI
        min_temps.append(min_temp)

        # Display image with ROI using correct frame rate
        # if idx > 0:
        #     if time.time() - show_time < 1/fps:
        #         time.sleep(1/fps - (time.time() - show_time))
        # show_time = time.time()
        cv2.imshow('Thermal Image', ti_vis)
        cv2.waitKey(10)

        # Print progress
        if idx+1 % 100 == 0:
            print('Progress: ', idx+1, '/', len(np_files))
    
    # Close all windows
    cv2.destroyAllWindows()

    # # Convert min_temps to numpy array
    # min_temps = np.array(min_temps)

    # # Plot min temperatures
    # plt.plot(min_temps)
    # plt.xlabel('Frame')
    # plt.ylabel('Min Temperature')
    # plt.title('Min Temperature in ROI')
    # plt.show()
    # plt.close()

    # Apply a Gaussian filter to the min temperatures
    min_temps_gauss = gaussian_filter(min_temps, sigma=fps/3)
    
    # Calculate the frequency from the min temperatures
    bpm, peaks = get_bpm_from_peaks(min_temps_gauss, fps=30, freq_min=5, freq_max=60, height=0)

    # Create subplots
    # First subplot is temperature before gaussian filtering
    fig, axs = plt.subplots(3)
    axs[0].plot(min_temps)
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Min Temperature')
    axs[0].set_title('Min Temperature in ROI')

    # Second subplot is temperature after gaussian filtering with peaks
    axs[1].plot(min_temps_gauss)
    axs[1].plot(peaks, min_temps_gauss[peaks], "x")
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Min Temperature')
    axs[1].set_title('Min Temperature in ROI (Gaussian Filtered)')

    # Third subplot is the frequency
    axs[2].plot(bpm)
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('BPM')
    axs[2].set_title('Breath Rate')

    # Tight layout
    plt.tight_layout()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()