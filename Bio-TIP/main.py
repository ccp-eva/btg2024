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
    ti_zip_file = "data.zip"
    save_folder = "data_npy"
    np_files = ti_extraction(ti_zip_file, save_folder)

    ## TODO:
    ## Loop over the np_files and convert each to an image
    ## using the ti_to_image function
    ## Display the image using matplotlib


if __name__ == "__main__":
    main()