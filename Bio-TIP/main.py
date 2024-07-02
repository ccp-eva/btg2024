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
    ## Loop over the np_files and convert each to an image
    ## using the ti_to_image function
    ## Display the image using matplotlib


if __name__ == "__main__":
    main()