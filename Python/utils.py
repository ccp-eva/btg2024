import cv2

def display_squarred_image(image_path, wait=0):
    # Read image

    # Get image dimensions

    # Condition to check if image is square
    if 1:
        # Display image
        print('Displaying image')
    else:
        # Check which dimension is the longest
        if 1:
            print('Width is the longest dimension')
        else:
            print('Height is the longest dimension')
        # Loop along the longest dimension
        for i in range(10):
            # Using condition, crop the image along the longest dimension
            if 1:
                print('Cropping along width')
            else:
                print('Cropping along height')
            if wait == 0:
                print('Press any key to continue')
            
            # Display the cropped image
            
            # Wait key
            cv2.waitKey(wait)
            print('Progress: ', i+1, '/', 10)
    # Wait for user to close the image
    print('Press any key to close the image')
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()
    return 1
     