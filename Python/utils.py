import cv2

def display_squarred_image(image_path, wait=0):
    # Read image
    img = cv2.imread(image_path)

    # Get image dimensions
    height, width, channels = img.shape
    print('Image dimensions:', height, 'x', width, 'x', channels)

    # Condition to check if image is square
    if height == width:
        # Display image
        print('Displaying image')
        cv2.imshow('Image', img)
    else:
        # Check which dimension is the longest
        if width > height:
            print('Width is the longest dimension')
            width_longest = True
            longest_dim = width
            shortest_dim = height
        else:
            print('Height is the longest dimension')
            width_longest = False
            longest_dim = height
            shortest_dim = width
        # Loop along the longest dimension
        for i in range(longest_dim-shortest_dim+1):
            # Using condition, crop the image along the longest dimension
            if width_longest:
                print('Cropping along width')
                img_show = img[:, i:shortest_dim+i]
            else:
                print('Cropping along height')
                img_show = img[i:shortest_dim+i, :]
            if wait == 0:
                print('Press any key to continue')
            
            # Display the cropped image
            cv2.imshow('Image', img_show)
            # Wait key
            cv2.waitKey(wait)
            print('Progress: ', i+1, '/', 10)
    # Wait for user to close the image
    print('Press any key to close the image')
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()
    return 1
     