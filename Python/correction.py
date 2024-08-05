import cv2

def display_squarred_image(image_path, wait=0):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height == width:
        cv2.imshow('image', image)
    else:
        if height > width:
            max_dimension = height
            min_dimension = width
        else:
            max_dimension = width
            min_dimension = height
        if wait ==0:
            print('Press any key to move the image')
        for i in range(max_dimension - min_dimension):
            if height > width:
                tmp_image = image[i:min_dimension + i, :]
            else:
                tmp_image = image[:, i:min_dimension + i]
            cv2.imshow('image', tmp_image)
            cv2.waitKey(wait)
            print('Progress: ', i, '/', max_dimension - min_dimension)
    print('Press any key to close the image')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1