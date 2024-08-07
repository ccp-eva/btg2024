## import
## try to import this or antigravity
# import this
# import antigravity

## print() function
## print Hello World!
# print('Hello World!')

## function definition and indentation:
## define your own function and call it
# def my_function():
#     global tmp
#     tmp = 1
#     print('Hello from my function!')
#     # Print tmp
#     print("tmp is", tmp)

'''my_function()
print("tmp is", tmp)
tmp += 1
print("tmp is", tmp)
tmp = tmp + 1
print("tmp is", tmp)
'''

# input_user = input("Enter a number: ")
# print("You entered", input_user)
# input_user = int(input_user)
# input_user+=1
# print("Now it's", input_user)

## Comments:
## use # to comment your code or triple quotes 
## use triple quotes to comment multiple lines

## local VS global variables:
## understand the difference between local and global variables
## debug mode: use vs code debug mode to understand the concept or pdb

## types of variables: use type() function to check the type of variable your are working with

## input from user: use input() function

## list & numpy arrays
## create a list and a numpy array

# l = [1, 2, 3, 4, 5]
# l2 = []
# print("list l is", l)
# for i in l:
#     # add only even numbers
#     if i % 2 == 0: # remainder of i divided by 2
#         print(i)
#         l2.append(2*i)
#     else:
#         l2.append(i)

# print("list l2 is", l2)
# import numpy as np
# print("l2 - l:", np.array(l2) - np.array(l))

# import numpy as np
# a = np.array(l)

## loops and conditions

## create a file utils.py and then:
## 0. In Run and Debug, create a launch.json file and add the following configuration: "cwd": "${fileDirname}",
## 1. write a function in utils called display_squared_image, taking as input an image path.
## 2. the function should read the image,
## 3. if the image is square: display the image
## 4. if not: display along the longest dimension a squared version of the image. (slowly or at the user convenience)
## 5. call the function in main.py with btg_2022.png and earth_temperature_timeline.png.
## If it freezes, try using ctrl + c (preferred) or ctrl + z to stop the execution.
from utils import *

image_path = 'Python/earth_temperature_timeline.png'
display_squarred_image(image_path, wait=5)
