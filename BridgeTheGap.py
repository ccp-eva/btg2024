import sys
sys.path.insert(1, 'Bio-TIP')
from utils import *

logo = cv2.imread('attachments/logo_color.png')
bridge = cv2.imread('attachments/bridge.png', cv2.IMREAD_UNCHANGED)

print('Seletect which part of the bridge should be bridge:')

roi = cv2.selectROI('Bridge The Gap', logo)

# Bridge The Gap
bridge = cv2.resize(bridge, (roi[2], roi[3]))
logo[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = logo[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] * np.stack((1-bridge[:,:,3]//255,)*3, axis=-1)

# Show the gap bridged
cv2.imshow('Bridge The Gap', logo)
print('You have bridged the gap!\nPress any key to close the window.')
cv2.waitKey(0)
cv2.destroyAllWindows()
print('The Python Test has been completed successfully!')