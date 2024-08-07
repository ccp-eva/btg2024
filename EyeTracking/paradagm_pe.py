import os
import pandas as pd

# Import some libraries from PsychoPy
from psychopy import core, event, visual, prefs
# prefs.hardware['audioLib'] = ['PTB']
from psychopy import sound

import tobii_research as tr


#%% Functions

# This will be called every time there is new gaze data
def gaze_data_callback(gaze_data):
    global trigger
    global gaze_data_buffer
    global winsize

    if len(trigger)==0:
        ev = ''
    else:
        ev = trigger
        trigger=[]
    
    # Extract the data we are interested in
    t  = gaze_data.system_time_stamp / 1000.0
    lx = gaze_data.left_eye.gaze_point.position_on_display_area[0] * winsize[0]
    ly = winsize[1] - gaze_data.left_eye.gaze_point.position_on_display_area[1] * winsize[1]
    lp = gaze_data.left_eye.pupil.diameter
    lv = gaze_data.left_eye.gaze_point.validity
    rx = gaze_data.right_eye.gaze_point.position_on_display_area[0] * winsize[0]
    ry = winsize[1] - gaze_data.right_eye.gaze_point.position_on_display_area[1] * winsize[1]
    rp = gaze_data.right_eye.pupil.diameter
    rv = gaze_data.right_eye.gaze_point.validity
        
    # Add gaze data to the buffer 
    gaze_data_buffer.append((t,lx,ly,lp,lv,rx,ry,rp,rv,ev))
    
        
def write_buffer_to_file(buffer, output_path):

    # Make a copy of the buffer and clear it
    buffer_copy = buffer[:]
    buffer.clear()
    
    # Define column names
    columns = ['time', 'L_X', 'L_Y', 'L_P', 'L_V', 
               'R_X', 'R_Y', 'R_P', 'R_V', 'Event']

    # Convert buffer to DataFrame
    out = pd.DataFrame(buffer_copy, columns=columns)
    
    # Check if the file exists
    file_exists = not os.path.isfile(output_path)
    
    # Write the DataFrame to an HDF5 file
    out.to_csv(output_path, mode='a', index =False, header = file_exists)
    
    
    
#%% Load and prepare stimuli

# Winsize
winsize = (960, 540)

# create a window
win = visual.Window(size = winsize,fullscr=False, units="pix", pos =(0,30), screen=1)

# Load images
current_path = os.path.dirname(os.path.realpath(__file__))
stimuli_path = os.path.join(current_path, "Files","EXP","Stimuli")

# Load images
fixation = visual.ImageStim(win, image=os.path.join(stimuli_path,'fixation.png'), size = (200, 200))
circle   = visual.ImageStim(win, image=os.path.join(stimuli_path,'circle.png'), size = (200, 200))
square   = visual.ImageStim(win, image=os.path.join(stimuli_path,'square.png'), size = (200, 200))
winning   = visual.ImageStim(win, image=os.path.join(stimuli_path,'winning.png'), size = (200, 200), pos=(250,0))
losing  = visual.ImageStim(win, image=os.path.join(stimuli_path,'loosing.png'), size = (200, 200), pos=(-250,0))

# Load sound
winning_sound = sound.Sound(os.path.join(stimuli_path,'winning.wav'))
losing_sound = sound.Sound(os.path.join(stimuli_path,'loosing.wav'))

# List of stimuli
cues = [circle, square] # put both cues in a list
rewards = [winning, losing] # put both rewards in a list
sounds = [winning_sound,losing_sound] # put both sounds in a list

# Create list of trials in which 0 means winning and 1 means losing
Trials = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1 ]

# List of stimuli
cues = [circle, square] # put both cues in a list
rewards = [winning, losing] # put both rewards in a list
sounds = [winning_sound,losing_sound] # put both sounds in a list

# Create list of trials in which 0 means winning and 1 means losing
Trials = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1 ]

#%% Record the data

# Find all connected eye trackers
found_eyetrackers = tr.find_all_eyetrackers()

# We will just use the first one
if len(found_eyetrackers) == 0:
    print("No eye tracker found. Data will not be recorded.")
    Eyetracker = None
else:
    Eyetracker = found_eyetrackers[0]

    #Start recording
    Eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


# Create an empty list we will append our data to
gaze_data_buffer = []

for trial in Trials:

    ### Present the fixation
    win.flip() # we flip to clean the window

    fixation.draw()
    win.flip()
    trigger = 'Fixation'
    core.wait(1)  # wait for 1 second

    ### Present the cue
    cues[trial].draw()
    win.flip()
    if trial ==0:
        trigger = 'Circle'
    else:
        trigger = 'Square'
    core.wait(3)  # wait for 3 seconds

    ### Wait for saccadic latencty
    win.flip()
    core.wait(0.75)

    ### Present the reward
    rewards[trial].draw()
    win.flip()

    if trial ==0:
        trigger = 'Reward'
    else:
        trigger = 'NoReward'
    sounds[trial].play()
    core.wait(2)  # wait for 2 second

    ### ISI
    win.flip()    # we re-flip at the end to clean the window
    clock = core.Clock()
    if Eyetracker is not None:
        write_buffer_to_file(gaze_data_buffer, os.path.join('DATA','RAW','recording.csv')
    while clock.getTime() < 1:
        pass
    
    ### Check for closing experiment
    keys = event.getKeys() # collect list of pressed keys
    if 'escape' in keys:
        win.close()  # close window
        if Eyetracker is not None:
            Eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback) # unsubscribe eyetracking
        core.quit()  # stop study
      
win.close() # close window
if Eyetracker is not None:
    Eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback) # unsubscribe eyetracking
core.quit() # stop study
