# Import some libraries from PsychoPy and others
import os
from psychopy import core, visual, sound, event 

# Winsize
winsize = (960, 540)

# Get path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

# create a window
win = visual.Window(size = winsize, fullscr=False, units="pix", pos =(0,30), screen=0)

#%% Load and prepare stimuli
stimuli_path = os.path.join(current_path,"Files","EXP","Stimuli")

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

#%% Trials

for trial in Trials:

    ### Present the fixation
    win.flip() # we flip to clean the window

    fixation.draw()
    win.flip()
    core.wait(1)  # wait for 1 second


    ### Present the cue
    cues[trial].draw()
    win.flip()
    core.wait(3)  # wait for 3 seconds


    ### Present the reward
    rewards[trial].draw()
    win.flip()
    sounds[trial].play()
    core.wait(2)  # wait for 1 second
    win.flip()    # we re-flip at the end to clean the window

    ### ISI
    clock = core.Clock() # start clock
    while clock.getTime() < 1:
        pass
      
    ### Check for closing experiment
    keys = event.getKeys() # collect list of pressed keys
    if 'escape' in keys:
        win.close()  # close window
        core.quit()  # stop study
        
win.close()
core.quit()


