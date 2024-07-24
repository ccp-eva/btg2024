import cv2
import numpy as np
from tkinter import filedialog
import pandas as pd
import pathlib

#### Basic functions
def getValue(fileID,variable): # Get a value (variable) from a trajectory file (fileID) using PagesJaunes 
    return pj[pj["traj"]==fileID][variable].to_list()[0] 


def angle_between(v1, v2): # Calculate the angle between 2 vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return abs((np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))-180)) 

def norm_trajectory(file,likelihood_crit = 0.95): #  Perspective correction of the trajectory

    res = pd.read_hdf(traj_directory + file)
    res.columns = res.columns.droplevel(0) 
    pos=res["pointer"]      

    #### Data Filtering ####
    pos.loc[pos["likelihood"] < likelihood_crit, "x"] = np.nan   
    pos.loc[pos["likelihood"] < likelihood_crit, "y"] = np.nan 
    pos.interpolate(method="linear", inplace= True)

    # Get from PagesJaunes the coordinates (pix) from the Newtwork loc trained on the videos
    p1 = [float(getValue(file,"p1x")),float(getValue(file,"p1y"))]
    p2 = [float(getValue(file,"p2x")),float(getValue(file,"p2y"))]
    p4 = [float(getValue(file,"p4x")),float(getValue(file,"p4y"))]
    p5 = [float(getValue(file,"p5x")),float(getValue(file,"p5y"))]

    # List of the 4-reference locations coordinates (origin: videos, Network loc)
    pts1 = np.float32([p1,p2,p4,p5]) 
    # List of the 4-reference locations absolute coordinates (origin: touchscreen program)
    pts2 = np.float32([[0, 0], [100, 0],[0,100], [100, 100]])

    ## Calculate Perspective Transformation Matrix
    m = cv2.getPerspectiveTransform(pts1, pts2) 

    # Apply the perspective correction
    x_t =(m[0,0]* pos["x"] + m[0,1]* pos["y"] + m[0,2])/(m[2,0]* pos["x"] + m[2,1]* pos["y"] + m[2,2])
    y_t =(m[1,0]* pos["x"] + m[1,1]* pos["y"] + m[1,2])/(m[2,0]* pos["x"] + m[2,1]* pos["y"] + m[2,2])

    return x_t,y_t 


## Calculate the changes of direction in the trajectory
def direction_change(x,y,file,angle_threshold=90): 
    ##### Initialise the data frame #####
    input_headers = [
        "subject","Group_L","date","Session","block","trials_count","set_num",
        "session_type","RSI","seq_type","seq_num","pos_1","pos_2",
        "ObjClicked_1","ObjClicked_2","ObjClicked_RSI_1","ObjClicked_RSI_2"
    ]
        
    output_headers = ["loc","change_bin","n_change","trial_stage"]

    res = pd.DataFrame(columns=input_headers+output_headers)

    for header in input_headers :
        res[header]= [getValue(file,header)]*30

    res["loc"] = ["loc_1","loc_2","loc_3","loc_4","loc_5","loc_6"]*5
    res["trial_stage"] = ["RSI_1"]*6 + ["RT_1"]*6 + ["RSI_2"]*6 + ["RT_2"]*6 + ["POST"]*6 
    res["change_bin"] = [0]*30
    res["n_change"] = [0]*30

    # Time code of each trial stage
    timecode = {
        "RSI_1":int(float(getValue(file,'RSI'))*fps),
        "RT_1":int(float(getValue(file,'RT_1'))*fps),
        "RSI_2":int(float(getValue(file,'RSI'))*fps),
        "RT_2":int(float(getValue(file,'RT_2'))*fps) 
    }
    
    #  Center coordinates of the six locations (x,y)
    locations = {
        "loc_1": (0,0),
        "loc_2": (-100,0),
        "loc_3": (200,0),
        "loc_4": (0,100),
        "loc_5": (100,100),
        "loc_6": (200,100),    
    }

    def current_stage(frame): 
        time = 0
        for stage in timecode : 
            time += timecode[stage]
            if frame < time:
                return stage
        return "POST"

    
    def in_hitbox(x,y,loc_center,hitbox_size=100): ## FILL THIS FUNCTION ##  This function return if x and y are within loc hitbox
        x_inside = ___FILL_THE_GAP___
        y_inside = ___FILL_THE_GAP___
        return x_inside and y_inside
    
    for frame in range(len(x)-3) :
        
        stage = current_stage(frame)

        v1 = [x[frame]-x[frame+1],y[frame]-y[frame+1]]
        v2 = [x[frame+1]-x[frame+2],y[frame+1]-y[frame+2]]
        v3 = [x[frame+2]-x[frame+3],y[frame+2]-y[frame+3]]

        angle_short = angle_between(v2,v3)
        angle_wide = angle_between(v1,v3)
        
        if (angle_short < angle_threshold) or (angle_wide < angle_threshold):
            direction_change = True
        else : 
            direction_change = False
        
        if ___FILL_THE_GAP___ :
            for loc in locations:
                # Retrieve the line in the dataframe corresponding to the current stage and current location
                current_res= (res["trial_stage"] == ___FILL_THE_GAP___) & (res["loc"] == ___FILL_THE_GAP___)
                if in_hitbox(x[frame+1],y[frame+1],loc): 
                    res.loc[current_res, "n_change"]  +=  1
                    res.loc[current_res,["change_bin"]] = 1

    return res



main_directory = str(pathlib.Path(__file__).parent.resolve())+"/"
pj = pd.read_csv(main_directory + 'PagesJaunes.csv')
traj_directory = main_directory + "Alex/H5/"

files = filedialog.askopenfilenames() # select trajectories h5
res_list = []
fps = 25

___FILL_THE_GAP___ = "FILL"

for file in files:
        file = file.split("/")[-1]
        x,y = norm_trajectory(file)
        res = direction_change(x,y,file)
        res_list.append(res)

res_total = pd.concat(res_list)

res_total.to_csv(main_directory + "Results/direction_change.csv",index=False)


