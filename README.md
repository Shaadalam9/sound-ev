# Multi-pedestrian interaction with automated vehicle
Framework for the analysis of different sounds emitted by an automated car while pedestrian is crossing the road.

## Setup
Tested with Python 3.9.20. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows):

**Step 1:**  

Clone the repository
```command line
git clone https://github.com/Shaadalam9/sound-ev
```

**Step 2:** 

Install Dependencies
```command line
pip install -r requirements.txt
```

**Step 3:**

Ensure you have the required datasets in the data/ directory.

**Step 4:**

Run the code:
```command line
python3 analysis.py
```

## Input
The simulator takes input from Meta Quest controller.


## Output
The simulator supports giving output to both a computer screen and a head-mounted display (HMD). It has been tested with Meta Quest 3.

## Installation
The simulator was tested on Windows 11 and macOS Sequoia 15.1.1. All functionality is supported by both platforms. However, support for input and output devices was tested only on Windows 11.

After checking out this project, launch Unity Hub to run the simulator with the correct version of Unity (currently **2022.3.5f1**).

## Configuration

The project is divided into three main scenes, located in the directory: `sound-unity/Assets/Scenes`. These scenes are structured as follows:

1. **MainMenu**:  
   - This is the introductory scene of the experiment. Participants are welcomed and briefed on the study objectives and procedures.

2. **Environment**:  
   - This scene contains all the experimental trials. Participants interact with the environment to record their willingness to cross the road under different conditions.

3. **EndMenu**:  
   - The final scene serves as a thank-you screen, providing closure to the experiment.

## Setup Instructions

### **Adjusting Participant Height**:
   Follow these steps to properly adjust the participant's height:

1. **Open the Settings Menu**
   - Access the **Settings** menu.
   - ![Quick Settings](ReadmeFiles/setting_nav.jpg)

2. **Navigate to Physical Space**
   - Select the **Physical Space** option.
   - ![Physical Space](ReadmeFiles/physical_space.png)

3. **Clear Boundary and Space Scan History**
   - Choose the option **Clear Boundary and Space Scan History** to reset any previous environment settings.
   - ![Clear Boundary](ReadmeFiles/clear_boundary.png)

4. **Set Floor Level**
   - Click on **Set Floor Level** and adjust it to ensure the floor level is correctly calibrated.
   - ![Set Floor](ReadmeFiles/floor_level.png)

5. **Set Boundary with Participant**
   - Return to the **Quick Settings** menu and have the participant wear the headset.
   - Instruct the participant to select **Boundary** from the menu.
   - ![Boundary](ReadmeFiles/boundary.png)

6. **Ensure Correct Orientation**
   - Guide the participant to look down at their shoes to verify correct alignment and orientation.

7. **Confirm Boundary**
   - Ask the participant to press **Confirm Boundary** to finalize the setup.
   - ![Confirm Boundary](ReadmeFiles/stationary_boundary.png)

8. **Hand Over Control**
   - Once the boundary is confirmed, the experimenter can take back the controller and headset.



### **Participant ID Setup**:  
   Before starting the experiment, you must configure the participant's unique ID. This can be done by:
   - Navigating to the `ConditionController` in the **Hierarchy** tab.
   - Filling in the `Write File Name` field with the participant's unique name or number.

   Once this is complete, click the **Play** button to enter play mode.

   ![Setup Instructions](ReadmeFiles/fig_1.png)


### **Connecting to Quest Link and Launching Unity**:
Set up the Unity environment and connect it to the Quest using the following steps:

1. **Launch Unity**
   - Ensure that Unity is open on your computer and the **Main Menu** environment is loaded.

2. **Open Quest Link**
   - Navigate to the **Settings** menu and select Quest Link via Meta Quest.
   - ![Quest Link](ReadmeFiles/Quest_link.png)

3. **Add Desktop Panel**
   - In Quest Link, click on **Add Desktop Panel**.
   - ![Add Desktop Panel](ReadmeFiles/add_desktop_panel.png)

4. **Select Unity**
   - Choose Unity from the available options in the desktop panel.
   - ![Select Unity](ReadmeFiles/select_unity.png)

5. **Verify Main Menu Scene**
   - Ensure the Main Menu scene is visible in the Quest.
   - Press the **Play** button in Unity to start the experiment.
   - ![Play Button](ReadmeFiles/play_button.png)

6. **Participant Instructions**
   - Have the participant wear the headset.
   - Allow them to read the on-screen instructions before proceeding with the experiment.

---

## Experiment Interaction

### Recording Willingness to Cross the Road

Participants record their willingness to cross the road using the **trigger button** on the controller. The interaction process is illustrated in the image below:  

![Trigger Button](ReadmeFiles/trigger.jpeg)

- **How to Record**:  
  Press the trigger button when you feel safe to cross the road.

---

## Data Collection

The experiment captures and saves data for analysis in the `data` folder. The folder structure is organized as follows:

1. **Participant-Specific Folder**:  
   - A unique folder is created for each participant, named after their unique ID with a timestamp.

2. **Contents of Each Folder**:  
   - **Mapping File**:  
     This file contains the sequence of trials. It can be used to link trial data with corresponding scenarios (using `video_id`).

   - **Trial Data**:  
     - For each trial, a CSV file is generated in the format `{Participant_ID}_{video_id}`.  
     - These files store the physical movements of the Head-Mounted Display (HMD) and controllers during the trial.

   - **Response Data**:  
     A separate CSV file is generated to store responses to the questions asked between trials.


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com
