# Electric vehicles emitting sound
Framework for the analysis of different sounds emitted by an automated car while pedestrian is crossing the road.

## Citation
If you use the simulator for academic work please cite the following papers:

> Bazilinskyy, P., Alam, M. S., & Martinez, R. M. (2025). Pedestrian crossing behaviour in front of electric vehicles emitting synthetic sounds: A virtual reality experiment. In 54th International Congress & Exposition on Noise Control Engineering (INTER-NOISE). São Paulo, Brazil. https://doi.org/10.3397/IN_2025_1076086

> Bazilinskyy, P., Alam, M. S., & Martinez, R. M. (2025). Psychoacoustic assessment of synthetic sounds for electric vehicles in a virtual reality experiment. In 11th Convention of the European Acoustics Association (Euronoise). Málaga, Spain. https://arxiv.org/abs/2510.25593

## Analysis
Analysis can be started by running python `sound-ev/analysis.py`. A number of CSV files used for data processing are saved in `sound-ev/_output/`. Visualisations of all data are saved in `sound-ev/_output/figures/`.

## Keypress data
[![Keypress](figures/all_videos_kp_slider_plot_TriggerValueRight.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/all_videos_kp_slider_plot_TriggerValueRight.html)
Percentage of the participants who pressed the key (i.e. felt safe to cross in front of the EV) as a function of time for each sound stimulus. The vertical line represents the passing moment. The asterisks at the bottom indicate significant differences with respect to ’Diesel’ sound, p < 0.001.

## Head rotation
[![Yaw angle](figures/all_videos_yaw_angle_Yaw.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/all_videos_yaw_angle_Yaw.html)
Mean yaw angle of head movement of pedestrians as a function of time for each sound stimulus. The value of 0 denotes participants looking perpendicular to the road, negative values correspond to the left direction (vehicle approach) and vice versa. The vertical line represents the passing moment. The asterisks at the bottom indicate significant differences with respect to 'Diesel' sound, p < 0.001.

[![Histogram of yaw angle](figures/yaw_histogram.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/yaw_histogram.html)
Histogram of yaw angle of head movement of pedestrians as a function of time for each sound stimulus. The value of 0 denotes participants looking perpendicular to the road, negative values correspond to the left direction (vehicle approach) and vice versa.

## Subjective responses
[![Barplot responses](figures/boxplot_response.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/boxplot_response.html)
Bar plots of the ratings for the questions regarding (top left) noticeability, (top right) informativeness, (bottom left) annoyance, and composite score (bottom right) per sound stimulus

[![Boxplot responses](figures/bar_response.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/bar_response.html)
Box plots of the ratings for the questions regarding (top left) noticeability, (top right) informativeness, (bottom left) annoyance, and composite score (bottom right) per sound stimulus

## Information on participants
[![Gender](figures/what_is_your_gender.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/what_is_your_gender.html)
Gender of the participants.

[![Age](figures/what_is_your_age_(in_years).png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/what_is_your_age_(in_years).html)
Age distribution of participants.

[![Hearing Aid](figures/are_you_wearing_any_seeing_aids_during_the_experiments.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/are_you_wearing_any_seeing_aids_during_the_experiments.html)
Are you wearing any seeing aids during the experiments?

[![License](figures/at_which_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/at_which_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle.html)
At which age did you obtain your first license for driving a car or motorcycle?

[![Transportation](figures/what_is_your_primary_mode_of_transportation.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/what_is_your_primary_mode_of_transportation.html)
What is your primary mode of transportation?

[![Driving](figures/about_how_many_kilometers_(miles)_did_you_drive_in_the_last_12_months.png?raw=true)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/sound-ev/blob/main/figures/about_how_many_kilometers_(miles)_did_you_drive_in_the_last_12_months.html)
"About how many kilometers (miles) did you drive in the last 12 months?"

## Setup
Tested with Python 3.9.11. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows):

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

## Configuration of project
Configuration of the project needs to be defined in `sound-ev/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `mapping`: CSV file that contains all data found in the videos.
* `plotly_template`: Template used to make graphs in the analysis.
* `output`: Directory where analysis results and intermediate output files will be saved.
* `figures`: Directory where final figures and plots are stored.
* `data`: Directory containing all raw and processed data files used in the analysis.
* `intake_questionnaire`: CSV file containing participant responses from the intake (pre-experiment) questionnaire.
* `post_experiment_questionnaire`: CSV file containing participant responses from the post-experiment questionnaire.
* `compare_trial`: Reference trial against which all other trials are compared during t-tests in the analysis.
* `kp_resolution`: Time bin size, in milliseconds, used for storing keypress data, which controls the resolution of keypress event logs.
* `yaw_resolution`: Time bin size, in milliseconds, used for storing yaw (head rotation) data, controlling the resolution of HMD orientation data.
* `smoothen_signal`:  Boolean toggle to enable or disable signal smoothing for data analysis.
* `freq`: Frequency parameter used by the One Euro Filter for signal smoothing.
* `mincutoff`: Minimum cutoff frequency for the One Euro Filter.
* `beta`: Beta value controlling the speed-versus-smoothness tradeoff in the One Euro Filter.
* `font_family`: Font family to be used in all generated figures for visual consistency.
* `font_size`: Font size to be applied to all text in generated figures.
* `p_value`: p-value threshold to be used for statistical significance testing (e.g., in t-tests).

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
   - ![Quick Settings](readme/setting_nav.jpg)

2. **Navigate to Physical Space**
   - Select the **Physical Space** option.
   - ![Physical Space](readme/physical_space.png)

3. **Clear Boundary and Space Scan History**
   - Choose the option **Clear Boundary and Space Scan History** to reset any previous environment settings.
   - ![Clear Boundary](readme/clear_boundary.png)

4. **Set Floor Level**
   - Click on **Set Floor Level** and adjust it to ensure the floor level is correctly calibrated.
   - ![Set Floor](readme/floor_level.png)

5. **Set Boundary with Participant**
   - Return to the **Quick Settings** menu and have the participant wear the headset.
   - Instruct the participant to select **Boundary** from the menu.
   - ![Boundary](readme/boundary.png)

6. **Ensure Correct Orientation**
   - Guide the participant to look down at their shoes to verify correct alignment and orientation.

7. **Confirm Boundary**
   - Ask the participant to press **Confirm Boundary** to finalize the setup.
   - ![Confirm Boundary](readme/stationary_boundary.png)

8. **Hand Over Control**
   - Once the boundary is confirmed, the experimenter can take back the controller and headset.



### **Participant ID Setup**:  
Before starting the experiment, configure the participant's unique ID:

1. **Open the Environment Scene**
   - Navigate to **Assets > Scenes**.
   - Select the **Environment** scene.
   - ![Environment](readme/environment_menu.png)

2. **Modify the ConditionController**
   - Locate the `ConditionController` in the **Hierarchy** tab.
   - Fill in the `Write File Name` field with the participant's unique name or ID.
   - ![Setup Instructions](readme/fig_1.png)


### **Connecting to Quest Link and Launching Unity**:
Set up the Unity environment and connect it to the Quest using the following steps:

1. **Launch Unity**
   - Ensure that Unity is open on your computer and the **MainMenu** environment is loaded.
   - ![MainMenu](readme/main_menu.png)

2. **Open Quest Link**
   - Navigate to the **Settings** menu and select Quest Link via Meta Quest.
   - ![Quest Link](readme/Quest_link.png)

3. **Add Desktop Panel**
   - In Quest Link, click on **Add Desktop Panel**.
   - ![Add Desktop Panel](readme/add_desktop_panel.png)

4. **Select Unity**
   - Choose Unity from the available options in the desktop panel.
   - ![Select Unity](readme/select_unity.png)

5. **Verify Main Menu Scene**
   - Ensure the Main Menu scene is visible in the Quest.
   - Press the **Play** button in Unity to start the experiment.
   - ![Play Button](readme/play_button.png)

6. **Participant Instructions**
   - Have the participant wear the headset.
   - Allow them to read the on-screen instructions before proceeding with the experiment.

7. **Resolving Overlapping Menus**
   - If another menu appears in front of the instructions:
     - ![Error](readme/error.png)
     - Press the Meta button on the right controller to resolve the issue.
     - ![Debug](readme/debug.png)


---

## Experiment Interaction

### Recording Willingness to Cross the Road

Participants record their willingness to cross the road using the **trigger button** on the controller. The interaction process is illustrated in the image below:  

![Trigger Button](readme/trigger.jpeg)

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
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com or pavlo.bazilinskyy@gmail.com.
