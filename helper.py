import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly as py
import pycountry
import math
from collections import defaultdict
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
from PIL import Image
import requests
from io import BytesIO
import base64
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, f_oneway
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import shutil

logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")

# Consts
SAVE_PNG = True
SAVE_EPS = True


# TODO: update requirements.txt
# TODO: Mark the time when the car has started to become visible, started to yield,
# stopped, started to accelerate and taking a turn finally
class HMD_helper:
    def __init__(self):
        pass

    @staticmethod
    def quaternion_to_euler(w, x, y, z):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        Roll is rotation around x-axis, pitch is rotation around y-axis, and yaw is rotation around z-axis.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def get_flag_image_url(country_name):
        """Fetches the flag image URL for a given country using ISO alpha-2 country codes."""
        try:
            # Convert country name to ISO alpha-2 country code
            country = pycountry.countries.lookup(country_name)
            # Use a flag API service that generates flags based on the country code
            return f"https://flagcdn.com/w320/{country.alpha_2.lower()}.png"  # Example API from flagcdn.com
        except LookupError:
            return None  # Return None if country not found

    def smoothen_filter(self, signal, type_flter='OneEuroFilter'):
        """Smoothen list with a filter.

        Args:
            signal (list): input signal to smoothen
            type_flter (str, optional): type_flter of filter to use.

        Returns:
            list: list with smoothened data.
        """
        if type_flter == 'OneEuroFilter':
            filter_kp = OneEuroFilter(freq=common.get_configs('freq'),            # frequency
                                      mincutoff=common.get_configs('mincutoff'),  # minimum cutoff frequency
                                      beta=common.get_configs('beta'))            # beta value
            return [filter_kp(value) for value in signal]
        else:
            logger.error('Specified filter {} not implemented.', type_flter)
            return -1

    def ttest(self, signal_1, signal_2, type='two-sided', paired=True):
        """
        Perform a t-test on two signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            type (str, optional): Type of t-test to perform. Options are "two-sided",
                                  "greater", or "less". Defaults to "two-sided".
            paired (bool, optional): Indicates whether to perform a paired t-test
                                     (`ttest_rel`) or an independent t-test (`ttest_ind`).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    `tr.common.get_configs('p_value')`.
        """
        # Check if the lengths of the two signals are the same
        if len(signal_1) != len(signal_2):
            logger.error('The lengths of signal_1 and signal_2 must be the same.')
            return -1
        # convert to numpy arrays if signal_1 and signal_2 are lists
        signal_1 = np.asarray(signal_1)
        signal_2 = np.asarray(signal_2)
        p_values = []  # record raw p value for each bin
        significance = []  # record binary flag (0 or 1) if p value < tr.common.get_configs('p_value'))
        # perform t-test for each value (treated as an independent bin)
        for i in range(len(signal_1)):
            if paired:
                t_stat, p_value = ttest_rel([signal_1[i]], [signal_2[i]], axis=-1, alternative=type)
            else:
                t_stat, p_value = ttest_ind([signal_1[i]], [signal_2[i]], axis=-1, alternative=type, equal_var=False)
            # record raw p value
            p_values.append(p_value)
            # determine significance for this value
            significance.append(int(p_value < common.get_configs('p_value')))
        # return raw p values and binary flags for significance for output
        return [p_values, significance]

    def anova(self, signals):
        """
        Perform an ANOVA test on three signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            signal_3 (list): Third signal, a list of numeric values.

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    `tr.common.get_configs('p_value')`.
        """
        # check if the lengths of the three signals are the same
        # convert signals to numpy arrays if they are lists
        p_values = []  # record raw p-values for each bin
        significance = []  # record binary flags (0 or 1) if p-value < tr.common.get_configs('p_value')
        # perform ANOVA test for each value (treated as an independent bin)
        transposed_data = list(zip(*signals['signals']))
        for i in range(len(transposed_data)):
            f_stat, p_value = f_oneway(*transposed_data[i])
            # record raw p-value
            p_values.append(p_value)
            # determine significance for this value
            significance.append(int(p_value < common.get_configs('p_value')))
        # return raw p-values and binary flags for significance for output
        return [p_values, significance]

    def twoway_anova_kp(self, signal1, signal2, signal3, output_console=True, label_str=None):
        """Perform twoway ANOVA on 2 independent variables and 1 dependent variable (as list of lists).

        Args:
            signal1 (list): independent variable 1.
            signal2 (list): independent variable 2.
            signal3 (list of lists): dependent variable 1 (keypress data).
            output_console (bool, optional): whether to print results to console.
            label_str (str, optional): label to add before console output.

        Returns:
            df: results of ANOVA
        """
        # prepare signal1 and signal2 to be of the same dimensions as signal3
        signal3_flat = [value for sublist in signal3 for value in sublist]
        # number of observations in the dependent variable
        n_observations = len(signal3_flat)
        # repeat signal1 and signal2 to match the length of signal3_flat
        signal1_expanded = np.tile(signal1, n_observations // len(signal1))
        signal2_expanded = np.tile(signal2, n_observations // len(signal2))
        # create a datafarme with data
        data = pd.DataFrame({'signal1': signal1_expanded,
                             'signal2': signal2_expanded,
                             'dependent': signal3_flat
                             })
        # perform two-way ANOVA
        model = ols('dependent ~ C(signal1) + C(signal2) + C(signal1):C(signal2)', data=data).fit()
        anova_results = anova_lm(model)
        # print results to console
        if output_console and not label_str:
            logger.info('Results for two-way ANOVA:\n', anova_results.to_string())
        if output_console and label_str:
            logger.info('Results for two-way ANOVA for ' + label_str + ':\n', anova_results.to_string())
        return anova_results

    @staticmethod
    def group_files_by_video_id(data_folder, video_data):
        # Read the main CSV to map video_id
        video_ids = video_data['video_id'].unique()

        grouped_data = defaultdict(list)

        # Traverse through the data folder and its subfolders
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.csv'):
                    # Extract the part of the filename after '_'
                    file_parts = file.split('_', maxsplit=2)
                    if len(file_parts) > 2:
                        file_video_id = file_parts[-1].split('.')[0]  # Extract video_id
                        if file_video_id in video_ids:
                            full_path = os.path.join(root, file)
                            grouped_data[file_video_id].append(full_path)

        return grouped_data

    @staticmethod
    def calculate_average_for_column(data_folder, mapping, column_name):
        """
        Calculate the average of values for a given column at each unique timestamp across grouped CSV files,
        considering only the rows within the range of 0 to video_length / 1000 and rounding timestamps to the
        nearest multiple of 0.02.

        Args:
            data_folder (dict): Location of the data folder.
            column_name (str): The name of the column to calculate the average for.
            mapping (DataFrame): A DataFrame containing the video_id and video_length information.

        Returns:
            dict: A dictionary where keys are video_ids and values are DataFrames with 'Timestamp' and average values.
        """
        timewise_averages = {}
        grouped_data = HMD_helper.group_files_by_video_id(data_folder, mapping)

        # TODO: use resolution param from config
        for video_id, file_paths in grouped_data.items():
            combined_data = []
            # Get the corresponding video_length from the mapping
            video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
            if video_length_row.empty:
                logger.info(f"Video length not found for video_id: {video_id}")
                continue

            video_length = video_length_row.values[0] / 1000

            for file_path in file_paths:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Check if the necessary columns exist
                    if 'Timestamp' in df.columns and column_name in df.columns:
                        # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                        df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length)]

                        # Round the Timestamp to the nearest multiple of 0.02
                        df["Timestamp"] = (df["Timestamp"] / 0.02).round() * 0.02

                        combined_data.append(df[['Timestamp', column_name]])

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

            if combined_data:
                # Concatenate all data for the current video_id
                combined_df = pd.concat(combined_data)

                # Group by 'Timestamp' and calculate the average for the column
                avg_df = combined_df.groupby('Timestamp', as_index=False)[column_name].mean()

                timewise_averages[video_id] = avg_df
            else:
                timewise_averages[video_id] = pd.DataFrame(columns=['Timestamp', column_name])

        return timewise_averages

    def get_sound_clip_name(self, df, video_id_value):
        result = df.loc[df["video_id"] == video_id_value, "sound_clip_name"]
        return result.iloc[0] if not result.empty else None

    def plot_kp(self, data_folder, mapping, output_folder, save_file=True):
        timewise_avgs = HMD_helper.calculate_average_for_column(data_folder, mapping, 'TriggerValueRight')
        # Create a Plotly figure
        fig = go.Figure()

        # Iterate through all trials and add traces
        for trial_name, df in timewise_avgs.items():
            # Ensure 'sound_clip_name' exists in the dataframe
            legend_name = self.get_sound_clip_name(mapping, trial_name) if "sound_clip_name" in mapping.columns else trial_name

            # TODO: add check for config param for smoothing
            smoothed_values = self.smoothen_filter(df["TriggerValueRight"].tolist())
            fig.add_trace(go.Scatter(
                x=df["Timestamp"],
                y=smoothed_values,
                mode='lines',
                name=legend_name
            ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Timestamp (s)",
            yaxis_title="Mean keypress value",
            legend_title="Trials",
            template=template
        )

        # save file to local output folder
        if save_file:
            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, 'kp', save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    def plot_yaw_movement(self, data_folder, mapping, output_folder, save_file=True):
        timewise_avgs = HMD_helper.calculate_average_for_column(data_folder, mapping, 'TriggerValueRight')
        # Create a Plotly figure
        fig = go.Figure()

        # Iterate through all trials and add traces
        for trial_name, df in timewise_avgs.items():
            smoothed_values = self.smoothen_filter(df["TriggerValueRight"].tolist())
            fig.add_trace(go.Scatter(
                x=df["Timestamp"],
                y=smoothed_values,
                mode='lines',
                name=trial_name
            ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Timestamp (s)",
            yaxis_title="Yaw",
            legend_title="Trials",
            template=template
        )

        # save file to local output folder
        if save_file:
            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, 'yaw_movement', save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    @staticmethod
    def gender_distribution(df, output_folder, save_file=True):
        # Check if df is a string (file path), and read it as a DataFrame if necessary
        if isinstance(df, str):
            df = pd.read_csv(df)
        # Count the occurrences of each gender
        gender_counts = df.groupby('What is your gender?').size().reset_index(name='count')

        # Drop any NaN values that may arise from invalid gender entries
        gender_counts = gender_counts.dropna(subset=['What is your gender?'])

        # Extract data for plotting
        genders = gender_counts['What is your gender?'].tolist()
        counts = gender_counts['count'].tolist()

        # Create the pie chart
        fig = go.Figure(data=[
            go.Pie(labels=genders, values=counts, hole=0.0, marker=dict(colors=['red', 'blue', 'green']),
                   showlegend=True)
        ])

        # Update layout
        fig.update_layout(
            legend_title_text="Gender"
        )

        # Save the figure in different formats
        # save file to local output folder
        if save_file:
            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, 'gender', save_final=True)
        # open it in localhost instead
        else:
            fig.show()
    
    @staticmethod
    def age_distribution(df, output_folder, save_file=True):
        # Check if df is a string (file path), and read it as a DataFrame if necessary
        if isinstance(df, str):
            df = pd.read_csv(df)

        # Count the occurrences of each age
        age_counts = df.groupby('What is your age (in years)?').size().reset_index(name='count')

        # Convert the 'What is your age (in years)?' column to numeric (ignoring errors for non-numeric values)
        age_counts['What is your age (in years)?'] = pd.to_numeric(age_counts['What is your age (in years)?'],
                                                                   errors='coerce')

        # Drop any NaN values that may arise from invalid age entries
        age_counts = age_counts.dropna(subset=['What is your age (in years)?'])

        # Sort the DataFrame by age in ascending order
        age_counts = age_counts.sort_values(by='What is your age (in years)?')

        # Extract data for plotting
        age = age_counts['What is your age (in years)?'].tolist()
        counts = age_counts['count'].tolist()

        # Add ' years' to each age label
        age_labels = [f"{int(a)} years" for a in age]  # Convert age values back to integers

        # Create the pie chart
        fig = go.Figure(data=[
            go.Pie(labels=age_labels, values=counts, hole=0.0, showlegend=True, sort=False)
        ])

        # Update layout
        fig.update_layout(
            legend_title_text="Age"
        )

        # save file to local output folder
        if save_file:
            # Final adjustments and display
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            self.save_plotly_figure(fig, 'age', save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    @staticmethod
    def replace_nationality_variations(df):
        # Define a dictionary mapping variations of nationality names to consistent values
        nationality_replacements = {
            "NL": "Netherlands",
            "The Netherlands": "Netherlands",
            "netherlands": "Netherlands",
            "Netherlands ": "Netherlands",
            "Nederlandse": "Netherlands",
            "Dutch": "Netherlands",
            "Bulgarian": "Bulgaria",
            "bulgarian": "Bulgaria",
            "INDIA": "India",
            "Indian": "India",
            "indian": "India",
            "italian": "Italy",
            "Italian": "Italy",
            "Chinese": "China",
            "Austrian": "Austria",
            "Maltese": "Malta",
            "Indonesian": "Indonesia",
            "Portuguese": "Portugal",
            "Romanian": "Romania"

        }

        # Replace all variations of nationality with the consistent values using a dictionary
        df['What is your nationality?'] = df['What is your nationality?'].replace(nationality_replacements, regex=True)

        return df

    @staticmethod
    def rotate_image_90_degrees(image_url):
        """Rotates an image from the URL by 90 degrees and converts it to base64."""
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        rotated_img = img.rotate(90, expand=True)  # Rotate the image by 90 degrees
        # Save the rotated image to a BytesIO object
        rotated_image_io = BytesIO()
        rotated_img.save(rotated_image_io, format="PNG")
        rotated_image_io.seek(0)

        # Convert the rotated image to base64
        base64_image = base64.b64encode(rotated_image_io.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

    @staticmethod
    def demographic_distribution(df, output_folder):
        # Check if df is a string (file path), and read it as a DataFrame if necessary
        if isinstance(df, str):
            df = pd.read_csv(df)

        df = HMD_helper.replace_nationality_variations(df)

        # Count the occurrences of each age
        demo_counts = df.groupby('What is your nationality?').size().reset_index(name='count')

        # Convert the 'What is your age (in years)?' column to numeric (ignoring errors for non-numeric values)
        demo_counts['What is your nationality??'] = pd.to_numeric(demo_counts['What is your nationality?'],
                                                                  errors='coerce')

        # Drop any NaN values that may arise from invalid age entries
        demo_counts = demo_counts.dropna(subset=['What is your nationality?'])

        # Extract data for plotting
        demo = demo_counts['What is your nationality?'].tolist()
        counts = demo_counts['count'].tolist()

        # Fetch flag image URLs and rotate images based on nationality
        flag_images = {}
        for country in demo:
            flag_url = HMD_helper.get_flag_image_url(country)
            if flag_url:
                rotated_image_base64 = HMD_helper.rotate_image_90_degrees(flag_url)  # Rotate the image by 90 degrees
                flag_images[country] = rotated_image_base64  # Store the base64-encoded rotated image

        # Create the bar chart (basic bars without filling)
        fig = go.Figure(data=[
            go.Bar(name='Country', x=demo, y=counts, marker=dict(color='white', line=dict(color='black', width=1)))
        ])

        # Calculate width of each bar for full image fill
        bar_width = (1.0 / len(demo)) * 8.8  # Assuming evenly spaced bars

        # Add flag images as overlays for each country
        for i, country in enumerate(demo):
            if country in flag_images:
                fig.add_layout_image(
                    dict(
                        source=flag_images[country],  # Embed the base64-encoded rotated image
                        xref="x",
                        yref="y",
                        x=country,  # Position the image on the x-axis at the correct bar
                        y=counts[i],  # Position the image at the top of the bar
                        sizex=bar_width,  # Adjust the width of the flag image
                        sizey=counts[i],  # Adjust the height of the flag to fit the bar height
                        xanchor="center",
                        yanchor="top",
                        sizing="stretch"
                    )
                )

        # Update layout
        fig.update_layout(
            xaxis_title='Country',
            yaxis_title='Number of participant',
            xaxis=dict(tickmode='array', tickvals=demo, ticktext=demo),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Save the figure in different formats
        base_filename = "demographic"
        fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
        fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
        fig.write_image(os.path.join(output_folder, base_filename + ".svg"),
                        width=1600, height=900, scale=3, format="svg")
        pio.write_html(fig, file=os.path.join(output_folder, base_filename + ".html"), auto_open=True)

    @staticmethod
    def read_slider_data(data_folder, output_folder):
        participant_data = {}
        all_trials = set()
        
        # iterate over participant folders
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # extract participant id
            match = re.match(r'Participant_(\d+)_', folder)
            if not match:
                continue
            participant_id = int(match.group(1))
            
            # find the main csv file containing slider data
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if re.match(rf'Participant_{participant_id}_\d+_\d+\.csv', file):
                    df = pd.read_csv(file_path, header=None, names=["trial", "noticeability", "info", "annoyance"])
                    df.set_index("trial", inplace=True)
                    participant_data[participant_id] = df
                    all_trials.update(df.index)
                    break
        
        # create a sorted list of all trials
        all_trials = sorted(all_trials, key=lambda x: (x != "test", x))
        
        # construct the dataframe
        columns = []
        for trial in all_trials:
            columns.extend([f"{trial}_noticeability", f"{trial}_info", f"{trial}_annoyance"])
        
        result_data = []
    
        for participant_id, df in participant_data.items():
            row = {"participant_id": participant_id}
            for trial in all_trials:
                if trial in df.index:
                    row[f"{trial}_noticeability"], row[f"{trial}_info"], row[f"{trial}_annoyance"] = df.loc[trial]
                else:
                    row[f"{trial}_noticeability"], row[f"{trial}_info"], row[f"{trial}_annoyance"] = None, None, None
            result_data.append(row)
        
        result_df = pd.DataFrame(result_data, columns=["participant_id"] + columns)

        # sort dataframe by participant_id
        result_df = result_df.sort_values(by="participant_id").reset_index(drop=True)
        
        # save to csv
        output_path = os.path.join(output_folder, 'slider_input.csv')
        result_df.to_csv(output_path, index=False)
        logger.info(f"Slider data saved to {output_path}")

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_final=True):
        """Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): Height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # Create directory if it doesn't exist
        output_folder = "_output"
        output_final = "figures"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML
        logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(output_folder, filename + ".html"))
        # also save the final figure
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"),  auto_open=False)

        try:
            # Save as PNG
            if SAVE_PNG:
                logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".png"), width=width, height=height,
                                scale=scale)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(output_folder, filename + ".png"),
                                os.path.join(output_final, filename + ".png"))

            # Save as EPS
            if SAVE_EPS:
                logger.info(f"Saving eps file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".eps"), width=width, height=height)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(output_folder, filename + ".eps"),
                                os.path.join(output_final, filename + ".eps"))
        except ValueError:
            logger.error(f"Value error raised when attempted to save image {filename}.")
