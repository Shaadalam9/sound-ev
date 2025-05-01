import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly as py
from plotly import subplots
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
import settings_dir as settings_dir

logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")

# Consts
SAVE_PNG = True
SAVE_EPS = True


# TODO: Mark the time when the car has started to become visible, started to yield,
# stopped, started to accelerate and taking a turn finally
class HMD_helper:

    # set template for plotly output
    template = common.get_configs('plotly_template')
    smoothen_signal = common.get_configs('smoothen_signal')
    folder_figures = 'figures'  # subdirectory to save figures
    folder_stats = 'statistics'  # subdirectory to save statistical output
    res = common.get_configs('kp_resolution')

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

        p_values = []
        significance = []
        threshold = common.get_configs("p_value")

        for i in range(len(signal_1)):
            data1 = signal_1[i]
            data2 = signal_2[i]

            # Skip if data is empty
            if not data1 or not data2 or len(data1) != len(data2) and paired:
                p_values.append(1.0)
                significance.append(0)
                continue

            try:
                if paired:
                    t_stat, p_val = ttest_rel(data1, data2, alternative=type)
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=type)
            except Exception as e:
                logger.warning(f"Skipping t-test at time index {i} due to error: {e}")
                p_val = 1.0

            p_values.append(p_val)
            significance.append(int(p_val < threshold))

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

    def get_sound_clip_name(self, df, video_id_value):
        result = df.loc[df["video_id"] == video_id_value, "sound_clip_name"]
        return result.iloc[0] if not result.empty else None

    def gender_distribution(self, df, output_folder, save_file=True):
        # Check if df is a string (file path), and read it as a DataFrame if necessary
        if isinstance(df, str):
            df = pd.read_csv(df)
        # Count the occurrences of each gender
        gender_counts = df.groupby('What is your gender?').size().reset_index(name='count')  # type: ignore

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
            self.save_plotly(fig, 'gender', save_final=True)
        # open it in localhost instead
        else:
            fig.show()

    def age_distribution(self, df, output_folder, save_file=True):
        # Check if df is a string (file path), and read it as a DataFrame if necessary
        if isinstance(df, str):
            df = pd.read_csv(df)

        # Count the occurrences of each age
        age_counts = df.groupby('What is your age (in years)?').size().reset_index(name='count')  # type: ignore

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
            self.save_plotly(fig, 'age', save_final=True)
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
            "Romanian": "Romania",
            "American": "United States",
            "Ghanaian": "Ghana",
            "Peruvian": "Peru",
            "greek": "Greece"
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
    def read_slider_data(data_folder, mapping, output_folder):
        participant_data = {}
        all_trials = set()

        # load mapping file
        mapping_dict = dict(zip(mapping["video_id"], mapping["sound_clip_name"]))

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
        all_trials = sorted([t for t in all_trials if t != "test"],
                            key=lambda x: int(re.search(r'\d+', x).group()))  # type: ignore
        all_trials.insert(0, "test") if "test" in all_trials else None

        # construct separate dataframes for each slider
        slider_data = {"noticeability": [], "info": [], "annoyance": []}

        for participant_id, df in sorted(participant_data.items()):
            row = {"participant_id": participant_id}
            for trial in all_trials:
                if trial in df.index:
                    row[trial] = df.loc[trial].to_list()
                else:
                    row[trial] = [None, None, None]

            slider_data["noticeability"].append([participant_id] + [vals[0] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["info"].append([participant_id] + [vals[1] for vals in row.values() if isinstance(vals, list)])
            slider_data["annoyance"].append([participant_id] + [vals[2] for vals in row.values() if isinstance(vals, list)])  # noqa: E501

        # convert lists to dataframes and rename columns based on mapping file
        for slider, data in slider_data.items():
            df = pd.DataFrame(data, columns=["participant_id"] + all_trials)
            df.rename(columns={trial: mapping_dict.get(trial, trial) for trial in all_trials}, inplace=True)

            # add average row
            avg_values = df.iloc[:, 1:].mean(skipna=True)
            avg_row = pd.DataFrame([["average"] + avg_values.tolist()], columns=df.columns)
            df = pd.concat([df, avg_row], ignore_index=True)

            # save each slider dataframe separately
            output_path = os.path.join(output_folder, f"slider_input_{slider}.csv")
            df.to_csv(output_path, index=False)

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            path (str): folder for saving file.
            remove_margins (bool, optional): remove white margins around EPS figure.
            width (int, optional): width of figures to be saved.
            height (int, optional): height of figures to be saved.
            save_eps (bool, optional): save image as EPS file.
            save_png (bool, optional): save image as PNG file.
            save_html (bool, optional): save image as html file.
            open_browser (bool, optional): open figure in the browse.
            save_mp4 (bool, optional): save video as MP4 file.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # build path
        path = os.path.join(settings_dir.output_dir, self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # build path for final figure
        path_final = os.path.join(settings_dir.root_dir, self.folder_figures)
        if save_final and not os.path.exists(path_final):
            os.makedirs(path_final)
        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195 or len(path_final) + len(name) > 195:
            name = name[:200 - len(path) - 5]
        # save as html
        if save_html:
            if open_browser:
                # open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'))
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
            else:
                # do not open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'), auto_open=False)
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=2, r=2, t=20, b=12))
        # save as eps
        if save_eps:
            fig.write_image(os.path.join(path, name + '.eps'), width=width, height=height)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.eps'), width=width, height=height)
        # save as png
        if save_png:
            fig.write_image(os.path.join(path, name + '.png'), width=width, height=height)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.png'), width=width, height=height)
        # save as mp4
        if save_mp4:
            fig.write_image(os.path.join(path, name + '.mp4'), width=width, height=height)

    def plot_kp_slider_videos(self, df, y: list, y_legend_kp=None, x=None, events=None, events_width=1,
                              events_dash='dot', events_colour='black', events_annotations_font_size=20,
                              events_annotations_colour='black', xaxis_kp_title='Time (s)',
                              yaxis_kp_title='Percentage of trials with response key pressed',
                              xaxis_kp_range=None, yaxis_kp_range=None, stacked=False, pretty_text=False,
                              orientation='v', xaxis_slider_title='Stimulus', yaxis_slider_show=False,
                              yaxis_slider_title=None, show_text_labels=False, xaxis_ticklabels_slider_show=True,
                              yaxis_ticklabels_slider_show=False, name_file='kp_videos_sliders', save_file=False,
                              save_final=False, fig_save_width=1320, fig_save_height=680, legend_x=0.7, legend_y=0.95,
                              font_family=None, font_size=None, ttest_signals=None, ttest_marker='circle',
                              ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                              ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                              anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                              anova_annotations_colour='black', ttest_anova_row_height=0.5, xaxis_step=5,
                              yaxis_step=5, y_legend_bar=None, line_width=1, bar_font_size=None):
        """Plot keypresses with multiple variables as a filter and slider questions for the stimuli.

        Args:
            df (dataframe): dataframe with stimuli data.
            y (list): column names of dataframe to plot.
            y_legend_kp (list, optional): names for variables for keypress data to be shown in the legend.
            x (list): values in index of dataframe to plot for. If no value is given, the index of df is used.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            xaxis_kp_title (str, optional): title for x axis. for the keypress plot
            yaxis_kp_title (str, optional): title for y axis. for the keypress plot
            xaxis_kp_range (None, optional): range of x axis in format [min, max] for the keypress plot.
            yaxis_kp_range (None, optional): range of x axis in format [min, max] for the keypress plot.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            orientation (str, optional): orientation of bars. v=vertical, h=horizontal.
            xaxis_slider_title (None, optional): title for x axis. for the slider data plot.
            yaxis_slider_show (bool, optional): show y axis or not.
            yaxis_slider_title (None, optional): title for y axis. for the slider data plot.
            show_text_labels (bool, optional): output automatically positioned text labels.
            xaxis_ticklabels_slider_show (bool, optional): show tick labels for slider plot.
            yaxis_ticklabels_slider_show (bool, optional): show tick labels for slider plot.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            legend_x (float, optional): location of legend, percentage of x axis.
            legend_y (float, optional): location of legend, percentage of y axis.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            ttest_signals (list, optional): signals to compare with ttest. None = do not compare.
            ttest_marker (str, optional): symbol of markers for the ttest.
            ttest_marker_size (int, optional): size of markers for the ttest.
            ttest_marker_colour (str, optional): colour of markers for the ttest.
            ttest_annotations_font_size (int, optional): font size of annotations for ttest.
            ttest_annotations_colour (str, optional): colour of annotations for ttest.
            anova_signals (dict, optional): signals to compare with ANOVA. None = do not compare.
            anova_marker (str, optional): symbol of markers for the ANOVA.
            anova_marker_size (int, optional): size of markers for the ANOVA.
            anova_marker_colour (str, optional): colour of markers for the ANOVA.
            anova_annotations_font_size (int, optional): font size of annotations for ANOVA.
            anova_annotations_colour (str, optional): colour of annotations for ANOVA.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            xaxis_step (int): step between ticks on x axis.
            yaxis_step (int): step between ticks on y axis.
            y_legend_bar (list, optional): names for variables for bar data to be shown in the legend.
            line_width (int): width of the keypress line.
        """
        # logger.info('Creating figure keypress and slider data for {}.', df.index.tolist())
        # calculate times
        times = df['Timestamp'].values
        # plotly
        fig = subplots.make_subplots(rows=2,
                                     cols=2,
                                     column_widths=[0.85, 0.15],
                                     # subplot_titles=('Mean keypress values', 'Responses to sliders'),
                                     specs=[[{"rowspan": 2}, {}],
                                            [None, {}]],
                                     horizontal_spacing=0.05,
                                     # vertical_spacing=0.1,
                                     shared_xaxes=False,
                                     shared_yaxes=False)
        # adjust ylim, if ttest results need to be plotted
        if ttest_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_kp_range[0] = round(yaxis_kp_range[0] - len(ttest_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore
        # adjust ylim, if anova results need to be plotted
        if anova_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_kp_range[0] = round(yaxis_kp_range[0] - len(anova_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore
        # plot keypress data
        for row_number, key in enumerate(y):
            values = df[key]  # or whatever logic fits
            if y_legend_kp:
                name = y_legend_kp[row_number]
            else:
                name = key
            # smoothen signal
            if self.smoothen_signal:
                pass
                if isinstance(values, pd.Series):
                    values = values.tolist()
                    values = self.smoothen_filter(values)
                # if not isinstance(values, (list, tuple, np.ndarray)):
                #     values = [values]
                # values = self.smoothen_filter(values)
            # plot signal
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     line=dict(width=line_width),
                                     name=name),
                          row=1,
                          col=1)
        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_kp_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)

        # update axis
        if xaxis_step:
            fig.update_xaxes(title_text=xaxis_kp_title, range=xaxis_kp_range, dtick=xaxis_step, row=1, col=1)
        else:
            fig.update_xaxes(title_text=xaxis_kp_title, range=xaxis_kp_range, row=1, col=1)
        fig.update_yaxes(title_text=yaxis_kp_title, showgrid=False, range=yaxis_kp_range, row=1, col=1)
        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
        # Plot slider data
        # use index of df if none is given
        if not x:
            x = df.index

        # draw ttest and anova rows
        self.draw_ttest_anova(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_kp_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker=ttest_marker,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              anova_signals=anova_signals,
                              anova_marker=anova_marker,
                              anova_marker_size=anova_marker_size,
                              anova_marker_colour=anova_marker_colour,
                              anova_annotations_font_size=anova_annotations_font_size,
                              anova_annotations_colour=anova_annotations_colour,
                              ttest_anova_row_height=ttest_anova_row_height)
        # update axis
        fig.update_xaxes(title_text=None, row=1, col=2)
        fig.update_xaxes(title_text=None, row=2, col=2)
        fig.update_yaxes(title_text=yaxis_slider_title, row=2, col=2)
        fig.update_yaxes(visible=yaxis_slider_show, row=1, col=2)
        fig.update_yaxes(visible=yaxis_slider_show, row=2, col=2)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=yaxis_ticklabels_slider_show, row=2, col=2)
        fig.update_xaxes(showticklabels=xaxis_ticklabels_slider_show, row=1, col=2)
        fig.update_yaxes(showticklabels=yaxis_ticklabels_slider_show, row=2, col=2)
        # update template
        fig.update_layout(template=self.template)
        # manually add grid lines for non-negative y values only
        for y in range(0, yaxis_kp_range[1] + 1, yaxis_step):  # type: ignore
            fig.add_shape(type="line",
                          x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 0,
                          x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else 1,
                          y0=y,
                          y1=y,
                          line=dict(color='#333333' if common.get_configs('plotly_template') == 'plotly_dark' else '#e5ecf6',  # noqa: E501
                                    width=1),
                          xref='x',
                          yref='y',
                          layer='below')
        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')
        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')
        # legend
        fig.update_layout(legend=dict(x=legend_x, y=legend_y, bgcolor='rgba(0,0,0,0)'))
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def draw_ttest_anova(self, fig, times, name_file, yaxis_range, yaxis_step, ttest_signals, ttest_marker,
                         ttest_marker_size, ttest_marker_colour, ttest_annotations_font_size, ttest_annotations_colour,
                         anova_signals, anova_marker, anova_marker_size, anova_marker_colour,
                         anova_annotations_font_size, anova_annotations_colour, ttest_anova_row_height):
        """Draw ttest and anova test rows.

        Args:
            fig (figure): figure object.
            name_file (str): name of file to save.
            yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
            yaxis_step (int): step between ticks on y axis.
            ttest_signals (list): signals to compare with ttest. None = do not compare.
            ttest_marker (str): symbol of markers for the ttest.
            ttest_marker_size (int): size of markers for the ttest.
            ttest_marker_colour (str): colour of markers for the ttest.
            ttest_annotations_font_size (int): font size of annotations for ttest.
            ttest_annotations_colour (str): colour of annotations for ttest.
            anova_signals (dict): signals to compare with ANOVA. None = do not compare.
            anova_marker (str): symbol of markers for the ANOVA.
            anova_marker_size (int): size of markers for the ANOVA.
            anova_marker_colour (str): colour of markers for the ANOVA.
            anova_annotations_font_size (int): font size of annotations for ANOVA.
            anova_annotations_colour (str): colour of annotations for ANOVA.
            ttest_anova_row_height (int): height of row of ttest/anova markers.
        """
        # count lines to calculate increase in coordinates of drawing
        counter_ttest = 0
        # count lines to calculate increase in coordinates of drawing
        counter_anova = 0
        # output ttest
        if ttest_signals:
            for signals in ttest_signals:
                # receive significance values
                [p_values, significance] = self.ttest(signal_1=signals['signal_1'],
                                                      signal_2=signals['signal_2'],
                                                      paired=signals['paired'])  # type: ignore

                # save results to csv
                time_step = 0.02  # or read from config if defined elsewhere
                timestamps = [round(i * time_step, 2) for i in range(len(signals['signal_1']))]
                self.save_stats_csv(t=timestamps,
                                    p_values=p_values,
                                    name_file=signals['label'] + '_' + name_file + '.csv')
                # only proceed if there are significant results
                if any(significance):  # Check if any significance is true (i.e., any stars)
                    # add to the plot
                    # plot stars based on random lists
                    marker_x = []  # x-coordinates for stars
                    marker_y = []  # y-coordinates for stars

                    # assuming `times` and `signals['signal_1']` correspond to x and y data points
                    for i in range(len(significance)):
                        if significance[i] == 1:  # if value indicates a star
                            marker_x.append(times[i])  # use the corresponding x-coordinate
                            # dynamically set y-coordinate, offset by ttest_anova_row_height for each signal_index
                            marker_y.append(-ttest_anova_row_height - counter_ttest * ttest_anova_row_height)

                    # add scatter plot trace with cleaned data
                    fig.add_trace(go.Scatter(x=marker_x,
                                             y=marker_y,
                                             # list of possible values: https://plotly.com/python/marker-style
                                             mode='markers',
                                             marker=dict(symbol=ttest_marker,  # marker
                                                         size=ttest_marker_size,  # adjust size
                                                         color=ttest_marker_colour),  # adjust colour
                                             text=p_values,
                                             showlegend=False,
                                             hovertemplate=signals['label'] + ': time=%{x}, p_value=%{text}'),
                                  row=1,
                                  col=1)
                    # add label with signals that are compared
                    fig.add_annotation(text=signals['label'],
                                       # put labels at the start of the x axis, as they are likely no significant
                                       # effects in the start of the trial
                                       x=0.2,
                                       # draw in the negative range of y axis
                                       y=-ttest_anova_row_height - counter_ttest * ttest_anova_row_height,
                                       xanchor="left",  # aligns the left edge
                                       showarrow=False,
                                       font=dict(size=ttest_annotations_font_size, color=ttest_annotations_colour))
                    # increase counter of lines drawn
                    counter_ttest = counter_ttest + 1
        # output ANOVA
        if anova_signals:
            # if ttest was plotted, take into account for y of the first row or marker
            if counter_ttest > 0:
                counter_anova = counter_ttest
            # calculate for given signals one by one
            for signals in anova_signals:
                # receive significance values
                [p_values, significance] = self.anova(signals)
                # save results to csv
                self.save_stats_csv(t=list(range(len(signals['signals'][0]))),
                                    p_values=p_values,
                                    name_file=signals['label'] + '_' + name_file + '.csv')
                # only proceed if there are significant results
                if any(significance):  # Check if any significance is true (i.e., any stars)
                    # add to the plot
                    marker_x = []  # x-coordinates for stars
                    marker_y = []  # y-coordinates for stars
                    # assuming `times` and `signals['signal_1']` correspond to x and y data points
                    for i in range(len(significance)):
                        if significance[i] == 1:  # if value indicates a star
                            marker_x.append(times[i])  # use the corresponding x-coordinate
                            # dynamically set y-coordinate, slightly offset for each signal_index
                            marker_y.append(-ttest_anova_row_height - counter_anova * ttest_anova_row_height)
                    # add scatter plot trace with cleaned data
                    fig.add_trace(go.Scatter(x=marker_x,
                                             y=marker_y,
                                             # list of possible values: https://plotly.com/python/marker-style
                                             mode='markers',
                                             marker=dict(symbol=anova_marker,  # marker
                                                         size=anova_marker_size,  # adjust size
                                                         color=anova_marker_colour),  # adjust colour
                                             text=p_values,
                                             showlegend=False,
                                             hovertemplate='time=%{x}, p_value=%{text}'),
                                  row=1,
                                  col=1)
                    # add label with signals that are compared
                    fig.add_annotation(text=signals['label'],
                                       # put labels at the start of the x axis, as they are likely no significant
                                       # effects in the start of the trial
                                       x=0.2,
                                       # draw in the negative range of y axis
                                       y=-ttest_anova_row_height - counter_anova * ttest_anova_row_height,
                                       xanchor="left",  # aligns the left edge
                                       showarrow=False,
                                       font=dict(size=anova_annotations_font_size, color=anova_annotations_colour))
                # increase counter of lines drawn
                counter_anova = counter_anova + 1
        # hide ticks of negative values on y axis assuming that ticks are at step of 5
        # calculate number of rows below x-axis (from t-test and anova)
        n_rows = counter_ttest + (counter_anova - counter_ttest if counter_anova > 0 else 0)

        # extend y-axis range downwards if needed
        min_y = -ttest_anova_row_height * (n_rows + 1)
        max_y = yaxis_range[1]

        # generate new y-axis ticks from extended min_y to max_y, but hide the negative ones
        r = range(0, int(max_y) + 1, yaxis_step)
        tickvals = list(r)
        ticktext = [str(t) if t >= 0 else '' for t in r]

        # apply updated layout
        fig.update_layout(yaxis=dict(
            range=[min_y, max_y],
            tickvals=tickvals,
            ticktext=ticktext
        ))

    def save_stats_csv(self, t, p_values, name_file):
        """Save results of statistical test in csv.

        Args:
            t (list): list of time slices.
            p_values (list): list of p values.
            name_file (str): name of file.
        """
        path = os.path.join(common.get_configs("output"), self.folder_stats)  # where to save csv
        # build path
        if not os.path.exists(path):
            os.makedirs(path)
        df = pd.DataFrame(columns=['t', 'p-value'])  # dataframe to save to csv
        df['t'] = t
        df['p-value'] = p_values
        df.to_csv(os.path.join(path, name_file))

    def draw_events(self, fig, yaxis_range, events, events_width, events_dash, events_colour,
                    events_annotations_font_size, events_annotations_colour):
        """Draw lines and annotations of events.

        Args:
            fig (figure): figure object.
            yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
            events (list): list of events to draw formatted as values on x axis.
            events_width (int): thickness of the vertical lines.
            events_dash (str): type of the vertical lines.
            events_colour (str): colour of the vertical lines.
            events_annotations_font_size (int): font size of annotations for the vertical lines.
            events_annotations_colour (str): colour of annotations for the vertical lines.
        """
        # count lines to calculate increase in coordinates of drawing
        counter_lines = 0
        # draw lines with annotations for events
        if events:
            for event in events:
                # draw start
                fig.add_shape(type='line',
                              x0=event['start'],
                              y0=0,
                              x1=event['start'],
                              y1=yaxis_range[1],
                              line=dict(color=events_colour,
                                        dash=events_dash,
                                        width=events_width))
                # draw other elements only is start and finish are not the same
                if event['start'] != event['end']:
                    # draw finish
                    fig.add_shape(type='line',
                                  x0=event['end'],
                                  y0=0,
                                  x1=event['end'],
                                  y1=yaxis_range[1],
                                  line=dict(color=events_colour,
                                            dash=events_dash,
                                            width=events_width))
                    # draw horizontal line
                    fig.add_annotation(ax=event['start'],
                                       axref='x',
                                       ay=yaxis_range[1] - counter_lines * 2 - 2,
                                       ayref='y',
                                       x=event['end'],
                                       arrowcolor='black',
                                       xref='x',
                                       y=yaxis_range[1] - counter_lines * 2 - 2,
                                       yref='y',
                                       arrowwidth=events_width,
                                       arrowside='end+start',
                                       arrowsize=1,
                                       arrowhead=2)
                    # draw text label
                    fig.add_annotation(text=event['annotation'],
                                       x=(event['end'] + event['start']) / 2,
                                       y=yaxis_range[1] - counter_lines * 2 - 1,  # use ylim value and draw lower
                                       showarrow=False,
                                       font=dict(size=events_annotations_font_size, color=events_annotations_colour))
                # just draw text label
                else:
                    fig.add_annotation(text=event['annotation'],
                                       x=event['start'] + 1.1,
                                       y=yaxis_range[1] - counter_lines * 2 - 0.2,  # use ylim value and draw lower
                                       showarrow=False,
                                       font=dict(size=events_annotations_font_size, color=events_annotations_colour))
                # increase counter of lines drawn
                counter_lines = counter_lines + 1

    def avg_csv_files(self, data_folder, mapping):

        grouped_data = HMD_helper.group_files_by_video_id(data_folder, mapping)

        for video_id, file_locations in grouped_data.items():
            combined_data = []
            video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
            if video_length_row.empty:
                logger.info(f"Video length not found for video_id: {video_id}")
                continue

            video_length = video_length_row.values[0] / 1000
            for file_location in file_locations:
                df = pd.read_csv(file_location)

                # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length)]

                # Round the Timestamp to the nearest multiple of 0.02
                df["Timestamp"] = (df["Timestamp"] / 0.02).round() * 0.02

                combined_data.append(df)

            # Concatenate all DataFrames row-wise
            combined_df = pd.concat(combined_data, ignore_index=True)

            # Group by 'Timestamp' and calculate the average for the column
            avg_df = combined_df.groupby('Timestamp', as_index=False).mean()

            # Save dataframe in the output folder
            avg_df.to_csv(os.path.join(common.get_configs("output"), f"{video_id}_avg_df.csv"))

    def export_participant_trigger_matrix(self, data_folder, video_id, output_file, mapping):
        """
        Export a matrix of TriggerValueRight values per participant for a given video.

        Args:
            data_folder (str): Path to folder containing participant data.
            video_id (str): Target video_id (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_trigger_002.csv').
            mapping (DataFrame): The mapping DataFrame containing video length info.
        """
        participant_matrix = {}
        all_timestamps = set()

        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            match = re.match(r'Participant_(\d+)_', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            for file in os.listdir(folder_path):
                if f"_{video_id}.csv" in file:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    if "Timestamp" not in df or "TriggerValueRight" not in df:
                        continue

                    # Round timestamps to 0.02s resolution
                    df["Timestamp"] = (df["Timestamp"] / 0.02).round() * 0.02

                    # Store trigger values
                    participant_matrix[f"P{participant_id}"] = dict(zip(df["Timestamp"], df["TriggerValueRight"]))
                    all_timestamps.update(df["Timestamp"])
                    break

        # Get aligned timestamps based on video length from mapping
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # convert ms to sec
            all_timestamps = np.round(np.arange(0, video_length_sec + 0.02, 0.02), 2).tolist()

            # Save timestamps
            ts_output_path = output_file.replace(".csv", "_timestamps.csv")
            pd.DataFrame({"Timestamp": all_timestamps}).to_csv(ts_output_path, index=False)
            logger.info(f"Saved timestamp range for video {video_id} to {ts_output_path}")
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values).fillna(0)

        combined_df.to_csv(output_file, index=False)
        logger.info(f"Exported participant trigger matrix for video {video_id} to {output_file}")

    def plot(self, mapping):
        """
        Generate a comparison plot of keypress data and subjective slider ratings
        across different video trials relative to a test condition.

        This function processes participant trigger matrices for each trial,
        aligns timestamps, attaches slider-based subjective ratings (annoyance,
        informativeness, noticeability), and prepares data for visualization
        including significance testing (t-tests) between the test condition and each trial.

        Args:
            mapping (pd.DataFrame): A dataframe containing metadata about videos,
                                    including 'video_id' and 'sound_clip_name'.
        """
        # Filter out the 'test' and 'est' video IDs from further processing
        video_id = mapping["video_id"]
        video_id = video_id[~video_id.isin(["test", "est"])]

        all_dfs = []         # List to collect averaged keypress data per trial
        all_labels = []      # List to store corresponding sound labels
        ttest_signals = []   # List to hold signals for pairwise t-test analysis

        data_folder = common.get_configs("data")  # Fetch path to data directory

        # --- Prepare TEST Data (used as baseline for comparison) ---
        self.export_participant_trigger_matrix(
            data_folder=data_folder,
            video_id="test",
            output_file="_output/participant_trigger_test.csv",
            mapping=mapping
        )

        # Load the test participant keypress matrix
        test_raw_df = pd.read_csv("_output/participant_trigger_test.csv")
        test_matrix = test_raw_df.drop(columns=["Timestamp"]).values.tolist()

        # --- Process Each Trial Video ---
        for video in video_id:
            display_name = mapping.loc[mapping["video_id"] == video, "display_name"].values[0]

            # Export and align participant keypress matrix for the current video
            self.export_participant_trigger_matrix(
                data_folder=data_folder,
                video_id=video,
                output_file=f"_output/participant_trigger_{video}.csv",
                mapping=mapping
            )

            # Load keypress matrix for current video
            trial_raw_df = pd.read_csv(f"_output/participant_trigger_{video}.csv")
            trial_matrix = trial_raw_df.drop(columns=["Timestamp"]).values.tolist()

            # Load averaged keypress signal over time
            df = pd.read_csv(f"_output/{video}_avg_df.csv")

            # Append data and label for plotting
            all_dfs.append(df)
            all_labels.append(display_name)

            # Store signals for statistical comparison with test condition
            ttest_signals.append({
                "signal_1": test_matrix,
                "signal_2": trial_matrix,
                "paired": True,
                "label": f"{display_name}"
            })

        # --- Combine DataFrames for Plotting ---
        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]  # Assumes all trials share the same time index

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df["TriggerValueRight"]

        # --- Generate Plot ---
        self.plot_kp_slider_videos(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_kp_range=[0, 1],
            yaxis_slider_title="Slider rating (%)",
            name_file="all_videos_kp_slider_plot",
            show_text_labels=True,
            pretty_text=True,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=0.02
        )
