import pandas as pd
import os
import plotly.graph_objects as go
import plotly as py
from plotly import subplots
from plotly.subplots import make_subplots
import pycountry
import math
from collections import defaultdict
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, f_oneway
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

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

        # returns in radians
        return roll, pitch, yaw

    def average_quaternions_eigen(self, quaternions):
        """
        Average a list of quaternions using Markley's method (via eigen decomposition:https://doi.org/10.2514/1.28949).

        Args:
            quaternions (List[List[float]]): List of [w, x, y, z] quaternions.

        Returns:
            np.ndarray: Averaged quaternion as [w, x, y, z]
        """
        if len(quaternions) == 0:
            raise ValueError("No quaternions to average.")
        elif len(quaternions) == 1:
            return np.array(quaternions[0])

        # Convert to numpy array and ensure shape (N, 4)
        q_arr = np.array(quaternions)

        # Normalize each quaternion to unit length
        q_arr = np.array([q / np.linalg.norm(q) for q in q_arr])

        # Ensure quaternions are all in the same hemisphere
        # Flip quaternions with negative dot product to the first
        reference = q_arr[0]
        for i in range(1, len(q_arr)):
            if np.dot(reference, q_arr[i]) < 0:
                q_arr[i] = -q_arr[i]

        # Form the symmetric accumulator matrix
        A = np.zeros((4, 4))
        for q in q_arr:
            q = q.reshape(4, 1)  # Make column vector
            A += q @ q.T         # Outer product

        # Normalize by number of quaternions (optional)
        A /= len(q_arr)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        avg_q = eigenvectors[:, np.argmax(eigenvalues)]  # Pick eigenvector with largest eigenvalue

        # Ensure scalar-first order: [w, x, y, z]
        return avg_q if avg_q[0] >= 0 else -avg_q  # Normalise sign

    def compute_yaw_from_quaternions(self, data_folder, video_id, mapping, output_file):
        """
        Computes the average yaw angle per timestamp using quaternions for a given video_id.

        Args:
            data_folder (str): Base folder where participant CSVs are stored.
            video_id (str): The video ID to process.
            mapping (pd.DataFrame): Mapping file that includes video lengths.
            output_file (str): Path to output CSV with average yaw angle.
        """
        grouped_data = self.group_files_by_video_id(data_folder, mapping)
        files = grouped_data.get(video_id, [])

        if not files:
            logger.warning(f"No CSV files found for video_id={video_id}")
            return

        all_data = []

        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if video_length_row.empty:
            logger.warning(f"Video length not found in mapping for video_id={video_id}")
            return
        video_length = video_length_row.values[0] / 1000  # Convert ms to seconds

        for file_path in files:
            df = pd.read_csv(file_path)
            required_cols = {"Timestamp", "HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}
            if not required_cols.issubset(df.columns):
                continue

            df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length + 0.01)]
            df["Timestamp"] = ((df["Timestamp"] / 0.02).round() * 0.02).astype(float)

            grouped = df.groupby("Timestamp")[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]].apply(
                lambda group: self.quaternion_to_euler(*self.average_quaternions_eigen(group.values))[2]  # yaw
            ).reset_index(name="Yaw")
            all_data.append(grouped)

        if not all_data:
            logger.warning(f"No valid quaternion data for video_id={video_id}")
            return

        avg_df = pd.concat(all_data).groupby("Timestamp", as_index=False).mean()
        avg_df.to_csv(output_file, index=False)
        logger.info(f"Averaged yaw angle saved to: {output_file}")

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
                                     (ttest_rel) or an independent t-test (ttest_ind).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    tr.common.get_configs('p_value').
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
                    tr.common.get_configs('p_value').
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

    def plot_column_distribution(self, df, columns, output_folder, save_file=True):
        """
        Plots and prints distributions of specified survey columns.

        Parameters:
            df (DataFrame or str): DataFrame or path to CSV.
            columns (list): List of column names to analyze.
            output_folder (str): Folder where plots will be saved.
            save_file (bool): Whether to save plots or just show them.
        """
        if isinstance(df, str):
            df = pd.read_csv(df)

        for column in columns:
            if column not in df.columns:
                print(f"Column not found: {column}")
                continue

            print(f"\n--- Distribution for: '{column}' ---")
            # Drop missing
            data = df[column].dropna().astype(str).str.strip()
            value_counts = data.value_counts()

            # Print counts
            for value, count in value_counts.items():
                print(f"{value}: {count}")

            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.0)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = column.replace(" ", "_").replace("?", "").lower()
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def distribution_plots(self, df, column_names, output_folder, save_file=True):

        if isinstance(df, str):
            df = pd.read_csv(df)

        for column_name in column_names:
            if column_name not in df.columns:
                logger.warning(f"Column not found: {column_name}")
                continue

            # Try numeric conversion
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            is_numeric = pd.api.types.is_numeric_dtype(temp_series)

            # Drop NaNs
            df_clean = df.dropna(subset=[column_name]).copy()

            if df_clean.empty:
                logger.warning(f"No valid data in column: {column_name}")
                continue

            if is_numeric:
                # Numeric column processing
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
                mean_val = df_clean[column_name].mean()
                std_val = df_clean[column_name].std()
                logger.info(f"{column_name} - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

                value_counts = df_clean[column_name].round().value_counts().sort_index()
                labels = [f"{int(v)}" for v in value_counts.index]
                values = value_counts.values
            else:
                # Categorical column processing
                df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
                value_counts = df_clean[column_name].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
                logger.info(f"{column_name} - Response counts: {dict(zip(labels, values))}")

            # Plotting
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=0.0, showlegend=True, sort=False)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = column_name.replace(" ", "_").replace("?", "").lower()
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

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
        path = os.path.join(common.get_configs("output"), self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # build path for final figure
        path_final = os.path.join(common.get_configs("figures"), self.folder_figures)
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
                              xaxis_kp_title_offset=0, yaxis_kp_title_offset=0,
                              xaxis_kp_range=None, yaxis_kp_range=None, stacked=False,
                              pretty_text=False, orientation='v', xaxis_slider_title='Stimulus',
                              yaxis_slider_show=False, yaxis_slider_title=None, show_text_labels=False,
                              xaxis_ticklabels_slider_show=True, yaxis_ticklabels_slider_show=False,
                              name_file='kp_videos_sliders', save_file=False, save_final=False,
                              fig_save_width=1320, fig_save_height=680, legend_x=0.7, legend_y=0.95,
                              font_family=None, font_size=None, ttest_signals=None, ttest_marker='circle',
                              ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                              ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                              anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                              anova_annotations_colour='black', ttest_anova_row_height=0.5, xaxis_step=5,
                              yaxis_step=5, y_legend_bar=None, line_width=1, bar_font_size=None):
        """
        Plot keypresses with multiple variables as a filter and slider questions for the stimuli.

        Args:
            df (dataframe): DataFrame with stimuli data.
            y (list): Column names of DataFrame to plot.
            y_legend_kp (list, optional): Names for variables for keypress data to be shown in the legend.
            x (list, optional): Values in index of DataFrame to plot for. If None, the index of df is used.
            events (list, optional): List of events to draw, formatted as values on x axis.
            events_width (int, optional): Thickness of the vertical lines.
            events_dash (str, optional): Style of the vertical lines (e.g., 'dot', 'dash').
            events_colour (str, optional): Colour of the vertical lines.
            events_annotations_font_size (int, optional): Font size for annotations on vertical lines.
            events_annotations_colour (str, optional): Colour for annotations on vertical lines.
            xaxis_kp_title (str, optional): Title for x axis of the keypress plot.
            yaxis_kp_title (str, optional): Title for y axis of the keypress plot.
            xaxis_kp_title_offset (float, optional): Horizontal offset for x axis title of keypress plot.
            yaxis_kp_title_offset (float, optional): Vertical offset for y axis title of keypress plot.
            xaxis_kp_range (list or None, optional): Range of x axis in format [min, max] for keypress plot.
            yaxis_kp_range (list or None, optional): Range of y axis in format [min, max] for keypress plot.
            stacked (bool, optional): Whether to show bars as stacked chart.
            pretty_text (bool, optional): Prettify tick labels by replacing underscores with spaces and capitalizing.
            orientation (str, optional): Orientation of bars; 'v' = vertical, 'h' = horizontal.
            xaxis_slider_title (str, optional): Title for x axis of the slider data plot.
            yaxis_slider_show (bool, optional): Whether to show y axis on slider plot.
            yaxis_slider_title (str, optional): Title for y axis of the slider data plot.
            show_text_labels (bool, optional): Whether to output automatically positioned text labels.
            xaxis_ticklabels_slider_show (bool, optional): Whether to show tick labels for slider x axis.
            yaxis_ticklabels_slider_show (bool, optional): Whether to show tick labels for slider y axis.
            name_file (str, optional): Name of file to save.
            save_file (bool, optional): Whether to save the plot as an HTML file.
            save_final (bool, optional): Whether to save the figure as a final image in /figures.
            fig_save_width (int, optional): Width of the figure when saving.
            fig_save_height (int, optional): Height of the figure when saving.
            legend_x (float, optional): X location of legend as percentage of plot width.
            legend_y (float, optional): Y location of legend as percentage of plot height.
            font_family (str, optional): Font family to use in the figure.
            font_size (int, optional): Font size to use in the figure.
            ttest_signals (list, optional): Signals to compare using t-test.
            ttest_marker (str, optional): Marker style for t-test points.
            ttest_marker_size (int, optional): Size of t-test markers.
            ttest_marker_colour (str, optional): Colour of t-test markers.
            ttest_annotations_font_size (int, optional): Font size of t-test annotations.
            ttest_annotations_colour (str, optional): Colour of t-test annotations.
            anova_signals (dict, optional): Signals to compare using ANOVA.
            anova_marker (str, optional): Marker style for ANOVA points.
            anova_marker_size (int, optional): Size of ANOVA markers.
            anova_marker_colour (str, optional): Colour of ANOVA markers.
            anova_annotations_font_size (int, optional): Font size of ANOVA annotations.
            anova_annotations_colour (str, optional): Colour of ANOVA annotations.
            ttest_anova_row_height (float, optional): Height per row for t-test/ANOVA marker rows.
            xaxis_step (int): Step between ticks on x axis.
            yaxis_step (float): Step between ticks on y axis.
            y_legend_bar (list, optional): Names for variables in bar data for legend.
            line_width (int): Line width for keypress data plot.
        """

        # logger.info('Creating figure keypress and slider data for {}.', df.index.tolist())
        # calculate times
        times = df['Timestamp'].values
        # plotly
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=False,
                                     shared_yaxes=False)
        # adjust ylim, if ttest results need to be plotted
        if ttest_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_kp_range[0] = (yaxis_kp_range[0] - len(ttest_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore

        # adjust ylim, if anova results need to be plotted
        if anova_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_kp_range[0] = (yaxis_kp_range[0] - len(anova_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501  # type: ignore

        # track plotted values to compute min/max for ticks
        all_values = []

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

            all_values.extend(values)  # type: ignore # collect values for y-axis tick range

            name = y_legend_kp[row_number] if y_legend_kp else key

            # plot signal
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     line=dict(width=line_width),
                                     name=name), row=1, col=1)

        # draw events
        HMD_helper.draw_events(fig=fig,
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
        # Find actual y range across all series
        # actual_ymin = min([min(df[y_col]) for y_col in y])
        # actual_ymax = max([max(df[y_col]) for y_col in y])
        actual_ymin = min(all_values)
        actual_ymax = max(all_values)

        # Generate ticks from 0 up to actual_ymax
        positive_ticks = np.arange(0, actual_ymax + yaxis_step, yaxis_step)

        # Generate ticks from 0 down to actual_ymin (note: ymin is negative)
        negative_ticks = np.arange(0, actual_ymin - yaxis_step, -yaxis_step)

        # Combine and sort ticks
        visible_ticks = np.sort(np.unique(np.concatenate((negative_ticks, positive_ticks))))

        # Update y-axis with only relevant tick marks
        fig.update_yaxes(
            showgrid=True,
            range=[actual_ymin, actual_ymax],
            tickvals=visible_ticks,  # only show ticks for data range
            tickformat='.2f',
            row=1, col=1
        )

        fig.add_annotation(
            text=yaxis_kp_title,
            xref='paper', yref='paper',
            x=xaxis_kp_title_offset,  # left side of the plot
            y=0.5 + yaxis_kp_title_offset,  # middle + offset
            showarrow=False,
            textangle=-90,
            # font=dict(size=font_size or common.get_configs('font_size')),
            xanchor='center',
            yanchor='middle'
        )

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
        for y in np.arange(0, yaxis_kp_range[1] + 0.01, yaxis_step):  # type: ignore
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
                             remove_margins=False,
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
        # Save original axis limits
        original_min, original_max = yaxis_range
        # Counters for marker rows
        counter_ttest = 0
        counter_anova = 0

        # --- t-test markers ---
        if ttest_signals:
            for comp in ttest_signals:
                p_vals, sig = self.ttest(
                    signal_1=comp['signal_1'], signal_2=comp['signal_2'], paired=comp['paired']
                )  # type: ignore

                # Save csv
                times_csv = [round(i * 0.02, 2) for i in range(len(comp['signal_1']))]
                self.save_stats_csv(t=times_csv,
                                    p_values=p_vals,
                                    name_file=f"{comp['label']}_{name_file}.csv")

                if any(sig):
                    xs, ys = [], []
                    y_offset = original_min - ttest_anova_row_height * (counter_ttest + 1)
                    for i, s in enumerate(sig):
                        if s:
                            xs.append(times[i])
                            ys.append(y_offset)
                    # plot markers
                    fig.add_trace(go.Scatter(x=xs,
                                             y=ys,
                                             mode='markers',
                                             marker=dict(symbol=ttest_marker,
                                                         size=ttest_marker_size,
                                                         color=ttest_marker_colour),
                                             text=p_vals,
                                             showlegend=False,
                                             hovertemplate=f"{comp['label']}: time=%{{x}}, p=%{{text}}"))
                    # label row
                    fig.add_annotation(x=times[0] - (times[-1] - times[0]) * 0.0,
                                       y=y_offset,
                                       text=comp['label'],
                                       xanchor='right',
                                       showarrow=False,
                                       font=dict(size=ttest_annotations_font_size,
                                                 color=ttest_annotations_colour))
                    counter_ttest += 1

        # --- ANOVA markers ---
        if anova_signals:
            counter_anova = counter_ttest
            for comp in anova_signals:
                p_vals, sig = self.anova(comp)
                self.save_stats_csv(t=list(range(len(comp['signals'][0]))),
                                    p_values=p_vals,
                                    name_file=f"{comp['label']}_{name_file}.csv")

                if any(sig):
                    xs, ys = [], []
                    y_offset = original_min - ttest_anova_row_height * (counter_anova + 1)
                    for i, s in enumerate(sig):
                        if s:
                            xs.append(times[i])
                            ys.append(y_offset)

                    fig.add_trace(go.Scatter(x=xs,
                                             y=ys,
                                             mode='markers',
                                             marker=dict(symbol=anova_marker,
                                                         size=anova_marker_size,
                                                         color=anova_marker_colour),
                                             text=p_vals,
                                             showlegend=False,
                                             hovertemplate=f"{comp['label']}: time=%{{x}}, p=%{{text}}"))
                    fig.add_annotation(x=times[0] - (times[-1] - times[0]) * 0.05,
                                       y=y_offset,
                                       text=comp['label'],
                                       xanchor='right',
                                       showarrow=False,
                                       font=dict(size=anova_annotations_font_size,
                                                 color=anova_annotations_colour))
                counter_anova += 1

        # --- Adjust axis ---
        n_rows = counter_ttest + max(0, counter_anova - counter_ttest)
        min_y = original_min - ttest_anova_row_height * (n_rows + 1)
        # Use dtick + tickformat for float ticks
        fig.update_layout(yaxis=dict(
            range=[min_y, original_max],
            dtick=yaxis_step,
            tickformat='.2f'
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

    @staticmethod
    def draw_events(fig, yaxis_range, events, events_width, events_dash, events_colour,
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
        """
        Averages multiple CSV files corresponding to the same video ID. Each file is expected to contain
        time-series data, including quaternion rotations and potentially other columns. The output is a
        CSV file with averaged values for each timestamp across the files.

        Parameters:
            data_folder (str): Path to the folder containing input CSV files.
            mapping (pd.DataFrame): A DataFrame containing metadata, including 'video_id' and 'video_length'.

        Outputs:
            For each video_id, saves an averaged DataFrame as a CSV in the output directory.
            The output CSV is named as "<video_id>_avg_df.csv".
        """

        # Group file paths by video_id using a helper function
        grouped_data = HMD_helper.group_files_by_video_id(data_folder, mapping)

        # Process each video ID and its associated files
        for video_id, file_locations in grouped_data.items():
            all_dfs = []

            # Retrieve the video length from the mapping DataFrame
            video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
            if video_length_row.empty:
                logger.info(f"Video length not found for video_id: {video_id}")
                continue

            video_length = video_length_row.values[0] / 1000  # Convert milliseconds to seconds

            # Read and process each file associated with the video ID
            for file_location in file_locations:
                df = pd.read_csv(file_location)

                # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length + 0.01)]

                # Round the Timestamp to the nearest multiple of 0.02
                df["Timestamp"] = ((df["Timestamp"] / 0.02).round() * 0.02).astype(float)

                all_dfs.append(df)

            # Skip if no dataframes were collected
            if not all_dfs:
                continue

            # Concatenate all DataFrames row-wise
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Group by 'Timestamp'
            grouped = combined_df.groupby('Timestamp')

            avg_rows = []
            for timestamp, group in grouped:
                row = {'Timestamp': timestamp}

                # Perform SLERP-based quaternion averaging if quaternion columns are present
                if {"HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}.issubset(group.columns):
                    quats = group[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]].values.tolist()
                    avg_quat = self.average_quaternions_eigen(quats)
                    row.update({
                        "HMDRotationW": avg_quat[0],
                        "HMDRotationX": avg_quat[1],
                        "HMDRotationY": avg_quat[2],
                        "HMDRotationZ": avg_quat[3],
                    })

                # Average all remaining columns (excluding Timestamp and quaternion cols)
                other_cols = [col for col in group.columns if col not in ["Timestamp",
                                                                          "HMDRotationW",
                                                                          "HMDRotationX",
                                                                          "HMDRotationY",
                                                                          "HMDRotationZ"]]
                for col in other_cols:
                    row[col] = group[col].mean()

                avg_rows.append(row)

            # Create a new DataFrame from the averaged rows
            avg_df = pd.DataFrame(avg_rows)

            # Save dataframe in the output folder
            avg_df.to_csv(os.path.join(common.get_configs("output"), f"{video_id}_avg_df.csv"), index=False)

    def export_participant_trigger_matrix(self, data_folder, video_id, output_file, column_name, mapping):
        """
        Export a matrix of column name values per participant for a given video.

        Args:
            data_folder (str): Path to folder containing participant data.
            video_id (str): Target video_id (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_trigger_002.csv').
            column_name (str): .
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

                    if "Timestamp" not in df or column_name not in df:
                        continue

                    # Round timestamps to 0.02s resolution
                    df["Timestamp"] = ((df["Timestamp"] / 0.02).round() * 0.02).astype(float)
                    df["Timestamp"] = df["Timestamp"].round(2)

                    # Store trigger values
                    participant_matrix[f"P{participant_id}"] = dict(zip(df["Timestamp"], df[column_name]))
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
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})

        # ⚠️ Do NOT fill missing with 0 – preserve NaN for clarity
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values)

        combined_df.to_csv(output_file, index=False)

    def export_participant_yaw_matrix(self, data_folder, video_id, output_file, mapping):
        """
        Export a matrix of yaw angles (computed from quaternions) per participant for a given video.

        Args:
            data_folder (str): Path to folder containing participant data.
            video_id (str): Target video_id (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_yaw_002.csv').
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

                    required_cols = {"Timestamp", "HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}
                    if not required_cols.issubset(df.columns):
                        continue

                    # Normalize timestamps
                    df["Timestamp"] = ((df["Timestamp"] / 0.02).round() * 0.02).astype(float)
                    df["Timestamp"] = df["Timestamp"].round(2)

                    # Group by timestamp and compute yaw
                    yaw_by_time = df.groupby("Timestamp")[["HMDRotationW",
                                                           "HMDRotationX",
                                                           "HMDRotationY",
                                                           "HMDRotationZ"]].apply(
                        lambda group: self.quaternion_to_euler(*self.average_quaternions_eigen(group.values))[2]  # yaw
                    ).reset_index(name="Yaw")

                    participant_matrix[f"P{participant_id}"] = dict(zip(yaw_by_time["Timestamp"], yaw_by_time["Yaw"]))
                    all_timestamps.update(yaw_by_time["Timestamp"])
                    break

        # Determine aligned timestamps using mapping
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # convert ms to sec
            all_timestamps = np.round(np.arange(0, video_length_sec + 0.02, 0.02), 2).tolist()

            # Save timestamps
            ts_output_path = output_file.replace(".csv", "_timestamps.csv")
            pd.DataFrame({"Timestamp": all_timestamps}).to_csv(ts_output_path, index=False)
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values)

        combined_df.to_csv(output_file, index=False)

    def plot_column(self, mapping, column_name="TriggerValueRight"):
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
            column_name (str): The name of the column to extract for plotting
                               (e.g., 'TriggerValueRight', 'TriggerValueLeft').
        """
        # Filter out the 'test' and 'est' video IDs from further processing
        video_id = mapping["video_id"]
        plot_videos = video_id[~video_id.isin(["est"])]

        all_dfs = []
        all_labels = []
        ttest_signals = []

        data_folder = common.get_configs("data")

        # Prepare test data
        self.export_participant_trigger_matrix(
            data_folder=data_folder,
            video_id="test",
            output_file=f"_output/participant_{column_name}_test.csv",
            column_name=column_name,
            mapping=mapping
        )

        test_raw_df = pd.read_csv(f"_output/participant_{column_name}_test.csv")
        test_matrix = test_raw_df.drop(columns=["Timestamp"]).values.tolist()

        # Process each video (including 'test')
        for video in plot_videos:
            display_name = mapping.loc[mapping["video_id"] == video, "display_name"].values[0]

            self.export_participant_trigger_matrix(
                data_folder=data_folder,
                video_id=video,
                output_file=f"_output/participant_{column_name}_{video}.csv",
                column_name=column_name,
                mapping=mapping
            )

            trial_raw_df = pd.read_csv(f"_output/participant_{column_name}_{video}.csv")
            trial_matrix = trial_raw_df.drop(columns=["Timestamp"]).values.tolist()

            df = pd.read_csv(f"_output/{video}_avg_df.csv")

            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in file: _output/{video}_avg_df.csv")

            all_dfs.append(df)
            all_labels.append(display_name)

            # Skip t-test if comparing test to itself
            if video != "test":
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # Combine DataFrames
        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df[column_name]

        # Plotting
        self.plot_kp_slider_videos(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_kp_range=[0.45, 1],
            yaxis_slider_title="Slider rating (%)",
            xaxis_kp_title_offset=-0.035,  # type: ignore
            yaxis_kp_title_offset=0.18,  # type: ignore
            name_file=f"all_videos_kp_slider_plot_{column_name}",
            show_text_labels=True,
            pretty_text=True,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=0.03,
            legend_x=0.78,
            legend_y=1,
            xaxis_step=3,
            yaxis_step=0.25,  # type: ignore
            line_width=3,
            fig_save_width=1800,
            fig_save_height=900,
            save_file=True,
            save_final=True
        )

    def plot_yaw(self, mapping, column_name="Yaw"):
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
            column_name (str): The name of the column to extract for plotting
                               (e.g., 'TriggerValueRight', 'TriggerValueLeft').
        """
        # Filter out the 'test' and 'est' video IDs from further processing
        video_id = mapping["video_id"]
        plot_videos = video_id[~video_id.isin(["est"])]

        all_dfs = []
        all_labels = []
        ttest_signals = []

        data_folder = common.get_configs("data")

        # Prepare test data
        self.export_participant_yaw_matrix(
            data_folder=data_folder,
            video_id="test",
            output_file=f"_output/participant_{column_name}_test.csv",
            mapping=mapping
        )

        test_raw_df = pd.read_csv(f"_output/participant_{column_name}_test.csv")
        test_matrix = test_raw_df.drop(columns=["Timestamp"]).values.tolist()

        # Process each trial (including 'test' for plotting)
        for video in plot_videos:
            display_name = mapping.loc[mapping["video_id"] == video, "display_name"].values[0]

            self.export_participant_yaw_matrix(
                data_folder=data_folder,
                video_id=video,
                output_file=f"_output/participant_{column_name}_{video}.csv",
                mapping=mapping
            )

            trial_raw_df = pd.read_csv(f"_output/participant_{column_name}_{video}.csv")
            trial_matrix = trial_raw_df.drop(columns=["Timestamp"]).values.tolist()

            yaw_csv = f"_output/yaw_avg_{video}.csv"
            self.compute_yaw_from_quaternions(data_folder, video, mapping, yaw_csv)
            df = pd.read_csv(yaw_csv)

            all_dfs.append(df)
            all_labels.append(display_name)

            # Skip t-test of 'test' vs. itself
            if video != "test":
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # Combine DataFrames
        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df[column_name]

        # Plotting
        self.plot_kp_slider_videos(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_kp_range=[0.03, 0.1],
            yaxis_kp_title="Radian",
            xaxis_kp_title_offset=-0.035,  # type: ignore
            yaxis_kp_title_offset=0.22,  # type: ignore
            name_file=f"all_videos_yaw_angle_{column_name}",
            show_text_labels=True,
            pretty_text=True,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=0.01,
            xaxis_step=3,
            yaxis_step=0.03,  # type: ignore
            legend_x=0.8,
            legend_y=0.2,
            line_width=3,
            fig_save_width=1800,
            fig_save_height=900,
            save_file=True,
            save_final=True
        )

    def plot_individual_csvs_plotly(self, csv_paths, mapping_df):
        """
        Reads three CSV files, extracts the 'average' row, and creates 3 subplots
        (bar charts), each showing 15 sound clip averages with standard deviation.

        Parameters:
            csv_paths (list of str): List of three file paths to CSVs.
        """
        if len(csv_paths) != 3:
            raise ValueError("Please provide exactly three CSV file paths.")

        # Load display name mapping
        mapping_dict = dict(zip(mapping_df['sound_clip_name'], mapping_df['display_name']))

        # Read dataframes and extract average and std rows
        avgs, stds = [], []

        for path in csv_paths:
            df = pd.read_csv(path)
            avg_row = df[df['participant_id'] == 'average']
            if avg_row.empty:
                raise ValueError(f"No 'average' row found in {path}")

            numeric_df = df[df['participant_id'] != 'average'].drop(columns='participant_id').astype(float)
            std_row = numeric_df.std()
            avg_row = avg_row.drop(columns='participant_id').iloc[0].astype(float)

            avgs.append(avg_row)
            stds.append(std_row)

        columns = avgs[0].index.tolist()
        display_names = [mapping_dict.get(col, col) for col in columns]

        # Create a 1x3 subplot layout (one subplot per CSV)
        fig = make_subplots(rows=1, cols=3, subplot_titles=['Annoyance', 'Info', 'Noticeability'])

        for i in range(3):
            means = avgs[i]
            deviations = stds[i]
            max_val = max(means)
            y_max = max_val + 1.5  # Add buffer for label visibility

            fig.add_trace(
                go.Bar(
                    x=display_names,
                    y=means,
                    name=f'CSV{i+1}',
                    showlegend=False
                ),
                row=1,
                col=i+1
            )

            # Add rotated annotations (vertical labels)
            for j, (x_val, y_val, m, d) in enumerate(zip(display_names, means, means, deviations)):
                fig.add_annotation(
                    text=f"{m:.2f} ({d:.2f})",
                    x=x_val,
                    y=y_val + 0.15,  # slightly above the bar
                    showarrow=False,
                    textangle=-90,  # rotate text vertically
                    font=dict(size=12),
                    xanchor='center',
                    yanchor='bottom',
                    row=1,
                    col=i+1
                )
            fig.update_yaxes(range=[0, y_max], row=1, col=i+1)

        fig.update_layout(
            height=700,
            width=1800,
            margin=dict(t=80, b=120, l=40, r=40),
            showlegend=False
        )

        fig.update_xaxes(tickangle=45)
        self.save_plotly(fig, 'bar_repsonse', save_final=True)
