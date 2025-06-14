# by Shadab Alam <shaadalam.5u@gmail.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd
import os


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
HMD = HMD_helper()

template = common.get_configs("plotly_template")
data_folder = common.get_configs("data")  # new location of the csv file with participant id
mapping = pd.read_csv(common.get_configs("mapping"))  # mapping file
output_folder = common.get_configs("output")
intake_questionnaire = common.get_configs("intake_questionnaire")   # intake questionnaire
post_experiment_questionnaire = common.get_configs("post_experiment_questionnaire")  # post-experiment questionnaire

intake_columns_to_plot = [
    "What is your gender?",
    "Are you wearing any seeing aids during the experiments?",
    "How often in the last month have you experienced virtual reality?",
    "What is your primary mode of transportation?",
    "On average, how often did you drive a vehicle in the last 12 months?",
    "About how many kilometers (miles) did you drive in the last 12 months?",
    "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)",  # noqa: E501
    "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.",  # noqa: E501
    "How often do you do the following?: Disregarding the speed limit on a motorway.",
    "How often do you do the following?: Disregarding the speed limit on a residential road.",
    "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency.",  # noqa: E501
    "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you.",  # noqa: E501
    "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user.",
    "How often do you do the following?: Using a mobile phone without a hands free kit."
]

post_columns_to_plot = [
    "The type of sound that the car was emitting affected my decision to cross the road."
]

intake_columns_distribution_to_plot = [
    "What is your age (in years)?",
    "At which age did you obtain your first license for driving a car or motorcycle?"
]

post_columns_distribution_to_plot = [
    "How stressful did you feel during the experiment?",
    "How anxious did you feel during the experiment?",
    "How realistic did you find the experiment?",
    "How would you rate your overall experience in this experiment?"
]

try:
    # Check if the directory already exists
    if not os.path.exists(output_folder):
        # Create the directory
        os.makedirs(output_folder)
        print(f"Directory '{output_folder}' created successfully.")
except Exception as e:
    print(f"Error occurred while creating directory: {e}")

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")

    # Read and process data
    HMD.read_slider_data(data_folder, mapping, output_folder)

    # Keypress data
    HMD.plot_column(mapping,
                    column_name="TriggerValueRight",
                    xaxis_range=[0, 11],
                    yaxis_range=[0, 100],
                    xaxis_title="Time, [s]",
                    yaxis_title="Percentage of trials with trigger key pressed",
                    margin=dict(l=120, r=2, t=12, b=12))

    # Head rotation
    HMD.plot_yaw(mapping,
                 xaxis_range=[0, 11],
                 yaxis_range=[-0.06, 0.06],
                 xaxis_title="Time, [s]",
                 yaxis_title="Yaw angle, [radian]",
                 margin=dict(l=100, r=2, t=10, b=10))
    HMD.plot_yaw_histogram(mapping, angle=30, num_bins=30, smoothen_filter_param=True)

    # Subjective responses
    HMD.plot_individual_csvs(csv_paths=["_output/slider_input_noticeability.csv",  # Noticeability
                                        "_output/slider_input_info.csv",           # Informativeness
                                        "_output/slider_input_annoyance.csv"],     # Annoyance
                             mapping_df=mapping,
                             font_size=30,
                             vertical_spacing=0.27,
                             height=1500,
                             width=1600,
                             margin=dict(t=40, b=100, l=10, r=10))

    HMD.plot_individual_csvs_barplot(csv_paths=["_output/slider_input_noticeability.csv",  # Noticeability
                                                "_output/slider_input_info.csv",           # Informativeness
                                                "_output/slider_input_annoyance.csv"],     # Annoyance
                                     mapping_df=mapping)

    # Information on participants
    HMD.plot_column_distribution(intake_questionnaire,
                                 intake_columns_to_plot,
                                 output_folder="output",
                                 save_file=True)
    HMD.plot_column_distribution(post_experiment_questionnaire,
                                 post_columns_to_plot,
                                 output_folder="output",
                                 save_file=True)
    HMD.distribution_plots(intake_questionnaire,
                           intake_columns_distribution_to_plot,
                           output_folder="output",
                           save_file=True)
    HMD.distribution_plots(post_experiment_questionnaire,
                           post_columns_distribution_to_plot,
                           output_folder="output",
                           save_file=True)
