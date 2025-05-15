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
    # HMD.read_slider_data(data_folder,
    #                      mapping,
    #                      output_folder)
    # HMD.plot_column_distribution(intake_questionnaire,
    #                              intake_columns_to_plot,
    #                              output_folder="output",
    #                              save_file=True)
    # HMD.plot_column_distribution(post_experiment_questionnaire,
    #                              post_columns_to_plot,
    #                              output_folder="output",
    #                              save_file=True)
    # HMD.distribution_plots(intake_questionnaire,
    #                        intake_columns_distribution_to_plot,
    #                        output_folder="output",
    #                        save_file=True)
    # HMD.distribution_plots(post_experiment_questionnaire,
    #                        post_columns_distribution_to_plot,
    #                        output_folder="output",
    #                        save_file=True)
    HMD.avg_csv_files(data_folder,
                      mapping)
    # HMD.plot_column(mapping,
    #                 column_name="TriggerValueRight")
    HMD.plot_yaw(mapping)
    # HMD.plot_individual_csvs_plotly(["_output/slider_input_annoyance.csv",
    #                                  "_output/slider_input_info.csv",
    #                                  "_output/slider_input_noticeability.csv"], mapping)
    # HMD.plot_yaw_angle_histograms(mapping, angle=10, num_bins=30, calibrate=False, smoothen_filter_param=False)
