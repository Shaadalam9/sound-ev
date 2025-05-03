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
    HMD.read_slider_data(data_folder, mapping, output_folder)
    HMD.gender_distribution(intake_questionnaire, output_folder)
    HMD.age_distribution(intake_questionnaire, output_folder)
    HMD.demographic_distribution(intake_questionnaire, output_folder)
    HMD.avg_csv_files(data_folder, mapping)
    HMD.plot(mapping, column_name="TriggerValueRight")
    HMD.plot_yaw(mapping)
    HMD.plot_individual_csvs_plotly(["_output/slider_input_annoyance.csv",
                                     "_output/slider_input_info.csv",
                                     "_output/slider_input_noticeability.csv"], mapping)
