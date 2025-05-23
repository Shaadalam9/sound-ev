import numpy as np
import pandas as pd
import ast
from utils.HMD import HMD_yaw

HMD_class = HMD_yaw()


class Tools():
    def __init__(self) -> None:
        pass

    def average_dataframe_vectors_with_timestamp(self, df, column_name):
        result = []

        for idx, row in df.iterrows():
            timestamp = row["Timestamp"]
            row_means = []

            for col in df.columns:
                if col == "Timestamp":
                    continue
                val = row[col]
                try:
                    vec = ast.literal_eval(val) if isinstance(val, str) else val
                    if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                        row_means.append(np.mean(vec))
                except:  # noqa:E722
                    continue

            row_avg = np.mean(row_means) if row_means else np.nan
            result.append({"Timestamp": timestamp, column_name: row_avg})

        return pd.DataFrame(result)

    def extract_time_series_values(self, df):
        """
        Extracts a list of lists from a DataFrame where each inner list contains
        all the values for a specific timestamp, flattened from the list-encoded columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame with 'Timestamp' and string-encoded list columns.

        Returns:
            list of list: A list where each inner list contains values for one timestamp.
        """
        all_values_by_timestep = []

        for _, row in df.iterrows():
            row_values = []
            for col in df.columns:
                if col == 'Timestamp':
                    continue
                value = row[col]
                if pd.isna(value):
                    parsed_list = []  # or [None] or [0] depending on what makes sense for you
                else:
                    parsed_list = ast.literal_eval(value)
                row_values.extend(parsed_list)
            all_values_by_timestep.append(row_values)

        return all_values_by_timestep
