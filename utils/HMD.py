from custom_logger import CustomLogger
import math
import numpy as np
import os
import pandas as pd
from collections import defaultdict

logger = CustomLogger(__name__)  # use custom logger


class HMD_yaw():
    def __init__(self) -> None:
        pass

    def quaternion_to_euler(self, w, x, y, z):
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

    def group_files_by_video_id(self, data_folder, video_data):
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
