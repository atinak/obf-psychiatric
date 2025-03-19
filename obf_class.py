import torch
import torch.nn as nn
from scipy.stats import entropy as shannon_entropy

from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Union
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OBFDataset(Dataset):
    def __init__(self, data_dir: str, groups: List[str], transform=None, id_from_filename: bool = True, save: bool = False, seq_len: int = 1, use_seq: bool = False):
        self.save = save
        self.data_dir = data_dir
        self.groups = groups
        self.transform = transform
        self.id_from_filename = id_from_filename
        self.use_seq = use_seq
        self.seq_len = seq_len  # Sequence length
        self.data, self.label_encoder = self._load_and_preprocess_data()  # Load and preprocess
        self.scaler = StandardScaler()  # Initialize StandardScaler
        
        # Instead of transforming features here, we'll do it on-the-fly in __getitem__
        # This is generally better for flexibility and memory efficiency.
        self.features = self.data[['daily_mean', 'daily_std', 'daily_median', 'daily_zeros', 'daily_25th',
                                  'daily_75th', 'morning_mean', 'afternoon_mean', 'evening_mean',
                                  'night_mean', 'dominant_freq', 'power_dominant_freq', 'rolling_mean_2h',
                                  'rolling_std_2h', 'activity_lag_1', 'activity_lag_2',
                                  'mean_std_interaction', 'median_zeros_interaction',
                                   'log_daily_mean', 'sqrt_daily_std', 'square_daily_median','entropy','fractal_dimension']].values
        
        self.labels = self.data['label'].values
        self.features = torch.tensor(self.scaler.fit_transform(self.features), dtype=torch.float32) #to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Use long for CrossEntropyLoss
        # Reshape for sequence models

        if self.use_seq:
          # Reshape for sequence models only if use_seq is True
          self.features = self.features.reshape(-1, self.seq_len, self.features.shape[1])
        
        self.load_preprocessed_data('./preprocessed_data/')

    def _load_and_preprocess_data(self) -> Tuple[pd.DataFrame, LabelEncoder]:
        all_data = []
        for group in self.groups:
            data_files = glob.glob(os.path.join(self.data_dir, group, "*.csv"))
            info_file = os.path.join(self.data_dir, f"{group}-info.csv")
            if not data_files or not os.path.exists(info_file):
                logging.warning(f"Data/info files missing for group: {group}")
                continue

            all_group_data = []  # Collect all data for this group
            for file in data_files:
                try:
                    df = pd.read_csv(file)
                    if self.id_from_filename:
                        filename = os.path.basename(file)
                        subject_id = filename.split('_')[-1].split('.')[0]
                        try:
                            df['number'] = group + "_" + str(int(subject_id))
                        except ValueError:
                            logging.error(f"Invalid ID in filename: {filename}")
                            continue
                    elif 'number' not in df.columns:
                        logging.error(f"'number' missing and id_from_filename=False: {file}")
                        continue
                    all_group_data.append(df)  # Accumulate dataframes

                except Exception as e:
                    logging.error(f"Error reading file {file}: {e}")
                    continue

            if not all_group_data:
                logging.warning(f"No data loaded for group: {group}")
                continue
            data_df = pd.concat(all_group_data, ignore_index=True) #concat
            info_df = pd.read_csv(info_file)

            # Add the group before preprocessing
            data_df['group'] = group
            info_df['group'] = group  # Add to info_df as well

            # --- Preprocessing ---
            data_df['activity'] = data_df['activity'].ffill()
            data_df.dropna(subset=['timestamp', 'date'], inplace=True)

            try:
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                data_df['date'] = pd.to_datetime(data_df['date']).dt.date
            except ValueError as e:
                logging.error(f"Date/time conversion error: {e}")
                raise

            if 'number' in info_df.columns:
                if not pd.api.types.is_string_dtype(info_df['number']):
                    info_df['number'] = group + '_' + info_df['number'].astype(str)
            else:
                logging.error(f"'number' missing in info_df for group: {group}")
                continue #skip this group

            # Merge data and info
            merged_df = pd.merge(data_df, info_df, on=['number', 'group'], how='left', suffixes=('', '_info'))
            # merged_df.to_csv('./merged_df.csv', index=False)
            # --- Feature Engineering (Call the functions here) ---
            merged_df = calculate_time_of_day_features(merged_df)
            merged_df = calculate_frequency_domain_features(merged_df)
            merged_df = calculate_rolling_window_features(merged_df)
            merged_df = calculate_lag_features(merged_df)
            merged_df = calculate_nonlinear_transformations(merged_df)

            # Daily Aggregation (CORRECTED)
            daily_features = merged_df.groupby(['number', 'date', 'group'], observed=True).agg(
                daily_mean=('activity', 'mean'),
                daily_std=('activity', 'std'),
                daily_median=('activity', 'median'),
                daily_zeros=('activity', lambda x: (x == 0).sum()),
                daily_25th=('activity', lambda x: x.quantile(0.25)),
                daily_75th=('activity', lambda x: x.quantile(0.75)),
                morning_mean=('morning_mean', 'first'),
                afternoon_mean=('afternoon_mean', 'first'),
                evening_mean=('evening_mean', 'first'),
                night_mean=('night_mean', 'first'),
                dominant_freq=('dominant_freq', 'first'),
                power_dominant_freq=('power_dominant_freq', 'first'),
                rolling_mean_2h=('rolling_mean_2h', 'first'),
                rolling_std_2h=('rolling_std_2h', 'first'),
                activity_lag_1=('activity_lag_1', 'first'),
                activity_lag_2=('activity_lag_2', 'first'),
                entropy=('entropy','first'),
                fractal_dimension=('fractal_dimension','first')
            ).reset_index()

            # Interaction and Nonlinear Features (after aggregation)
            daily_features['mean_std_interaction'] = daily_features['daily_mean'] * daily_features['daily_std']
            daily_features['median_zeros_interaction'] = daily_features['daily_median'] * daily_features['daily_zeros']
            daily_features['log_daily_mean'] = np.log1p(daily_features['daily_mean'])
            daily_features['sqrt_daily_std'] = np.sqrt(daily_features['daily_std'])
            daily_features['square_daily_median'] = daily_features['daily_median'] ** 2

            all_data.append(daily_features)


        final_df = pd.concat(all_data, ignore_index=True)
        final_df.dropna(inplace=True)  # Drop rows with any NaN

        # --- Label Encoding ---
        label_encoder = LabelEncoder()
        final_df['label'] = label_encoder.fit_transform(final_df['group']) #create encoded labels
        if self.save: final_df.to_csv('./all_features.csv', index=False)
        return final_df, label_encoder

    def plot_activity_heatmap(self, group: str, ax: Optional[plt.Axes] = None) -> None:
        """
        Plots the motor activity heatmap for a specified group.

        Args:
            group: The name of the group to plot (e.g., 'control', 'depression').
            ax: Optional matplotlib Axes object to plot on.  If None, creates a new figure.
        """
        
        df = self.preprocessed_data.get(group)
        if df is None:
            logging.error(f"Group '{group}' not found for heatmap plotting.")
            return

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
          df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create 'hour' and 'day_of_week' columns
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

        # Pivot the data to create the heatmap structure
        heatmap_data = df.pivot_table(
            index='hour',
            columns='day_of_week',
            values='activity',
            aggfunc='mean'  # Average activity for each hour/day combination
        )

        # Reorder columns to be in standard week order
        days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        heatmap_data = heatmap_data[days_order]

        # Create the heatmap using seaborn
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure() #get the figure

        sns.heatmap(heatmap_data, cmap="viridis", ax=ax)
        ax.set_title(f'Motor Activity Heatmap for {group}')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Hour of the Day')

    def plot_all_heatmaps(self, groups: Optional[List[str]] = None) -> None:
        """
        Plots heatmaps for multiple groups or all available groups
        """
        
        if groups is None:
          groups = list(self.preprocessed_data.keys())

        num_groups = len(groups)
        if num_groups == 0:
            print("No groups to plot.")
            return

        fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 5)) #adjust fig size
        if num_groups == 1:
          axes = [axes] #to handle the single subplot case

        for i, group in enumerate(groups):
            self.plot_activity_heatmap(group, ax=axes[i])

        plt.tight_layout()
        plt.show()

    def save_preprocessed_data(self, directory: str):
        """Saves the preprocessed data to CSV files."""
        os.makedirs(directory, exist_ok=True)
        for group, df in self.preprocessed_data.items():
            filepath = os.path.join(directory, f"{group}_preprocessed.csv")
            df.to_csv(filepath, index=False)
            
    def load_preprocessed_data(self, directory: str):
        """Loads preprocessed data from CSV files."""
        self.preprocessed_data = {}  # Clear existing
        for group in ['adhd', 'clinical', 'control', 'depression', 'schizophrenia']:
            filepath = os.path.join(directory, f"{group}_preprocessed.csv")
            if os.path.exists(filepath):
                try:
                    self.preprocessed_data[group] = pd.read_csv(filepath)
                    # Convert timestamp and date columns back to datetime
                    self.preprocessed_data[group]['timestamp'] = pd.to_datetime(self.preprocessed_data[group]['timestamp'])
                    self.preprocessed_data[group]['date'] = pd.to_datetime(self.preprocessed_data[group]['date']).dt.date
                    logging.info(f"Loaded preprocessed data for group: {group}")
                except Exception as e:
                    logging.error(f"Error loading preprocessed data for {group}: {e}")
            else:
                logging.warning(f"Preprocessed data file not found for group: {group}")
                
    def __len__(self):
        if self.use_seq:
            return len(self.data) - self.seq_len + 1  # Correct length for sequences
        else:
            return len(self.data)


    def __getitem__(self, idx):
        if self.use_seq:
            # Handle sequence logic (get a sequence of length seq_len)
            end_idx = idx + self.seq_len
            if end_idx > len(self.data):  # Prevent out-of-bounds access - #CORRECTED
                end_idx = len(self.data)
                start_idx = end_idx - self.seq_len
                if start_idx < 0: # sequence too short
                    start_idx = 0
                    end_idx = min(self.seq_len, len(self.data))
            else:
                start_idx = idx
            features = self.features[start_idx:end_idx]
            label = self.labels[end_idx - 1] #label of the last day.

        else:
            features = self.features[idx]
            label = self.labels[idx]
        return features, label

    def get_classes(self):
        return self.label_encoder.classes_

    
# --- Feature Engineering Functions ---
def calculate_time_of_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean activity for different times of day."""
    df['hour'] = df['timestamp'].dt.hour
    df['morning_mean'] = df[(df['hour'] >= 6) & (df['hour'] < 12)]['activity']
    df['afternoon_mean'] = df[(df['hour'] >= 12) & (df['hour'] < 18)]['activity']
    df['evening_mean'] = df[(df['hour'] >= 18) & (df['hour'] < 24)]['activity']
    df['night_mean'] = df[(df['hour'] >= 0) & (df['hour'] < 6)]['activity']
    return df

def calculate_frequency_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features using the Fast Fourier Transform (FFT)."""
    # Group by 'number' and 'date' and apply FFT to each group's 'activity'
    fft_results = df.groupby(['number', 'date'])['activity'].apply(lambda x: np.fft.fft(x.values))

    # Function to calculate dominant frequency and its power for a single FFT result
    def get_dominant_frequency(fft_result):
        fft_abs = np.abs(fft_result)
        dominant_freq_index = np.argmax(fft_abs[1:]) + 1  # Exclude 0 frequency
        power_dominant_freq = fft_abs[dominant_freq_index]

        # Calculate the actual frequency.  Need to know sampling rate.
        # Assuming 1 sample per minute, the sampling rate is 1/60 Hz.
        sampling_rate = 1/60  # Samples per second
        freqs = np.fft.fftfreq(len(fft_result), d=1/sampling_rate/60) #get freqs in minutes
        dominant_freq = freqs[dominant_freq_index]
        return dominant_freq, power_dominant_freq

    # Apply the function to get dominant freq and power, then join back to the original
    dominant_freqs = fft_results.apply(get_dominant_frequency).apply(pd.Series)
    dominant_freqs.columns = ['dominant_freq', 'power_dominant_freq']
    dominant_freqs = dominant_freqs.reset_index()  # Make 'number' and 'date' regular columns
    df = pd.merge(df, dominant_freqs, on=['number', 'date'], how='left')

    return df

def calculate_rolling_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rolling mean and standard deviation."""
    df['rolling_mean_2h'] = df.groupby(['number', 'date'])['activity'].transform(lambda x: x.rolling(window=120, min_periods=1).mean())  # 120 min
    df['rolling_std_2h'] = df.groupby(['number', 'date'])['activity'].transform(lambda x: x.rolling(window=120, min_periods=1).std())
    return df

def calculate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates lagged activity values."""
    df['activity_lag_1'] = df.groupby(['number', 'date'])['activity'].transform(lambda x: x.shift(1))
    df['activity_lag_2'] = df.groupby(['number', 'date'])['activity'].transform(lambda x: x.shift(2))
    return df



def calculate_nonlinear_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate non-linear features like entropy and fractal dimension."""
    # --- Shannon Entropy ---
    def calculate_entropy(activity_series):
        """Calculate Shannon Entropy for a given activity series."""
        if len(activity_series) == 0:
            return np.nan
        # Normalize the activity series to create a probability distribution
        activity_series = activity_series / np.sum(activity_series)
        return shannon_entropy(activity_series)

    # --- Higuchi Fractal Dimension ---
    def calculate_higuchi_fractal_dimension(activity_series, k_max=10):
        """Calculate Higuchi Fractal Dimension for a given activity series."""
        if len(activity_series) == 0:
            return np.nan
        
        # Convert pandas Series to numpy array for integer indexing
        activity_series = activity_series.values
        
        N = len(activity_series)
        L = []
        for k in range(1, k_max + 1):
            Lk = 0
            for m in range(k):
                # Calculate the length of the curve for each segment
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(activity_series[m + i * k] - activity_series[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / (int((N - m) / k) * k ** 2)
                Lk += Lmk
            L.append(np.log(Lk / k))
        # Fit a line to the log-log plot to estimate the fractal dimension
        x = np.log(np.arange(1, k_max + 1))
        y = np.array(L)
        slope, _ = np.polyfit(x, y, 1)
        return slope

    # Apply entropy and fractal dimension calculations
    df['entropy'] = df.groupby(['number', 'date'])['activity'].transform(calculate_entropy)
    df['fractal_dimension'] = df.groupby(['number', 'date'])['activity'].transform(calculate_higuchi_fractal_dimension)

    return df

