import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

WORKING_DIR = os.getcwd()
ANOMALY_FILE = 'credit-card-fraud.csv'

def download_file_from_gcs(source, destination):
    """Download files from GCS to WORKING_DIR/.

    Args:
        source: GCS path to the training data
        destination: GCS path to the validation data.
    Returns:
        The local data paths where the data is downloaded.
    """

    local_file_names = [destination]
    print("Local File Names: ", local_file_names)
    gcs_input_paths = [source]
    print("GCS Input Paths: ", gcs_input_paths)

    raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
                                  for local_file_name in local_file_names]
    print("Raw Local Files Data Paths: ", raw_local_files_data_paths)

    for i, gcs_input_path in enumerate(gcs_input_paths):
        if gcs_input_path:
            subprocess.check_call(['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

    return raw_local_files_data_paths

def _load_data(path):
    """Loads Credit-Card-Fraud Dataset locally or from Google Cloud.

    Args:
        path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')

    Returns:
        A Pandas DataFrame containing both features and target

    Raises:
        ValueError: In case path is not defined.
    """

    if not path:
        raise ValueError('No training file defined')
    if path.startswith('gs://'):
        download_file_from_gcs(path, destination=ANOMALY_FILE)
        path = ANOMALY_FILE

    fraud = pd.read_csv(path)

    return fraud


def preprocess(train_file):
    """Loads and Splits Credit-Card-Fraud Data into Training and Testing Sets

    Args:
        train_file: (str) the path to the training file

    Returns:
        tuple: training, validation, and test data.
    """

    fraud = _load_data(path=train_file)

    fraud['Time'] = StandardScaler().fit_transform(fraud['Time'].values.reshape(-1, 1))
    fraud['Amount'] = StandardScaler().fit_transform(fraud['Amount'].values.reshape(-1, 1))

    X = fraud.drop('Class', axis=1)
    y = fraud['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train = pd.concat([X_train, y_train], axis=1)

    train = train.loc[train['Class'] == 0]
    X_train = train.drop('Class', axis=1)
    y_train = train['Class']

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                      stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloader(X, batch_size):
    """Creates Pytorch Dataloaders for a given 2-D Array

    Args:
        X: (ndarray) A 2-D array of features
        batch_size: (int) The batch-size for the Dataloaders

    Returns:
        Pytorch DataLoader: A Dataloader object.
    """

    X_tensor = torch.FloatTensor(X)

    X_dl = torch.utils.data.DataLoader(X_tensor, batch_size)

    return X_dl