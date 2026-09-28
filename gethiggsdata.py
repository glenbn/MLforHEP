from google.colab import drive
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Mount your Google Drive
drive.mount('/content/drive')

# 2. Define path to your dataset stored in Google Drive
extracted_csv_gz_path = "/content/drive/MyDrive/MLforHEP/higgs.csv.gz"

# 2. Extract the ZIP file to disk (if the gzipped CSV is not already extracted)
if not os.path.exists(extracted_csv_gz_path):
    print(f"Extracting {extracted_csv_gz_path} from {extracted_csv_gz_path} archive...")
    # Unzip higgs.zip, which should contain HIGGS.csv.gz
    !unzip -q -o {extracted_csv_gz_path }
    print("Extraction complete!")
else:
    print(f"{extracted_csv_gz_path} already exists. Skipping extraction from {extracted_csv_gz_path}.")


# Define column names
columns = [
    'class_label',
    'lepton_pT', 'lepton_eta', 'lepton_phi',
    'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
    'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
    'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
    'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

# 3. Read directly from the unzipped CSV (Pandas handles .gz decompression)
print(f"Reading 100,000 rows from {extracted_csv_gz_path}...")
df = pd.read_csv(extracted_csv_gz_path, names=columns, nrows=100000)

print(f"Success! Data shape: {df.shape}")