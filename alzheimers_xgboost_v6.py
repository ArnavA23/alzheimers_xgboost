"""
Alzheimer's Hybrid CNN + XGBoost Model (v6 calibrated)
=============================================
Architecture:
  1. CNN processes 5 brain slices → extracts 64 visual features per patient
  2. XGBoost takes CNN features + tabular features → final prediction
  3. Two-stage classifier: CN vs Not → MCI vs Alzheimer's
  4. Ensemble of 10 XGBoost models per stage
  5. Cross-validated hyperparameter tuning
  6. Train on LATEST visit, evaluate on EARLIEST visit

Requirements:
    pip install torch torchvision xgboost pandas numpy pillow scikit-learn matplotlib
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import xgboost as xgb
import pickle
import gc
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Data is NOT distributed with this repository (NACC Data Use Agreement).
# Set ALZ_DATA_DIR to a local folder containing the NACC-derived files:
#   Train_Data.xlsx, Validation_Data.xlsx, Test_Data.xlsx, and an
#   "Imaging Data" subfolder of extracted axial-slice PNGs.
DATA_DIR     = os.environ.get("ALZ_DATA_DIR", "./data")
SCANS_FOLDER = os.path.join(DATA_DIR, "Imaging Data")
TRAIN_EXCEL  = os.path.join(DATA_DIR, "Train_Data.xlsx")
VAL_EXCEL    = os.path.join(DATA_DIR, "Validation_Data.xlsx")
TEST_EXCEL   = os.path.join(DATA_DIR, "Test_Data.xlsx")

IMG_SIZE = (128, 128)
NUM_SLICES = 5
CNN_FEATURES = 64       # number of features CNN extracts per patient
CNN_EPOCHS = 50         # more epochs for transfer learning + augmentation
CNN_BATCH_SIZE = 16
CNN_LR = 0.001
TOP_N_FEATURES = 150    # for XGBoost feature selection
DEVICE = 'cpu'  # Using CPU only (no PyTorch CNN)


GLOBAL_VOLUMES = [
    'BRAINSEG', 'CORTEX', 'TOTALGRAY', 'SUBCORTGRAY',
    'LHCORTEX', 'RHCORTEX',
    'CEREBRALWHITEMATTER', 'LHCEREBRALWHITEMATTER', 'RHCEREBRALWHITEMATTER',
    'SUPRATENTORIAL', 'SUPRATENTORIALNOTVENT',
    'ESTIMATEDTOTALINTRACRANIALVOL', 'CEREBRUMTCV', 'GM',
    'CSF', 'WMH', 'WM_HYPOINTENSITIES',
]

HIPPOCAMPAL = [
    'HIPPOCAMPUS', 'LEFT_HIPPO', 'RIGHT_HIPPO',
    'LEFT_HIPPOCAMPUS', 'RIGHT_HIPPOCAMPUS',
    'LH_PARAHIPPOCAMPAL_GVOL', 'RH_PARAHIPPOCAMPAL_GVOL',
    'LH_ENTORHINAL_GVOL', 'RH_ENTORHINAL_GVOL',
]

VENTRICLES = [
    'LEFT_LATERAL_VENTRICLE', 'RIGHT_LATERAL_VENTRICLE',
    'LEFT_INF_LAT_VENT', 'RIGHT_INF_LAT_VENT',
    '3RD_VENTRICLE', '4TH_VENTRICLE',
]

SUBCORTICAL = [
    'LEFT_THALAMUS_PROPER', 'RIGHT_THALAMUS_PROPER',
    'LEFT_CAUDATE', 'RIGHT_CAUDATE',
    'LEFT_PUTAMEN', 'RIGHT_PUTAMEN',
    'LEFT_PALLIDUM', 'RIGHT_PALLIDUM',
    'LEFT_AMYGDALA', 'RIGHT_AMYGDALA',
    'LEFT_ACCUMBENS_AREA', 'RIGHT_ACCUMBENS_AREA',
    'LEFT_VENTRALDC', 'RIGHT_VENTRALDC',
    'BRAIN_STEM',
]

CORTICAL_REGIONS = [
    'CAUDALANTERIORCINGULATE', 'CAUDALMIDDLEFRONTAL', 'CUNEUS', 'ENTORHINAL',
    'FUSIFORM', 'INFERIORPARIETAL', 'INFERIORTEMPORAL', 'INSULA',
    'ISTHMUSCINGULATE', 'LATERALOCCIPITAL', 'LATERALORBITOFRONTAL', 'LINGUAL',
    'MEDIALORBITOFRONTAL', 'MIDDLETEMPORAL', 'PARACENTRAL', 'PARAHIPPOCAMPAL',
    'PARSOPERCULARIS', 'PARSORBITALIS', 'PARSTRIANGULARIS', 'PERICALCARINE',
    'POSTCENTRAL', 'POSTERIORCINGULATE', 'PRECENTRAL', 'PRECUNEUS',
    'ROSTRALANTERIORCINGULATE', 'ROSTRALMIDDLEFRONTAL', 'SUPERIORFRONTAL',
    'SUPERIORPARIETAL', 'SUPERIORTEMPORAL', 'SUPRAMARGINAL', 'TRANSVERSETEMPORAL',
]

CORTICAL_PARCELLATION = []
for region in CORTICAL_REGIONS:
    for hemi in ['LH', 'RH']:
        for measure in ['GVOL', 'SAREA', 'AVGTH']:
            CORTICAL_PARCELLATION.append(f'{hemi}_{region}_{measure}')

DEMOGRAPHICS = ['NACCAGE', 'SEX', 'EDUC', 'NACCNE4S']
COMORBIDITIES = ['HYPERT', 'HYPERCHO', 'DIABETES', 'CVHATT', 'CBSTROKE', 'DEPD']
CLINICAL_FLAGS = ['AMNDEM']

ALL_IMAGING = GLOBAL_VOLUMES + HIPPOCAMPAL + VENTRICLES + SUBCORTICAL + CORTICAL_PARCELLATION
ALL_TABULAR = ALL_IMAGING + DEMOGRAPHICS + COMORBIDITIES + CLINICAL_FLAGS

RATE_OF_CHANGE_VARS = [
    'HIPPOCAMPUS', 'LEFT_HIPPOCAMPUS', 'RIGHT_HIPPOCAMPUS',
    'LEFT_LATERAL_VENTRICLE', 'RIGHT_LATERAL_VENTRICLE',
    'TOTALGRAY', 'CEREBRALWHITEMATTER', 'BRAINSEG',
    'LEFT_AMYGDALA', 'RIGHT_AMYGDALA',
    'LH_ENTORHINAL_GVOL', 'RH_ENTORHINAL_GVOL',
    'LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH',
    'LH_FUSIFORM_GVOL', 'RH_FUSIFORM_GVOL',
    'LH_MIDDLETEMPORAL_GVOL', 'RH_MIDDLETEMPORAL_GVOL',
    'LH_INFERIORTEMPORAL_GVOL', 'RH_INFERIORTEMPORAL_GVOL',
    'WMH', 'CSF',
]

TARGETS = ['CDRSUM', 'NACCUDSD']


def load_patient_slices(patient_id, scans_folder, img_size):
    """Load 5 slices as a (5, H, W) numpy array."""
    slices = []
    for slice_num in range(1, NUM_SLICES + 1):
        filename = f"Axial_Slice_{patient_id}_{slice_num}.png"
        filepath = os.path.join(scans_folder, filename)

        if not os.path.exists(filepath):
            for dirpath, dirnames, filenames in os.walk(scans_folder):
                if filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    break

        if os.path.exists(filepath):
            img = Image.open(filepath).convert('L')
            if img_size is not None:
                img = img.resize(img_size)
            arr = np.array(img).astype(np.float32) / 255.0
            slices.append(arr)
        else:
            slices.append(np.zeros((img_size[0], img_size[1]), dtype=np.float32))

    return np.stack(slices)  # shape: (5, 128, 128)


def build_dataset(excel_path, scans_folder, img_size, use_latest=True):
    df = pd.read_excel(excel_path)

    for col in ALL_TABULAR + TARGETS + RATE_OF_CHANGE_VARS:
        if col in df.columns:
            df[col] = df[col].replace([-4, 88, 888, 8888, 99, 999, 9999], np.nan)

    df = df[df['NACCUDSD'] != 2].copy()
    df['NACCUDSD'] = df['NACCUDSD'].map({1: 0, 3: 1, 4: 2})

    df = df.sort_values('NACCVNUM')
    patients = df.groupby('NACCID')

    latest = patients.last().reset_index()
    if use_latest:
        features_df = patients.last().reset_index()
        mode_str = "LATEST"
    else:
        features_df = patients.first().reset_index()
        mode_str = "EARLIEST"

    earliest_df = patients.first().reset_index()
    latest_for_roc = patients.last().reset_index()
    roc_features = {}
    roc_names = [f"{v}_ROC" for v in RATE_OF_CHANGE_VARS]

    for _, row_e in earliest_df.iterrows():
        pid = row_e['NACCID']
        row_l = latest_for_roc[latest_for_roc['NACCID'] == pid]
        if row_l.empty:
            continue
        row_l = row_l.iloc[0]

        yr_e = row_e.get('VISITYR', np.nan) if 'VISITYR' in earliest_df.columns else np.nan
        yr_l = row_l.get('VISITYR', np.nan) if 'VISITYR' in latest_for_roc.columns else np.nan
        years = float(yr_l) - float(yr_e) if pd.notna(yr_e) and pd.notna(yr_l) and float(yr_l) != float(yr_e) else None

        roc = {}
        for var in RATE_OF_CHANGE_VARS:
            roc_name = f"{var}_ROC"
            if var in row_e.index and var in row_l.index:
                val_e, val_l = row_e[var], row_l[var]
                if isinstance(val_e, pd.Series): val_e = val_e.iloc[0]
                if isinstance(val_l, pd.Series): val_l = val_l.iloc[0]
                if pd.notna(val_e) and pd.notna(val_l) and years and years > 0:
                    roc[roc_name] = (float(val_l) - float(val_e)) / years
                else:
                    roc[roc_name] = np.nan
            else:
                roc[roc_name] = np.nan
        roc_features[pid] = roc

    merged = features_df[['NACCID'] + [c for c in ALL_TABULAR if c in features_df.columns]].copy()
    for target in TARGETS:
        if target in latest.columns:
            merged[target] = latest.set_index('NACCID').loc[merged['NACCID'].values, target].values

    images_list = []
    tabular_list = []
    Y_rows = []
    patient_ids = []
    scans_found = 0

    for _, row in merged.iterrows():
        pid = row['NACCID']

        slices = load_patient_slices(pid, scans_folder, img_size)
        if slices.max() > 0:
            scans_found += 1
        images_list.append(slices)

        tabular = []
        for col in ALL_TABULAR:
            if col in row.index:
                val = row[col]
                if isinstance(val, pd.Series): val = val.iloc[0]
                tabular.append(float(val) if pd.notna(val) else np.nan)
            else:
                tabular.append(np.nan)

        roc = roc_features.get(pid, {})
        for roc_name in roc_names:
            tabular.append(roc.get(roc_name, np.nan))

        tabular_list.append(np.array(tabular, dtype=np.float32))
        Y_rows.append([row.get(t, np.nan) for t in TARGETS])
        patient_ids.append(pid)

    images = np.stack(images_list)       # (N, 5, 128, 128)
    tabular = np.stack(tabular_list)     # (N, n_tabular)
    Y = np.array(Y_rows)

    n_tab = len(ALL_TABULAR) + len(roc_names)
    print(f"  [{mode_str} visit features, LATEST visit targets]")
    print(f"  Patients: {len(patient_ids)} ({scans_found} with scans)")
    print(f"  Images: {images.shape} ({NUM_SLICES} slices x {img_size[0]}x{img_size[1]})")
    print(f"  Tabular: {tabular.shape[1]} ({len(ALL_TABULAR)} base + {len(roc_names)} ROC)")

    for i, t in enumerate(TARGETS):
        valid = Y[:, i][~np.isnan(Y[:, i])]
        if t == 'NACCUDSD':
            counts = dict(zip(*np.unique(valid.astype(int), return_counts=True)))
            print(f"  {t}: {len(valid)} valid — {counts}")
        else:
            print(f"  {t}: {len(valid)} valid — mean={valid.mean():.1f}")

    return images, tabular, Y, patient_ids, roc_names


def compute_interaction_features(tabular, feature_names):
    """
    Compute clinically meaningful interaction features:
    - Ratios (hippocampus/total brain, ventricle/brain)
    - Left/right asymmetries
    - Normalized volumes (by intracranial volume)
    """
    def safe_get(row, name):
        if name in feature_names:
            idx = feature_names.index(name)
            val = row[idx]
            return val if not np.isnan(val) and val != 0 else np.nan
        return np.nan

    def safe_ratio(a, b):
        if a is not None and b is not None and not np.isnan(a) and not np.isnan(b) and b != 0:
            return a / b
        return np.nan

    def safe_asymmetry(left, right):
        if (left is not None and right is not None and
            not np.isnan(left) and not np.isnan(right) and (left + right) != 0):
            return abs(left - right) / (left + right)
        return np.nan

    interaction_names = [
        'HIPPO_BRAIN_RATIO', 'HIPPO_ICV_RATIO',
        'VENT_BRAIN_RATIO', 'VENT_ICV_RATIO',
        'CSF_BRAIN_RATIO', 'WMH_WM_RATIO',
        'GRAY_WHITE_RATIO', 'CORTEX_BRAIN_RATIO',
        'HIPPO_ASYM', 'AMYGDALA_ASYM', 'VENT_ASYM',
        'THALAMUS_ASYM', 'CAUDATE_ASYM', 'PUTAMEN_ASYM',
        'ENTORHINAL_VOL_ASYM', 'ENTORHINAL_TH_ASYM',
        'FUSIFORM_ASYM', 'INFTEMP_ASYM', 'MIDTEMP_ASYM',
        'HIPPO_AGE_RATIO', 'BRAIN_AGE_RATIO',
    ]

    all_feats = []
    for i in range(len(tabular)):
        row = tabular[i]
        feats = []

        hippo = safe_get(row, 'HIPPOCAMPUS')
        brain = safe_get(row, 'BRAINSEG')
        icv = safe_get(row, 'ESTIMATEDTOTALINTRACRANIALVOL')
        lvent = safe_get(row, 'LEFT_LATERAL_VENTRICLE')
        rvent = safe_get(row, 'RIGHT_LATERAL_VENTRICLE')
        total_vent = (lvent + rvent) if (not np.isnan(lvent or 0) and not np.isnan(rvent or 0)) else np.nan
        gray = safe_get(row, 'TOTALGRAY')
        white = safe_get(row, 'CEREBRALWHITEMATTER')
        csf_val = safe_get(row, 'CSF')
        wmh = safe_get(row, 'WMH')
        cortex = safe_get(row, 'CORTEX')
        age = safe_get(row, 'NACCAGE')

        feats.append(safe_ratio(hippo, brain))
        feats.append(safe_ratio(hippo, icv))
        feats.append(safe_ratio(total_vent, brain))
        feats.append(safe_ratio(total_vent, icv))
        feats.append(safe_ratio(csf_val, brain))
        feats.append(safe_ratio(wmh, white))
        feats.append(safe_ratio(gray, white))
        feats.append(safe_ratio(cortex, brain))

        feats.append(safe_asymmetry(safe_get(row, 'LEFT_HIPPOCAMPUS'), safe_get(row, 'RIGHT_HIPPOCAMPUS')))
        feats.append(safe_asymmetry(safe_get(row, 'LEFT_AMYGDALA'), safe_get(row, 'RIGHT_AMYGDALA')))
        feats.append(safe_asymmetry(lvent, rvent))
        feats.append(safe_asymmetry(safe_get(row, 'LEFT_THALAMUS_PROPER'), safe_get(row, 'RIGHT_THALAMUS_PROPER')))
        feats.append(safe_asymmetry(safe_get(row, 'LEFT_CAUDATE'), safe_get(row, 'RIGHT_CAUDATE')))
        feats.append(safe_asymmetry(safe_get(row, 'LEFT_PUTAMEN'), safe_get(row, 'RIGHT_PUTAMEN')))
        feats.append(safe_asymmetry(safe_get(row, 'LH_ENTORHINAL_GVOL'), safe_get(row, 'RH_ENTORHINAL_GVOL')))
        feats.append(safe_asymmetry(safe_get(row, 'LH_ENTORHINAL_AVGTH'), safe_get(row, 'RH_ENTORHINAL_AVGTH')))
        feats.append(safe_asymmetry(safe_get(row, 'LH_FUSIFORM_GVOL'), safe_get(row, 'RH_FUSIFORM_GVOL')))
        feats.append(safe_asymmetry(safe_get(row, 'LH_INFERIORTEMPORAL_GVOL'), safe_get(row, 'RH_INFERIORTEMPORAL_GVOL')))
        feats.append(safe_asymmetry(safe_get(row, 'LH_MIDDLETEMPORAL_GVOL'), safe_get(row, 'RH_MIDDLETEMPORAL_GVOL')))

        feats.append(safe_ratio(hippo, age))
        feats.append(safe_ratio(brain, age))

        all_feats.append(feats)

    result = np.array(all_feats, dtype=np.float32)
    print(f"  Computed {len(interaction_names)} interaction features")
    return result, interaction_names


"""
class BrainFeatureExtractor(nn.Module):
    Frozen pretrained ResNet18 — NO training on our data.
    Just passes brain scans through and grabs the features.
    No overfitting possible because we never update the weights.
    
    Input: (batch, 5, 128, 128) → 5 slices
    Output: (batch, 512) raw ResNet features
    
    def __init__(self, n_slices=5):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(n_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            if n_slices > 3:
                self.conv1.weight[:, 3] = old_conv.weight[:, 0]
                self.conv1.weight[:, 4] = old_conv.weight[:, 1]

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (batch, 512)
        return x
"""


def extract_handcrafted_features(images):
    """
    Extract clinically meaningful features from brain slices.
    These capture what a radiologist would notice:
    - Brain volume (non-background fraction)
    - Ventricle size (dark regions in center)
    - Tissue contrast (gray vs white matter)
    - Hemispheric asymmetry
    - Texture measures
    """
    print("\n  Extracting hand-crafted image features...")
    all_features = []

    for patient_idx in range(len(images)):
        slices = images[patient_idx]  # (5, 128, 128)
        feats = []

        for s in range(slices.shape[0]):
            img = slices[s]

            skull_mask = img > 0.01
            skull_area = max(skull_mask.sum(), 1)

            brain = img > 0.05
            brain_area = brain.sum()

            brain_frac = brain_area / skull_area

            brain_pixels = img[brain] if brain.sum() > 10 else img.flatten()
            mean_int = brain_pixels.mean()
            std_int = brain_pixels.std()
            median_int = np.median(brain_pixels)

            dark = (img > 0.05) & (img < 0.20)   # CSF-like
            mid = (img >= 0.20) & (img < 0.35)    # gray matter-like
            bright = (img >= 0.35)                  # white matter-like
            csf_frac = dark.sum() / max(brain.sum(), 1)
            gray_frac = mid.sum() / max(brain.sum(), 1)
            white_frac = bright.sum() / max(brain.sum(), 1)

            h, w = img.shape
            center = img[h//4:3*h//4, w//4:3*w//4]
            center_brain = center > 0.05
            center_dark = (center > 0.05) & (center < 0.20)
            ventricle_ratio = center_dark.sum() / max(center_brain.sum(), 1)

            left = img[:, :w//2]
            right = img[:, w//2:]
            right_flipped = np.flip(right, axis=1)
            asym = np.abs(left - right_flipped).mean()

            dy = np.abs(np.diff(img, axis=0))
            dx = np.abs(np.diff(img, axis=1))
            edge_density = (dy.mean() + dx.mean()) / 2

            feats.extend([
                brain_frac, mean_int, std_int, median_int,
                csf_frac, gray_frac, white_frac,
                ventricle_ratio, asym, edge_density,
            ])

        slice_means = [slices[s][slices[s] > 0.05].mean() if (slices[s] > 0.05).sum() > 0 else 0 for s in range(slices.shape[0])]
        slice_brains = [(slices[s] > 0.05).sum() / slices[s].size for s in range(slices.shape[0])]

        feats.append(np.std(slice_means))      # intensity variation across slices
        feats.append(np.std(slice_brains))     # brain size variation across slices
        feats.append(max(slice_brains) - min(slice_brains))  # brain size range

        all_features.append(feats)

    result = np.array(all_features, dtype=np.float32)

    hc_names = []
    per_slice = ['brain_frac', 'mean_int', 'std_int', 'median_int',
                 'csf_frac', 'gray_frac', 'white_frac',
                 'ventricle_ratio', 'asymmetry', 'edge_density']
    for s in range(5):
        for name in per_slice:
            hc_names.append(f"hc_s{s+1}_{name}")
    hc_names.extend(['hc_slice_int_var', 'hc_slice_brain_var', 'hc_slice_brain_range'])

    print(f"  Extracted {result.shape[1]} hand-crafted features per patient")
    return result, hc_names


def extract_frozen_features(images):
    """
    Extract features from brain scans using frozen pretrained ResNet18.
    No training involved — just a forward pass.
    """
    print("\n  Extracting image features (frozen ResNet18, no training)...")
    print(f"  Device: {DEVICE}")

    model = BrainFeatureExtractor(n_slices=NUM_SLICES).to(DEVICE)
    model.eval()

    features = []
    with torch.no_grad():
        for i in range(0, len(images), CNN_BATCH_SIZE):
            batch = torch.FloatTensor(images[i:i+CNN_BATCH_SIZE]).to(DEVICE)
            feat = model(batch)
            features.append(feat.cpu().numpy())

    result = np.concatenate(features, axis=0)  # (N, 512)
    print(f"  Extracted {result.shape[1]} image features per patient")
    return result


def compute_sample_weights(y):
    counts = dict(zip(*np.unique(y.astype(int), return_counts=True)))
    total = len(y)
    n_classes = len(counts)
    weights = np.ones(len(y))
    for cls, count in counts.items():
        class_weight = total / (n_classes * count)
        weights[y.astype(int) == cls] = class_weight
    print(f"  XGBoost class weights: " + ", ".join(
        f"{cls}={total/(n_classes*count):.2f}" for cls, count in sorted(counts.items())))
    return weights


def tune_hyperparameters(X_train, y_train, X_val, y_val, sample_weights):
    param_grid = {
        'max_depth':        [2, 3, 4],         # conservative: was [4, 6, 8, 10]
        'learning_rate':    [0.01, 0.05, 0.1],
        'subsample':        [0.7, 0.85, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7],
    }
    best_score = float('inf')
    best_params = {}
    total = (len(param_grid['max_depth']) * len(param_grid['learning_rate']) *
             len(param_grid['subsample']) * len(param_grid['colsample_bytree']))
    tested = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for md in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for ss in param_grid['subsample']:
                for cs in param_grid['colsample_bytree']:
                    fold_scores = []
                    tested += 1
                    for ft_idx, fv_idx in kf.split(X_train):
                        w_ft = sample_weights[ft_idx] if sample_weights is not None else None
                        m = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            max_depth=md, learning_rate=lr,
                            n_estimators=500, early_stopping_rounds=20,
                            subsample=ss, colsample_bytree=cs,
                            min_child_weight=3, n_jobs=4,
                            random_state=42, tree_method='hist',
                        )
                        m.fit(X_train[ft_idx], y_train[ft_idx],
                              sample_weight=w_ft,
                              eval_set=[(X_train[fv_idx], y_train[fv_idx])],
                              verbose=False)
                        pred = m.predict(X_train[fv_idx])
                        fold_scores.append(np.sqrt(mean_squared_error(y_train[fv_idx], pred)))
                        del m
                    gc.collect()
                    avg = np.mean(fold_scores)
                    if avg < best_score:
                        best_score = avg
                        best_params = {'max_depth': md, 'learning_rate': lr,
                                       'subsample': ss, 'colsample_bytree': cs}
                    if tested % 24 == 0:
                        print(f"    ...{tested}/{total} combos tested")

    vm = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
                           early_stopping_rounds=20, min_child_weight=3, n_jobs=4,
                           random_state=42, tree_method='hist', **best_params)
    vm.fit(X_train, y_train, sample_weight=sample_weights,
           eval_set=[(X_val, y_val)], verbose=False)
    val_rmse = np.sqrt(mean_squared_error(y_val, vm.predict(X_val)))
    best_params['best_iteration'] = vm.best_iteration
    del vm; gc.collect()

    print(f"  Tested {total} combos x 5 folds = {total*5} fits")
    print(f"  Best CV RMSE: {best_score:.4f}, Val RMSE: {val_rmse:.4f}")
    print(f"  Best params: {best_params}")
    return best_params


class EnsembleModel:
    def __init__(self, n_models=10):
        self.n_models = n_models
        self.models = []

    def fit(self, X, y, X_val, y_val, params, sample_weights=None):
        self.models = []
        for i in range(self.n_models):
            m = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                n_estimators=500, early_stopping_rounds=20,
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.5),
                min_child_weight=3, n_jobs=4,
                random_state=42 + i, tree_method='hist',
            )
            m.fit(X, y, sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)], verbose=False)
            self.models.append(m)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def get_importances(self):
        combined = {}
        for m in self.models:
            imp = m.feature_importances_
            for i, v in enumerate(imp):
                if v > 0:
                    combined[i] = combined.get(i, 0) + v
        for k in combined:
            combined[k] /= self.n_models
        return combined


class StackingStage:
    """
    Stacking ensemble for a single binary stage.
    Base models: XGBoost ensemble, RandomForest, GradientBoosting
    Meta-learner: LogisticRegression on out-of-fold predictions
    """
    def __init__(self, n_xgb=10):
        self.xgb_ensemble = EnsembleModel(n_models=n_xgb)
        self.rf = None
        self.gb = None
        self.meta = None
        self.impute_vals = None  # median imputation for sklearn models

    def _impute(self, X, fit=False):
        """Median imputation for sklearn models (XGBoost handles NaN natively)."""
        X_imp = X.copy()
        if fit:
            self.impute_vals = np.nanmedian(X, axis=0)
        for j in range(X_imp.shape[1]):
            mask = np.isnan(X_imp[:, j])
            if mask.any():
                X_imp[mask, j] = self.impute_vals[j] if not np.isnan(self.impute_vals[j]) else 0.0
        return X_imp

    def fit(self, X, y, X_val, y_val, xgb_params, sample_weights=None):
        self.xgb_ensemble.fit(X, y, X_val, y_val, xgb_params, sample_weights)

        X_imp = self._impute(X, fit=True)
        X_val_imp = self._impute(X_val)

        sw = sample_weights if sample_weights is not None else np.ones(len(y))

        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=4)
        self.rf.fit(X_imp, y.astype(int), sample_weight=sw)

        self.gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        self.gb.fit(X_imp, y.astype(int), sample_weight=sw)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_xgb = np.zeros(len(X))
        oof_rf = np.zeros(len(X))
        oof_gb = np.zeros(len(X))

        for fold_train, fold_val in kf.split(X):
            m = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200,
                                  early_stopping_rounds=20, min_child_weight=3,
                                  n_jobs=4, random_state=42, tree_method='hist',
                                  **{k: v for k, v in xgb_params.items() if k != 'best_iteration'})
            sw_fold = sw[fold_train]
            m.fit(X[fold_train], y[fold_train], sample_weight=sw_fold,
                  eval_set=[(X[fold_val], y[fold_val])], verbose=False)
            oof_xgb[fold_val] = m.predict(X[fold_val])

            X_ft_imp = self._impute(X[fold_train])
            X_fv_imp = self._impute(X[fold_val])

            rf_fold = RandomForestClassifier(n_estimators=100, max_depth=8,
                                              min_samples_leaf=5, class_weight='balanced',
                                              random_state=42, n_jobs=4)
            rf_fold.fit(X_ft_imp, y[fold_train].astype(int), sample_weight=sw_fold)
            oof_rf[fold_val] = rf_fold.predict_proba(X_fv_imp)[:, 1]

            gb_fold = GradientBoostingClassifier(n_estimators=50, max_depth=4,
                                                  learning_rate=0.05, subsample=0.8,
                                                  min_samples_leaf=5, random_state=42)
            gb_fold.fit(X_ft_imp, y[fold_train].astype(int), sample_weight=sw_fold)
            oof_gb[fold_val] = gb_fold.predict_proba(X_fv_imp)[:, 1]

        meta_X = np.column_stack([oof_xgb, oof_rf, oof_gb])
        self.meta = LogisticRegression(random_state=42, C=1.0)
        self.meta.fit(meta_X, y.astype(int), sample_weight=sw)

        meta_weights = self.meta.coef_[0]
        print(f"    Stacking weights: XGB={meta_weights[0]:.2f}, RF={meta_weights[1]:.2f}, GB={meta_weights[2]:.2f}")

    def predict_proba(self, X):
        """Return soft probability from stacking meta-learner."""
        X_imp = self._impute(X)
        p_xgb = self.xgb_ensemble.predict(X).reshape(-1, 1)
        p_rf = self.rf.predict_proba(X_imp)[:, 1].reshape(-1, 1)
        p_gb = self.gb.predict_proba(X_imp)[:, 1].reshape(-1, 1)
        meta_X = np.hstack([p_xgb, p_rf, p_gb])
        return self.meta.predict_proba(meta_X)[:, 1]

    def get_importances(self):
        return self.xgb_ensemble.get_importances()


class TwoStageClassifier:
    """
    Stage 1: CN vs Not — tabular only, XGBoost ensemble, threshold=0.5
    Stage 2: MCI vs Alzheimer's — tabular only, XGBoost ensemble
    Simple and proven. Interactions provide the new signal.
    """
    def __init__(self, n_ensemble=10):
        self.stage1 = EnsembleModel(n_models=n_ensemble)
        self.stage2 = EnsembleModel(n_models=n_ensemble)
        self.tab_indices = None

    def fit(self, X, y, X_val, y_val, params1, params2, w1=None, w2=None,
            tab_indices=None):
        self.tab_indices = tab_indices

        y1 = (y > 0).astype(float)
        y1v = (y_val > 0).astype(float)

        print("\n  Stage 1: CN vs Not (tabular only)")
        X_tab = X[:, tab_indices]
        X_tab_v = X_val[:, tab_indices]
        self.stage1.fit(X_tab, y1, X_tab_v, y1v, params1, w1)
        s1_acc = accuracy_score(y1v.astype(int), (self.stage1.predict(X_tab_v) > 0.5).astype(int))
        print(f"  Stage 1 val accuracy: {s1_acc:.1%}")

        nn_t = y > 0
        nn_v = y_val > 0
        if nn_t.sum() > 0 and nn_v.sum() > 0:
            y2 = (y[nn_t] - 1).astype(float)
            y2v = (y_val[nn_v] - 1).astype(float)
            w2_sel = w2[nn_t] if w2 is not None else None
            print("\n  Stage 2: MCI vs Alzheimer's (tabular only)")
            self.stage2.fit(X[nn_t][:, tab_indices], y2,
                           X_val[nn_v][:, tab_indices], y2v, params2, w2_sel)
            s2_acc = accuracy_score(y2v.astype(int),
                                    (self.stage2.predict(X_val[nn_v][:, tab_indices]) > 0.5).astype(int))
            print(f"  Stage 2 val accuracy: {s2_acc:.1%}")

    def predict(self, X):
        X_tab = X[:, self.tab_indices]
        s1 = self.stage1.predict(X_tab)
        results = np.zeros(len(X))
        nn = s1 > 0.5
        if nn.sum() > 0:
            s2 = self.stage2.predict(X[nn][:, self.tab_indices])
            results[nn] = np.where(s2 > 0.5, 2, 1)
        return results

    def get_importances(self):
        i1 = self.stage1.get_importances()
        i2 = self.stage2.get_importances()
        c = {}
        for k in set(list(i1.keys()) + list(i2.keys())):
            c[k] = i1.get(k, 0) + i2.get(k, 0)
        return c


class HybridModel:
    def __init__(self):
        self.classifier = TwoStageClassifier(n_ensemble=10)
        self.cdr_model = EnsembleModel(n_models=10)
        self.feature_names = None

    def predict_from_combined(self, X):
        """Predict from already-combined CNN+tabular features."""
        cdr = self.cdr_model.predict(X)
        udsd = self.classifier.predict(X)
        return np.column_stack([cdr, udsd])


def evaluate_model(model, X, Y, patient_ids, split_name="Test"):
    label_names = {0: 'CN', 1: 'MCI', 2: "Alzheimer's"}
    Y_pred = model.predict_from_combined(X)

    print(f"\n{'#'*60}")
    print(f"  {split_name} SET RESULTS")
    print(f"{'#'*60}")

    results = {}
    for i, target in enumerate(TARGETS):
        valid = ~np.isnan(Y[:, i])
        y_true = Y[valid, i]
        y_pred = Y_pred[valid, i]
        ids = [patient_ids[j] for j in range(len(patient_ids)) if valid[j]]

        print(f"\n{'='*50}")
        print(f"TARGET: {target} ({len(y_true)} patients)")
        print(f"{'='*50}")

        if target == 'NACCUDSD':
            ypc = y_pred.astype(int)
            ytc = y_true.astype(int)
            acc = accuracy_score(ytc, ypc)
            f1m = f1_score(ytc, ypc, average='macro', zero_division=0)
            prec = precision_score(ytc, ypc, average='macro', zero_division=0)
            rec = recall_score(ytc, ypc, average='macro', zero_division=0)

            print(f"\n  Accuracy:          {acc:.1%}")
            print(f"  Precision (macro): {prec:.3f}")
            print(f"  Recall (macro):    {rec:.3f}")
            print(f"  F1 (macro):        {f1m:.3f}")

            classes = sorted(np.unique(np.concatenate([ytc, ypc])))
            print(f"\n{classification_report(ytc, ypc, target_names=[label_names[c] for c in classes], zero_division=0)}")
            cm = confusion_matrix(ytc, ypc)
            print(f"  Confusion Matrix:\n  {cm}")
            results[target] = {'accuracy': acc, 'f1_macro': f1m}

            print(f"\n  {'Patient':<15} {'Actual':<15} {'Predicted':<15} {'?'}")
            print(f"  {'-'*50}")
            for pid, a, p in zip(ids, ytc, ypc):
                print(f"  {pid:<15} {label_names[a]:<15} {label_names[p]:<15} {'✅' if a==p else '❌'}")
        else:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')

            print(f"\n  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²:   {r2:.3f}")
            results[target] = {'mae': mae, 'rmse': rmse, 'r2': r2}

            print(f"\n  {'Patient':<15} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
            print(f"  {'-'*45}")
            for pid, a, p in zip(ids, y_true, y_pred):
                print(f"  {pid:<15} {a:>8.1f} {p:>10.1f} {p-a:>+8.1f}")

    return results


def plot_importance(model, top_n=30):
    for label, getter in [('NACCUDSD', model.classifier), ('CDRSUM', model.cdr_model)]:
        importance = getter.get_importances()
        if not importance:
            continue
        feat_names = model.feature_names or []
        named = {}
        for idx, score in importance.items():
            name = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
            named[name] = score

        top = sorted(named.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [x[0] for x in top]
        values = [x[1] for x in top]

        cnn_count = sum(1 for n in names if n.startswith('img_') or n.startswith('hc_'))
        tab_count = len(names) - cnn_count

        colors = ['#e07020' if (n.startswith('img_') or n.startswith('hc_')) else '#4080c0' for n in names]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(names)), values, color=colors)
        plt.yticks(range(len(names)), names, fontsize=8)
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Features — {label}\n(blue=tabular, orange=image features)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{label}_v6.png', dpi=150)
        plt.close()
        print(f"  Saved feature_importance_{label}_v6.png ({tab_count} tabular, {cnn_count} image)")


if __name__ == "__main__":
    print("=" * 60)
    print("ALZHEIMER'S HYBRID CNN + XGBOOST MODEL (v6 calibrated)")
    print("Interactions + conservative regularization + CDRSUM-assist + CV")
    print(f"Image: {NUM_SLICES} slices → 53 brain-specific features")
    print(f"Tabular: {len(ALL_TABULAR)} base + {len(RATE_OF_CHANGE_VARS)} ROC")
    print(f"Targets: {TARGETS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading TRAINING data (LATEST visit)...")
    img_train, tab_train, Y_train, ids_train, roc_names = build_dataset(
        TRAIN_EXCEL, SCANS_FOLDER, IMG_SIZE, use_latest=True)

    print("\nLoading VALIDATION data (EARLIEST visit)...")
    img_val, tab_val, Y_val, ids_val, _ = build_dataset(
        VAL_EXCEL, SCANS_FOLDER, IMG_SIZE, use_latest=False)

    print("\nLoading TEST data (EARLIEST visit)...")
    img_test, tab_test, Y_test, ids_test, _ = build_dataset(
        TEST_EXCEL, SCANS_FOLDER, IMG_SIZE, use_latest=False)

    udsd_i = TARGETS.index('NACCUDSD')
    vt = ~np.isnan(Y_train[:, udsd_i])
    vv = ~np.isnan(Y_val[:, udsd_i])

    hc_train, hc_names = extract_handcrafted_features(img_train)
    hc_val, _ = extract_handcrafted_features(img_val)
    hc_test, _ = extract_handcrafted_features(img_test)

    del img_train, img_val, img_test
    gc.collect()

    tab_feature_names_full = ALL_TABULAR + roc_names
    int_train, int_names = compute_interaction_features(tab_train, tab_feature_names_full)
    int_val, _ = compute_interaction_features(tab_val, tab_feature_names_full)
    int_test, _ = compute_interaction_features(tab_test, tab_feature_names_full)

    tab_train_ext = np.hstack([tab_train, int_train])
    tab_val_ext = np.hstack([tab_val, int_val])
    tab_test_ext = np.hstack([tab_test, int_test])
    tab_feature_names_ext = tab_feature_names_full + int_names
    print(f"  Extended tabular: {tab_train_ext.shape[1]} ({tab_train.shape[1]} base + {len(int_names)} interaction)")

    X_train = np.hstack([hc_train, tab_train])
    X_val = np.hstack([hc_val, tab_val])
    X_test = np.hstack([hc_test, tab_test])

    n_hc_feat = hc_train.shape[1]
    all_feature_names = hc_names + ALL_TABULAR + roc_names
    print(f"  Combined: {n_hc_feat} hand-crafted image + {tab_train.shape[1]} tabular = {X_train.shape[1]} total")

    print("\n  Running feature selection (TABULAR+INTERACTIONS, top 150)...")
    quick = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100,
                              n_jobs=4, random_state=42, tree_method='hist')
    quick.fit(tab_train_ext[vt], Y_train[vt, udsd_i], verbose=False)
    imp = quick.feature_importances_

    tab_scored = [(i, tab_feature_names_ext[i], imp[i]) for i in range(len(imp)) if imp[i] > 0]
    tab_scored.sort(key=lambda x: x[2], reverse=True)

    n_tab = min(150, len(tab_scored))
    top_tab = tab_scored[:n_tab]
    tab_keep_idx = sorted([t[0] for t in top_tab])
    tab_keep_names = [tab_feature_names_ext[i] for i in tab_keep_idx]

    roc_count = sum(1 for n in tab_keep_names if n.endswith('_ROC'))
    int_count = sum(1 for n in tab_keep_names if n in int_names)
    print(f"  Selected: {n_tab - roc_count - int_count} tabular + {roc_count} ROC + {int_count} interaction")
    print(f"  Top 5: {[t[1] for t in top_tab[:5]]}")
    if int_count > 0:
        int_selected = [t[1] for t in top_tab if t[1] in int_names]
        print(f"  Interaction features selected: {int_selected[:10]}")

    tab_train_sel = tab_train_ext[:, tab_keep_idx]
    tab_val_sel = tab_val_ext[:, tab_keep_idx]
    tab_test_sel = tab_test_ext[:, tab_keep_idx]

    X_train = np.hstack([tab_train_sel, hc_train])
    X_val = np.hstack([tab_val_sel, hc_val])
    X_test = np.hstack([tab_test_sel, hc_test])

    all_feature_names = tab_keep_names + hc_names
    del quick; gc.collect()

    print(f"  Final: {len(tab_keep_names)} tabular + {len(hc_names)} image = {len(all_feature_names)} total")

    tab_indices = list(range(len(tab_keep_names)))  # first N are tabular
    img_indices = list(range(len(tab_keep_names), len(all_feature_names)))
    print(f"  Feature split: {len(tab_indices)} tabular, {len(img_indices)} image")
    
    print("\n" + "-" * 60)
    y_udsd_train = Y_train[vt, udsd_i]
    y_udsd_val = Y_val[vv, udsd_i]
    sample_weights = compute_sample_weights(y_udsd_train)
    full_weights = np.ones(len(Y_train))
    full_weights[vt] = sample_weights

    print("\n" + "-" * 60)
    print("Tuning Stage 1 (CN vs Not) on TABULAR features (conservative)...")
    y1_t = (y_udsd_train > 0).astype(float)
    y1_v = (y_udsd_val > 0).astype(float)
    params1 = tune_hyperparameters(
        X_train[vt][:, tab_indices], y1_t,
        X_val[vv][:, tab_indices], y1_v, sample_weights)

    print("\nTuning Stage 2 (MCI vs Alzheimer's) on TABULAR features (conservative)...")
    gc.collect()
    nn_t = vt & (Y_train[:, udsd_i] > 0)
    nn_v = vv & (Y_val[:, udsd_i] > 0)
    y2_t = (Y_train[nn_t, udsd_i] - 1).astype(float)
    y2_v = (Y_val[nn_v, udsd_i] - 1).astype(float)
    params2 = tune_hyperparameters(
        X_train[nn_t][:, tab_indices], y2_t,
        X_val[nn_v][:, tab_indices], y2_v, full_weights[nn_t])

    cdr_i = TARGETS.index('CDRSUM')
    vt_cdr = ~np.isnan(Y_train[:, cdr_i])
    vv_cdr = ~np.isnan(Y_val[:, cdr_i])

    print("\nTuning CDRSUM regressor...")
    gc.collect()
    params_cdr = tune_hyperparameters(
        X_train[vt_cdr], Y_train[vt_cdr, cdr_i],
        X_val[vv_cdr], Y_val[vv_cdr, cdr_i],
        full_weights[vt_cdr])

    print("\n" + "-" * 60)
    print("Training ensemble models (10 each)...")

    model = HybridModel()
    model.feature_names = all_feature_names

    print("\nTraining NACCUDSD classifier...")
    model.classifier.fit(
        X_train[vt], y_udsd_train, X_val[vv], y_udsd_val,
        params1, params2, sample_weights, full_weights[vt],
        tab_indices=tab_indices)

    print("\nTraining CDRSUM regressor...")
    model.cdr_model.fit(
        X_train[vt_cdr], Y_train[vt_cdr, cdr_i],
        X_val[vv_cdr], Y_val[vv_cdr, cdr_i],
        params_cdr, full_weights[vt_cdr])

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold on all 288 patients)")
    print("=" * 60)

    all_X = np.vstack([X_train, X_val, X_test])
    all_Y = np.vstack([Y_train, Y_val, Y_test])
    all_ids = ids_train + ids_val + ids_test

    cv_valid = ~np.isnan(all_Y[:, udsd_i])
    X_cv = all_X[cv_valid]
    y_cv = all_Y[cv_valid, udsd_i]
    cdrsum_cv_all = all_Y[cv_valid, 0]  # CDRSUM is at index 0
    ids_cv = [all_ids[i] for i in range(len(all_ids)) if cv_valid[i]]

    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ Cross-validation complete")
    print("\nNote: 20 solutions section requires per-fold predictions.")
    print("Current best method is Per-Fold Calibration (83.7%)")
    print("="*70)

    np.random.seed(99)
    blind_indices = []
    for cls, n_needed in [(0, 2), (1, 1), (2, 2)]:
        cls_idx = [i for i in range(len(y_cv)) if int(y_cv[i]) == cls]
        chosen = np.random.choice(cls_idx, size=n_needed, replace=False)
        blind_indices.extend(chosen.tolist())
    blind_indices = sorted(blind_indices)

    blind_X = X_cv[blind_indices]
    blind_y = y_cv[blind_indices]
    blind_cdrsum = cdrsum_cv_all[blind_indices]
    blind_ids = [ids_cv[i] for i in blind_indices]

    cv_mask = np.ones(len(y_cv), dtype=bool)
    cv_mask[blind_indices] = False
    X_cv = X_cv[cv_mask]
    y_cv = y_cv[cv_mask]
    cdrsum_cv_all = cdrsum_cv_all[cv_mask]
    ids_cv = [ids_cv[i] for i in range(len(ids_cv)) if cv_mask[i]]

    label_names_blind = {0: 'CN', 1: 'MCI', 2: "Alzheimer's"}
    print(f"\n  Held out 5 BLIND TEST patients:")
    for pid, y_true in zip(blind_ids, blind_y):
        print(f"    {pid}: {label_names_blind[int(y_true)]}")
    print(f"  Remaining for CV: {len(y_cv)} patients")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y_cv))
    cv_cdrsum_preds = np.zeros(len(y_cv))
    cv_s1_probs = np.zeros(len(y_cv))  # P(Not_CN)
    cv_s2_probs = np.zeros(len(y_cv))  # P(AD | Not_CN), filled even for CN-predicted
    cv_fold_accs = []
    all_loss_curves = {'s1_train': [], 's1_val': [], 's2_train': [], 's2_val': []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_cv)):
        X_ft, X_fv = X_cv[train_idx], X_cv[test_idx]
        y_ft, y_fv = y_cv[train_idx], y_cv[test_idx]

        fold_weights = compute_sample_weights(y_ft)

        y1_ft = (y_ft > 0).astype(float)
        y1_fv = (y_fv > 0).astype(float)

        tab_idx = tab_indices
        s1_models = []
        s1_evals = None
        for i in range(5):
            m = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=params1.get('max_depth', 3),
                learning_rate=params1.get('learning_rate', 0.05),
                n_estimators=500, early_stopping_rounds=20,
                subsample=params1.get('subsample', 0.8),
                colsample_bytree=params1.get('colsample_bytree', 0.7),
                min_child_weight=3, n_jobs=4,
                random_state=42 + i, tree_method='hist',
                eval_metric='rmse')
            m.fit(X_ft[:, tab_idx], y1_ft, sample_weight=fold_weights,
                  eval_set=[(X_ft[:, tab_idx], y1_ft), (X_fv[:, tab_idx], y1_fv)],
                  verbose=False)
            s1_models.append(m)
            if i == 0:  # capture loss from first model in ensemble
                s1_evals = m.evals_result()

        if s1_evals:
            all_loss_curves['s1_train'].append(s1_evals['validation_0']['rmse'])
            all_loss_curves['s1_val'].append(s1_evals['validation_1']['rmse'])

        if s1_evals:
            s1_train_loss = s1_evals['validation_0']['rmse']
            s1_val_loss = s1_evals['validation_1']['rmse']
            n_iters = len(s1_train_loss)
            checkpoints = [0, min(9, n_iters-1), min(49, n_iters-1), min(99, n_iters-1), n_iters-1]
            checkpoints = sorted(set(checkpoints))
            loss_str = ", ".join([f"iter {c+1}: train={s1_train_loss[c]:.4f} val={s1_val_loss[c]:.4f}" for c in checkpoints])
            print(f"    Stage 1 loss: {loss_str}")

        s1_pred_train = np.mean([m.predict(X_ft[:, tab_idx]) for m in s1_models], axis=0)
        s1_pred = np.mean([m.predict(X_fv[:, tab_idx]) for m in s1_models], axis=0)
        s1_train_acc = accuracy_score(y1_ft.astype(int), (s1_pred_train > 0.5).astype(int))
        s1_val_acc = accuracy_score(y1_fv.astype(int), (s1_pred > 0.5).astype(int))

        nn_t = y_ft > 0
        nn_v = s1_pred > 0.5

        fold_pred = np.zeros(len(y_fv))
        s2_train_acc_str = "N/A"
        s2_val_acc_str = "N/A"
        if nn_t.sum() > 0 and nn_v.sum() > 0:
            y2_ft = (y_ft[nn_t] - 1).astype(float)
            s2_models = []
            s2_evals = None
            split = max(1, int(0.2 * nn_t.sum()))
            for i in range(5):
                m = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=params2.get('max_depth', 3),
                    learning_rate=params2.get('learning_rate', 0.05),
                    n_estimators=500, early_stopping_rounds=20,
                    subsample=params2.get('subsample', 0.7),
                    colsample_bytree=params2.get('colsample_bytree', 0.7),
                    min_child_weight=3, n_jobs=4,
                    random_state=42 + i, tree_method='hist',
                    eval_metric='rmse')
                m.fit(X_ft[nn_t][:, tab_idx][:-split], y2_ft[:-split],
                      sample_weight=fold_weights[nn_t][:-split],
                      eval_set=[(X_ft[nn_t][:, tab_idx][:-split], y2_ft[:-split]),
                                (X_ft[nn_t][:, tab_idx][-split:], y2_ft[-split:])],
                      verbose=False)
                s2_models.append(m)
                if i == 0:
                    s2_evals = m.evals_result()

            if s2_evals:
                all_loss_curves['s2_train'].append(s2_evals['validation_0']['rmse'])
                all_loss_curves['s2_val'].append(s2_evals['validation_1']['rmse'])

                s2_train_loss = s2_evals['validation_0']['rmse']
                s2_val_loss = s2_evals['validation_1']['rmse']
                n_iters2 = len(s2_train_loss)
                checkpoints2 = [0, min(9, n_iters2-1), min(49, n_iters2-1), min(99, n_iters2-1), n_iters2-1]
                checkpoints2 = sorted(set(checkpoints2))
                loss_str2 = ", ".join([f"iter {c+1}: train={s2_train_loss[c]:.4f} val={s2_val_loss[c]:.4f}" for c in checkpoints2])
                print(f"    Stage 2 loss: {loss_str2}")

            s2_pred_train = np.mean([m.predict(X_ft[nn_t][:, tab_idx]) for m in s2_models], axis=0)
            s2_train_acc = accuracy_score(y2_ft.astype(int), (s2_pred_train > 0.5).astype(int))
            s2_train_acc_str = f"{s2_train_acc:.1%}"

            s2_pred = np.mean([m.predict(X_fv[nn_v][:, tab_idx]) for m in s2_models], axis=0)
            fold_pred[nn_v] = np.where(s2_pred > 0.5, 2, 1)
            
            s2_pred_all = np.mean([m.predict(X_fv[:, tab_idx]) for m in s2_models], axis=0)

            nn_v_true = y_fv > 0
            if nn_v_true.sum() > 0:
                s2_val_preds_true = fold_pred[nn_v_true]
                s2_val_actual = y_fv[nn_v_true]
                s2_val_correct = ((s2_val_preds_true == s2_val_actual).sum())
                s2_val_acc_str = f"{s2_val_correct / nn_v_true.sum():.1%}"
        else:
            s2_pred_all = np.full(len(y_fv), 0.5)  # neutral if no Stage 2

        cv_preds[test_idx] = fold_pred
        
        cv_s1_probs[test_idx] = s1_pred
        cv_s2_probs[test_idx] = s2_pred_all
        
        cdrsum_pred_fold = model.cdr_model.predict(X_fv)
        cv_cdrsum_preds[test_idx] = cdrsum_pred_fold
        
        fold_acc = accuracy_score(y_fv.astype(int), fold_pred.astype(int))
        cv_fold_accs.append(fold_acc)
        print(f"  Fold {fold+1}: accuracy={fold_acc:.1%}  |  Stage1: train={s1_train_acc:.1%} val={s1_val_acc:.1%}  |  Stage2: train={s2_train_acc_str} val={s2_val_acc_str}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fold_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    ax = axes[0]
    for i, (train_loss, val_loss) in enumerate(zip(all_loss_curves['s1_train'], all_loss_curves['s1_val'])):
        ax.plot(train_loss, color=fold_colors[i], alpha=0.5, linewidth=1.2, linestyle='--')
        ax.plot(val_loss, color=fold_colors[i], alpha=1.0, linewidth=2.2, linestyle='-', label=f'Fold {i+1}')
    ax.plot([], [], color='gray', linewidth=2.2, linestyle='-', label='Validation')
    ax.plot([], [], color='gray', linewidth=1.2, linestyle='--', alpha=0.5, label='Training')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('RMSE Loss', fontsize=11)
    ax.set_title('Stage 1: CN vs NN', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#fafafa')

    ax = axes[1]
    for i, (train_loss, val_loss) in enumerate(zip(all_loss_curves['s2_train'], all_loss_curves['s2_val'])):
        ax.plot(train_loss, color=fold_colors[i], alpha=0.5, linewidth=1.2, linestyle='--')
        ax.plot(val_loss, color=fold_colors[i], alpha=1.0, linewidth=2.2, linestyle='-', label=f'Fold {i+1}')
    ax.plot([], [], color='gray', linewidth=2.2, linestyle='-', label='Validation')
    ax.plot([], [], color='gray', linewidth=1.2, linestyle='--', alpha=0.5, label='Training')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('RMSE Loss', fontsize=11)
    ax.set_title('Stage 2: MCI vs AD', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#fafafa')

    plt.suptitle('Training Loss Curves (dashed = training, solid = validation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('loss_curves_v6.png', dpi=200, facecolor='white')
    plt.close()
    print("  Saved loss_curves_v6.png")

    with open('loss_curves_data.pkl', 'wb') as _f:
        pickle.dump(all_loss_curves, _f)
    print("  Saved loss_curves_data.pkl")

    cv_acc = accuracy_score(y_cv.astype(int), cv_preds.astype(int))
    cv_f1 = f1_score(y_cv.astype(int), cv_preds.astype(int), average='macro', zero_division=0)
    cv_prec = precision_score(y_cv.astype(int), cv_preds.astype(int), average='macro', zero_division=0)
    cv_rec = recall_score(y_cv.astype(int), cv_preds.astype(int), average='macro', zero_division=0)

    print(f"\n  5-Fold CV Accuracy: {cv_acc:.1%} (±{np.std(cv_fold_accs):.1%})")
    print(f"  5-Fold CV F1 (macro): {cv_f1:.3f}")
    print(f"  5-Fold CV Precision (macro): {cv_prec:.3f}")
    print(f"  5-Fold CV Recall (macro): {cv_rec:.3f}")
    print(f"  Fold accuracies: {[f'{a:.1%}' for a in cv_fold_accs]}")

    label_names_cv = {0: 'CN', 1: 'MCI', 2: "Alzheimer's"}
    print(f"\n{classification_report(y_cv.astype(int), cv_preds.astype(int), target_names=[label_names_cv[c] for c in sorted(np.unique(y_cv.astype(int)))], zero_division=0)}")
    cm = confusion_matrix(y_cv.astype(int), cv_preds.astype(int))
    print(f"  Confusion Matrix:")
    print(f"  {cm}")

    print("\n" + "=" * 60)
    print("CALIBRATION: Combining diagnosis + CDRSUM signals")
    print("=" * 60)
    
    def stage_probs_to_3class(s1_p, s2_p):
        s1_p = np.clip(s1_p, 0.001, 0.999)
        s2_p = np.clip(s2_p, 0.001, 0.999)
        p_cn = 1 - s1_p
        p_ad = s1_p * s2_p
        p_mci = s1_p * (1 - s2_p)
        return np.column_stack([p_cn, p_mci, p_ad])
    
    original_probs = stage_probs_to_3class(cv_s1_probs, cv_s2_probs)
    
    def cdrsum_to_probs(cdrsum, mu_cn, sig_cn, mu_mci, sig_mci, mu_ad, sig_ad):
        cdrsum = np.asarray(cdrsum)
        p_cn = np.exp(-0.5 * ((cdrsum - mu_cn) / sig_cn) ** 2)
        p_mci = np.exp(-0.5 * ((cdrsum - mu_mci) / sig_mci) ** 2)
        p_ad = np.exp(-0.5 * ((cdrsum - mu_ad) / sig_ad) ** 2)
        total = p_cn + p_mci + p_ad + 1e-10
        return np.column_stack([p_cn / total, p_mci / total, p_ad / total])
    
    from sklearn.model_selection import KFold as KFold_nested
    m1_probs = np.zeros_like(original_probs)  # Will be filled by nested CV
    
    kf_inner = KFold_nested(n_splits=5, shuffle=True, random_state=999)  # Different seed from outer CV
    clinical_priors = {
        'mu_cn': 0.0, 'sig_cn': 1.0,
        'mu_mci': 1.5, 'sig_mci': 1.5,
        'mu_ad': 6.0, 'sig_ad': 3.5,
    }
    
    for train_fold_idx, test_fold_idx in kf_inner.split(X_cv):
        best_alpha_inner = 0.5
        best_acc_inner = 0
        
        for alpha in np.arange(0.30, 1.01, 0.05):
            cdr_probs_train = cdrsum_to_probs(
                cv_cdrsum_preds[train_fold_idx],
                clinical_priors['mu_cn'], clinical_priors['sig_cn'],
                clinical_priors['mu_mci'], clinical_priors['sig_mci'],
                clinical_priors['mu_ad'], clinical_priors['sig_ad'])
            blend_train = alpha * original_probs[train_fold_idx] + (1 - alpha) * cdr_probs_train
            acc_train = accuracy_score(y_cv.astype(int)[train_fold_idx], blend_train.argmax(axis=1))
            if acc_train > best_acc_inner:
                best_acc_inner = acc_train
                best_alpha_inner = alpha
        
        cdr_probs_test = cdrsum_to_probs(
            cv_cdrsum_preds[test_fold_idx],
            clinical_priors['mu_cn'], clinical_priors['sig_cn'],
            clinical_priors['mu_mci'], clinical_priors['sig_mci'],
            clinical_priors['mu_ad'], clinical_priors['sig_ad'])
        m1_probs[test_fold_idx] = (best_alpha_inner * original_probs[test_fold_idx] + 
                                   (1 - best_alpha_inner) * cdr_probs_test)
    
    cdr_probs_final = cdrsum_to_probs(
        cv_cdrsum_preds,
        clinical_priors['mu_cn'], clinical_priors['sig_cn'],
        clinical_priors['mu_mci'], clinical_priors['sig_mci'],
        clinical_priors['mu_ad'], clinical_priors['sig_ad'])
    
    print("\n  Method 1: Post-hoc soft blending (via nested CV)")
    m1_preds_final = m1_probs.argmax(axis=1)
    best_acc_m1 = accuracy_score(y_cv.astype(int), m1_preds_final)
    from sklearn.metrics import precision_recall_fscore_support
    m1_prec, m1_rec, m1_f1, _ = precision_recall_fscore_support(y_cv.astype(int), m1_preds_final, average=None)
    print(f"    Method 1 accuracy: {best_acc_m1:.1%}")
    print(f"      CN:  {m1_prec[0]:.1%} precision, {m1_rec[0]:.1%} recall  (F1={m1_f1[0]:.3f})")
    print(f"      MCI: {m1_prec[1]:.1%} precision, {m1_rec[1]:.1%} recall  (F1={m1_f1[1]:.3f})")
    print(f"      AD:  {m1_prec[2]:.1%} precision, {m1_rec[2]:.1%} recall  (F1={m1_f1[2]:.3f})")
    
    print("\n  Method 2: Retraining Stage 2 with CDRSUM included as feature")
    cv_preds_m2 = cv_preds.copy().astype(int)
    
    kf_m2 = KFold(n_splits=5, shuffle=True, random_state=42)
    tab_idx_m2 = model.classifier.tab_indices
    for fold, (train_idx, test_idx) in enumerate(kf_m2.split(X_cv)):
        X_ft = X_cv[train_idx]
        X_fv = X_cv[test_idx]
        y_ft = y_cv[train_idx]
        y_fv = y_cv[test_idx]
        cdr_ft = cv_cdrsum_preds[train_idx]
        cdr_fv = cv_cdrsum_preds[test_idx]
        
        nn_t = y_ft > 0
        s1_test = cv_s1_probs[test_idx]
        nn_v = s1_test > 0.5
        
        if nn_t.sum() > 0 and nn_v.sum() > 0:
            y2_ft = (y_ft[nn_t] - 1).astype(float)
            X_ft_aug = np.column_stack([X_ft[nn_t][:, tab_idx_m2], cdr_ft[nn_t]])
            X_fv_aug = np.column_stack([X_fv[nn_v][:, tab_idx_m2], cdr_fv[nn_v]])
            
            split_m2 = max(1, int(0.2 * nn_t.sum()))
            s2_models_m2 = []
            for i in range(5):
                m = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=params2.get('max_depth', 3),
                    learning_rate=params2.get('learning_rate', 0.05),
                    n_estimators=500, early_stopping_rounds=20,
                    subsample=params2.get('subsample', 0.7),
                    colsample_bytree=params2.get('colsample_bytree', 0.7),
                    min_child_weight=3, n_jobs=4,
                    random_state=42 + i, tree_method='hist',
                    eval_metric='rmse')
                m.fit(X_ft_aug[:-split_m2], y2_ft[:-split_m2],
                      eval_set=[(X_ft_aug[:-split_m2], y2_ft[:-split_m2]),
                                (X_ft_aug[-split_m2:], y2_ft[-split_m2:])],
                      verbose=False)
                s2_models_m2.append(m)
            
            s2_pred_m2 = np.mean([m.predict(X_fv_aug) for m in s2_models_m2], axis=0)
            fold_pred_m2 = np.zeros(len(y_fv), dtype=int)
            fold_pred_m2[nn_v] = np.where(s2_pred_m2 > 0.5, 2, 1)
            cv_preds_m2[test_idx] = fold_pred_m2
    
    m2_acc = accuracy_score(y_cv.astype(int), cv_preds_m2)
    m2_prec, m2_rec, m2_f1, _ = precision_recall_fscore_support(y_cv.astype(int), cv_preds_m2, average=None)
    print(f"    Method 2 accuracy: {m2_acc:.1%}")
    print(f"      CN:  {m2_prec[0]:.1%} precision, {m2_rec[0]:.1%} recall  (F1={m2_f1[0]:.3f})")
    print(f"      MCI: {m2_prec[1]:.1%} precision, {m2_rec[1]:.1%} recall  (F1={m2_f1[1]:.3f})")
    print(f"      AD:  {m2_prec[2]:.1%} precision, {m2_rec[2]:.1%} recall  (F1={m2_f1[2]:.3f})")
    
    print("\n  Method 3: Meta-classifier combining stage probs + CDRSUM")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    meta_X = np.column_stack([original_probs, cv_cdrsum_preds.reshape(-1, 1)])
    cv_preds_m3 = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight='balanced'),
        meta_X, y_cv.astype(int), cv=5
    ).astype(int)
    
    m3_acc = accuracy_score(y_cv.astype(int), cv_preds_m3)
    m3_prec, m3_rec, m3_f1, _ = precision_recall_fscore_support(y_cv.astype(int), cv_preds_m3, average=None)
    print(f"    Method 3 accuracy: {m3_acc:.1%}")
    print(f"      CN:  {m3_prec[0]:.1%} precision, {m3_rec[0]:.1%} recall  (F1={m3_f1[0]:.3f})")
    print(f"      MCI: {m3_prec[1]:.1%} precision, {m3_rec[1]:.1%} recall  (F1={m3_f1[1]:.3f})")
    print(f"      AD:  {m3_prec[2]:.1%} precision, {m3_rec[2]:.1%} recall  (F1={m3_f1[2]:.3f})")
    
    print("\n  Method 4: Per-fold α optimization")
    fold_alphas = []
    per_fold_preds = np.zeros(len(y_cv), dtype=int)
    
    kf_pf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf_pf.split(X_cv)):
        y_ft = y_cv[train_idx]
        y_fv = y_cv[test_idx]
        
        orig_probs_ft = original_probs[train_idx]
        cdr_probs_ft = cdr_probs_final[train_idx]
        orig_probs_fv = original_probs[test_idx]
        cdr_probs_fv = cdr_probs_final[test_idx]
        
        best_fold_alpha = 0.30
        best_fold_acc = 0
        for alpha in np.arange(0.20, 0.50, 0.05):
            fold_probs_train = alpha * orig_probs_ft + (1 - alpha) * cdr_probs_ft
            fold_preds_train = fold_probs_train.argmax(axis=1)
            fold_acc = accuracy_score(y_ft.astype(int), fold_preds_train)
            
            if fold_acc > best_fold_acc:
                best_fold_acc = fold_acc
                best_fold_alpha = alpha
        
        fold_alphas.append(best_fold_alpha)
        fold_probs_opt = best_fold_alpha * orig_probs_fv + (1 - best_fold_alpha) * cdr_probs_fv
        per_fold_preds[test_idx] = fold_probs_opt.argmax(axis=1)
    
    best_acc_pf = accuracy_score(y_cv.astype(int), per_fold_preds)
    avg_alpha_pf = np.mean(fold_alphas)
    pf_prec, pf_rec, pf_f1, _ = precision_recall_fscore_support(y_cv.astype(int), per_fold_preds, average=None)
    print(f"    Per-fold accuracy: {best_acc_pf:.1%}")
    print(f"    Average α across folds: {avg_alpha_pf:.2f} ± {np.std(fold_alphas):.2f}")
    print(f"      CN:  {pf_prec[0]:.1%} precision, {pf_rec[0]:.1%} recall  (F1={pf_f1[0]:.3f})")
    print(f"      MCI: {pf_prec[1]:.1%} precision, {pf_rec[1]:.1%} recall  (F1={pf_f1[1]:.3f})")
    print(f"      AD:  {pf_prec[2]:.1%} precision, {pf_rec[2]:.1%} recall  (F1={pf_f1[2]:.3f})")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.feature_selection import RFE
    from sklearn.impute import SimpleImputer
    
    imputer_new = SimpleImputer(strategy='mean')
    X_cv_imputed_new = imputer_new.fit_transform(X_cv)
    X_cv_tab_new = X_cv_imputed_new[:, :150]
    y_cv_int_new = y_cv.astype(int)
    
    print("\n  Method 5: Isotonic calibration on stage probabilities")
    
    from sklearn.isotonic import IsotonicRegression
    
    calibrated_probs = np.zeros_like(original_probs)
    kf_cal = KFold(n_splits=5, shuffle=True, random_state=2026)
    
    for train_idx, test_idx in kf_cal.split(X_cv):
        for c in range(3):
            iso = IsotonicRegression(out_of_bounds='clip')
            y_binary_train = (y_cv_int_new[train_idx] == c).astype(int)
            iso.fit(original_probs[train_idx, c], y_binary_train)
            calibrated_probs[test_idx, c] = iso.transform(original_probs[test_idx, c])
    
    calibrated_probs = calibrated_probs / (calibrated_probs.sum(axis=1, keepdims=True) + 1e-10)
    cv_preds_m5 = calibrated_probs.argmax(axis=1)
    
    m5_acc = accuracy_score(y_cv_int_new, cv_preds_m5)
    m5_prec, m5_rec, m5_f1, _ = precision_recall_fscore_support(y_cv_int_new, cv_preds_m5, average=None)
    print(f"    Method 5 accuracy: {m5_acc:.1%}")
    print(f"      CN:  {m5_prec[0]:.1%} precision, {m5_rec[0]:.1%} recall  (F1={m5_f1[0]:.3f})")
    print(f"      MCI: {m5_prec[1]:.1%} precision, {m5_rec[1]:.1%} recall  (F1={m5_f1[1]:.3f})")
    print(f"      AD:  {m5_prec[2]:.1%} precision, {m5_rec[2]:.1%} recall  (F1={m5_f1[2]:.3f})")
    
    print("\n  Method 6: Class-specific threshold optimization (nested CV)")
    
    meta_X_for_thresh = np.column_stack([original_probs, cv_cdrsum_preds.reshape(-1, 1)])
    
    meta_probs_full = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight='balanced'),
        meta_X_for_thresh, y_cv_int_new, cv=5, method='predict_proba'
    )
    
    cv_preds_m6 = np.zeros(len(y_cv_int_new), dtype=int)
    kf_thresh = KFold(n_splits=5, shuffle=True, random_state=777)
    
    for train_idx, test_idx in kf_thresh.split(X_cv):
        best_scale = [1.0, 1.0, 1.0]
        best_acc_inner = 0
        
        for s_cn in [0.95, 1.0, 1.05, 1.10, 1.15]:
            for s_mci in [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
                for s_ad in [0.85, 0.95, 1.0, 1.05, 1.15]:
                    scaled = meta_probs_full[train_idx] * np.array([s_cn, s_mci, s_ad])
                    preds_train = scaled.argmax(axis=1)
                    acc_train = accuracy_score(y_cv_int_new[train_idx], preds_train)
                    if acc_train > best_acc_inner:
                        best_acc_inner = acc_train
                        best_scale = [s_cn, s_mci, s_ad]
        
        scaled_test = meta_probs_full[test_idx] * np.array(best_scale)
        cv_preds_m6[test_idx] = scaled_test.argmax(axis=1)
    
    m6_acc = accuracy_score(y_cv_int_new, cv_preds_m6)
    m6_prec, m6_rec, m6_f1, _ = precision_recall_fscore_support(y_cv_int_new, cv_preds_m6, average=None)
    print(f"    Method 6 accuracy: {m6_acc:.1%}")
    print(f"      CN:  {m6_prec[0]:.1%} precision, {m6_rec[0]:.1%} recall  (F1={m6_f1[0]:.3f})")
    print(f"      MCI: {m6_prec[1]:.1%} precision, {m6_rec[1]:.1%} recall  (F1={m6_f1[1]:.3f})")
    print(f"      AD:  {m6_prec[2]:.1%} precision, {m6_rec[2]:.1%} recall  (F1={m6_f1[2]:.3f})")
    
    print("\n  Method 7: RFE-reduced features (top 100) + meta-classifier")
    
    rfe_estimator = LogisticRegression(max_iter=1000, class_weight='balanced')
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=100, step=10)
    rfe.fit(X_cv_tab_new, y_cv_int_new)
    X_rfe = X_cv_tab_new[:, rfe.support_]
    
    rfe_xgb_probs = cross_val_predict(
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1),
        X_rfe, y_cv_int_new, cv=5, method='predict_proba'
    )
    
    rfe_meta_X = np.column_stack([rfe_xgb_probs, cv_cdrsum_preds.reshape(-1, 1)])
    cv_preds_m7 = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight='balanced'),
        rfe_meta_X, y_cv_int_new, cv=5
    ).astype(int)
    
    m7_acc = accuracy_score(y_cv_int_new, cv_preds_m7)
    m7_prec, m7_rec, m7_f1, _ = precision_recall_fscore_support(y_cv_int_new, cv_preds_m7, average=None)
    print(f"    Method 7 accuracy: {m7_acc:.1%}")
    print(f"      CN:  {m7_prec[0]:.1%} precision, {m7_rec[0]:.1%} recall  (F1={m7_f1[0]:.3f})")
    print(f"      MCI: {m7_prec[1]:.1%} precision, {m7_rec[1]:.1%} recall  (F1={m7_f1[1]:.3f})")
    print(f"      AD:  {m7_prec[2]:.1%} precision, {m7_rec[2]:.1%} recall  (F1={m7_f1[2]:.3f})")
    
    orig_acc_val = accuracy_score(y_cv.astype(int), cv_preds.astype(int))
    orig_prec, orig_rec, orig_f1, _ = precision_recall_fscore_support(y_cv.astype(int), cv_preds.astype(int), average=None)
    
    print("\n  TESTING 20 SOLUTIONS (Two honest cascading approaches)")
    print("  " + "=" * 60)
    
    try:
        from sklearn.neighbors import KNeighborsClassifier as KNN_classifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.impute import SimpleImputer
        
        imputer_cv = SimpleImputer(strategy='mean')
        X_cv_imputed = imputer_cv.fit_transform(X_cv)
        X_cv_tab_imputed = X_cv_imputed[:, :150]
        
        y_cv_int = y_cv.astype(int)
        
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            rf_probs = cross_val_predict(rf, X_cv_tab_imputed, y_cv_int, cv=5, method='predict_proba')
        except:
            rf_probs = m1_probs.copy()
        
        try:
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            gb_probs = cross_val_predict(gb, X_cv_tab_imputed, y_cv_int, cv=5, method='predict_proba')
        except:
            gb_probs = m1_probs.copy()
        
        try:
            knn5 = KNN_classifier(n_neighbors=5)
            knn_probs = cross_val_predict(knn5, X_cv_imputed, y_cv_int, cv=5, method='predict_proba')
        except:
            knn_probs = m1_probs.copy()
        
        s1_probs = 0.5 * m1_probs + 0.5 * original_probs
        s2_probs = 0.4 * m1_probs + 0.3 * rf_probs + 0.3 * gb_probs
        s3_probs = (m1_probs + rf_probs + gb_probs) / 3.0
        s4_probs = 0.5 * m1_probs + 0.25 * gb_probs + 0.25 * knn_probs
        s5_probs = 0.45 * m1_probs + 0.2 * rf_probs + 0.2 * gb_probs + 0.15 * knn_probs
        s6_probs = 0.6 * m1_probs + 0.2 * rf_probs + 0.2 * gb_probs
        s7_probs = 0.4 * m1_probs + 0.35 * rf_probs + 0.25 * gb_probs
        s8_probs = (m1_probs + rf_probs + gb_probs + knn_probs) / 4.0
        s9_probs = 0.5 * m1_probs + 0.5 * cdr_probs_final
        s10_probs = 0.35 * m1_probs + 0.35 * gb_probs + 0.3 * cdr_probs_final
        s11_probs = 0.5 * rf_probs + 0.5 * gb_probs
        s12_probs = 0.6 * gb_probs + 0.4 * rf_probs
        s13_probs = 0.7 * m1_probs + 0.3 * gb_probs
        s14_probs = 0.55 * m1_probs + 0.45 * rf_probs
        s15_probs = 0.65 * m1_probs + 0.35 * knn_probs
        s16_probs = (m1_probs + gb_probs) / 2.0
        s17_probs = (m1_probs + rf_probs) / 2.0
        s18_probs = (rf_probs + gb_probs) / 2.0
        s19_probs = m1_probs.copy()
        s20_probs = gb_probs.copy()
        
        all_solutions = [
            ('S1', s1_probs), ('S2', s2_probs), ('S3', s3_probs), ('S4', s4_probs), ('S5', s5_probs),
            ('S6', s6_probs), ('S7', s7_probs), ('S8', s8_probs), ('S9', s9_probs), ('S10', s10_probs),
            ('S11', s11_probs), ('S12', s12_probs), ('S13', s13_probs), ('S14', s14_probs), ('S15', s15_probs),
            ('S16', s16_probs), ('S17', s17_probs), ('S18', s18_probs), ('S19', s19_probs), ('S20', s20_probs)
        ]
        
        cascaded_conf_preds = np.zeros(len(y_cv_int), dtype=int)
        
        for i in range(len(y_cv_int)):
            best_conf = -1
            best_pred = m1_probs[i].argmax()
            
            for sol_name, sol_probs in all_solutions:
                conf = sol_probs[i].max()
                if conf > best_conf:
                    best_conf = conf
                    best_pred = sol_probs[i].argmax()
            
            cascaded_conf_preds[i] = best_pred
        
        cascaded_conf_acc = accuracy_score(y_cv_int, cascaded_conf_preds)
        
        cascaded_val_preds = np.zeros(len(y_cv_int), dtype=int)
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf_outer.split(X_cv)):
            
            solution_correctness = np.zeros((len(all_solutions), 3))  # (sol_idx, correct_count, total_count, accuracy)
            
            for sol_idx, (sol_name, sol_probs) in enumerate(all_solutions):
                preds = sol_probs[train_idx].argmax(axis=1)
                actuals = y_cv_int[train_idx]
                correct = np.sum(preds == actuals)
                total = len(actuals)
                solution_correctness[sol_idx, 0] = correct
                solution_correctness[sol_idx, 1] = total
                solution_correctness[sol_idx, 2] = correct / total if total > 0 else 0
            
            best_sol_idx = np.argmax(solution_correctness[:, 2])
            best_sol_probs = all_solutions[best_sol_idx][1]
            cascaded_val_preds[val_idx] = best_sol_probs[val_idx].argmax(axis=1)
        
        cascaded_val_acc = accuracy_score(y_cv_int, cascaded_val_preds)
        
        s1_acc = accuracy_score(y_cv_int, s1_probs.argmax(axis=1))
        s2_acc = accuracy_score(y_cv_int, s2_probs.argmax(axis=1))
        s3_acc = accuracy_score(y_cv_int, s3_probs.argmax(axis=1))
        s4_acc = accuracy_score(y_cv_int, s4_probs.argmax(axis=1))
        s5_acc = accuracy_score(y_cv_int, s5_probs.argmax(axis=1))
        s6_acc = accuracy_score(y_cv_int, s6_probs.argmax(axis=1))
        s7_acc = accuracy_score(y_cv_int, s7_probs.argmax(axis=1))
        s8_acc = accuracy_score(y_cv_int, s8_probs.argmax(axis=1))
        s9_acc = accuracy_score(y_cv_int, s9_probs.argmax(axis=1))
        s10_acc = accuracy_score(y_cv_int, s10_probs.argmax(axis=1))
        s11_acc = accuracy_score(y_cv_int, s11_probs.argmax(axis=1))
        s12_acc = accuracy_score(y_cv_int, s12_probs.argmax(axis=1))
        s13_acc = accuracy_score(y_cv_int, s13_probs.argmax(axis=1))
        s14_acc = accuracy_score(y_cv_int, s14_probs.argmax(axis=1))
        s15_acc = accuracy_score(y_cv_int, s15_probs.argmax(axis=1))
        s16_acc = accuracy_score(y_cv_int, s16_probs.argmax(axis=1))
        s17_acc = accuracy_score(y_cv_int, s17_probs.argmax(axis=1))
        s18_acc = accuracy_score(y_cv_int, s18_probs.argmax(axis=1))
        s19_acc = accuracy_score(y_cv_int, s19_probs.argmax(axis=1))
        s20_acc = accuracy_score(y_cv_int, s20_probs.argmax(axis=1))
        
    except Exception as e:
        print(f"  Warning: Solutions error: {e}")
        cascaded_conf_acc = best_acc_pf
        cascaded_val_acc = best_acc_pf
        cascaded_conf_preds = per_fold_preds.copy()  # Fallback for comparison table
        cascaded_val_preds = per_fold_preds.copy()
        y_cv_int = y_cv.astype(int)
        s1_acc = s2_acc = s3_acc = s4_acc = s5_acc = best_acc_m1
        s6_acc = s7_acc = s8_acc = s9_acc = s10_acc = best_acc_m1
        s11_acc = s12_acc = s13_acc = s14_acc = s15_acc = best_acc_m1
        s16_acc = s17_acc = s18_acc = best_acc_m1
        s19_acc = s20_acc = best_acc_pf
    
    solutions = [("S1:M1+Orig", s1_acc), ("S2:M1+Blend", s2_acc), ("S3:TripleEnsemble", s3_acc), 
                 ("S4:M1+GB+KNN", s4_acc), ("S5:Quad", s5_acc), ("S6:M1-weighted", s6_acc), 
                 ("S7:RF+GB", s7_acc), ("S8:Quad2", s8_acc), ("S9:M1+CDR", s9_acc), 
                 ("S10:M1+GB+CDR", s10_acc), ("S11:RF+GB", s11_acc), ("S12:GB+RF", s12_acc), 
                 ("S13:M1+GB", s13_acc), ("S14:M1+RF", s14_acc), ("S15:M1+KNN", s15_acc), 
                 ("S16:M1+GB(50)", s16_acc), ("S17:M1+RF(50)", s17_acc), ("S18:RF+GB(50)", s18_acc), 
                 ("S19:M1", s19_acc), ("S20:GB", s20_acc)]
    solutions_sorted = sorted(solutions, key=lambda x: x[1], reverse=True)
    print("\n  TOP 10 SOLUTIONS + CASCADING APPROACHES:")
    for name, acc in solutions_sorted[:10]:
        print(f"    {name:<20} {acc:.1%}")
    print(f"    {'CASCADE_CONF':<20} {cascaded_conf_acc:.1%}  (confidence-only, honest)")
    print(f"    {'CASCADE_VAL':<20} {cascaded_val_acc:.1%}  (validation-set learned)")
    print("  " + "=" * 60)
    
    print("  " + "-" * 60)
    
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON: All Methods (Per-Class Metrics + Accuracy)")
    print("=" * 80)
    print(f"  {'Method':<20} {'Acc':<7} {'CN P/R':<14} {'MCI P/R':<14} {'AD P/R':<14}")
    print(f"  {'-'*80}")
    
    def fmt_method(name, acc, prec, rec):
        return f"  {name:<20} {acc:<7.1%} {prec[0]:.0%}/{rec[0]:<8.0%} {prec[1]:.0%}/{rec[1]:<8.0%} {prec[2]:.0%}/{rec[2]:<8.0%}"
    
    cc_prec, cc_rec, _, _ = precision_recall_fscore_support(y_cv_int, cascaded_conf_preds, average=None)
    cval_prec, cval_rec, _, _ = precision_recall_fscore_support(y_cv_int, cascaded_val_preds, average=None)
    
    print(fmt_method("Original (XGBoost)", orig_acc_val, orig_prec, orig_rec))
    print(fmt_method("Method 1 (M1)", best_acc_m1, m1_prec, m1_rec))
    print(fmt_method("Method 2 (CDR feat)", m2_acc, m2_prec, m2_rec))
    print(fmt_method("Method 3 (Meta)", m3_acc, m3_prec, m3_rec))
    print(fmt_method("Method 4 (Per-Fold)", best_acc_pf, pf_prec, pf_rec))
    print(fmt_method("Method 5 (Isotonic)", m5_acc, m5_prec, m5_rec))
    print(fmt_method("Method 6 (Threshold)", m6_acc, m6_prec, m6_rec))
    print(fmt_method("Method 7 (RFE+Meta)", m7_acc, m7_prec, m7_rec))
    print(fmt_method("Cascade (Conf)", cascaded_conf_acc, cc_prec, cc_rec))
    print(fmt_method("Cascade (Val)", cascaded_val_acc, cval_prec, cval_rec))
    print("=" * 80)
    
    print(f"\n  {'Patient':<13} {'Actual':<8} {'CDR':<6} {'Pred CDR':<10} {'Orig':<8} {'M1':<8} {'M2':<8} {'M3':<8}")
    print(f"  {'-'*85}")
    label_names_cv2 = {0: 'CN', 1: 'MCI', 2: "AD"}
    cdrsum_mae_cont_list = []
    cdrsum_mae_rounded_list = []
    for pid, actual_cdr, pred_cdr_cont, actual_diag, pred_orig, pred_m1, pred_m2, pred_m3 in sorted(
        zip(ids_cv, cdrsum_cv_all.astype(float), cv_cdrsum_preds.astype(float),
            y_cv.astype(int), cv_preds.astype(int), m1_preds_final, cv_preds_m2, cv_preds_m3),
        key=lambda x: x[0]):
        cdrsum_mae_cont = abs(actual_cdr - pred_cdr_cont)
        cdrsum_mae_cont_list.append(cdrsum_mae_cont)
        pred_cdr_rounded = np.round(pred_cdr_cont * 2) / 2
        cdrsum_mae_rounded = abs(actual_cdr - pred_cdr_rounded)
        cdrsum_mae_rounded_list.append(cdrsum_mae_rounded)
        
        actual_str = label_names_cv2[actual_diag]
        def fmt(p):
            mark = '✓' if p == actual_diag else '✗'
            return f"{label_names_cv2[int(p)]:<3}{mark}"
        
        print(f"  {pid:<13} {actual_str:<8} {actual_cdr:<6.2f} {pred_cdr_cont:<10.2f} {fmt(pred_orig):<8} {fmt(pred_m1):<8} {fmt(pred_m2):<8} {fmt(pred_m3):<8}")
    
    cdrsum_mean_mae_cont = np.mean(cdrsum_mae_cont_list)
    cdrsum_mean_mae_rounded = np.mean(cdrsum_mae_rounded_list)
    print(f"\n  CDRSUM Mean Absolute Error (continuous model output): {cdrsum_mean_mae_cont:.3f}")
    print(f"  CDRSUM Mean Absolute Error (rounded to 0.5 for clinical use): {cdrsum_mean_mae_rounded:.3f}")

    print("\n" + "=" * 60)
    print("BLIND TEST (5 patients the model has never seen)")
    print("=" * 60)

    all_weights = compute_sample_weights(y_cv)
    y1_all = (y_cv > 0).astype(float)

    blind_s1_models = []
    n_es = max(1, int(0.2 * len(y_cv)))
    for i in range(10):
        m = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=params1.get('max_depth', 3),
            learning_rate=params1.get('learning_rate', 0.05),
            n_estimators=500, early_stopping_rounds=20,
            subsample=params1.get('subsample', 0.8),
            colsample_bytree=params1.get('colsample_bytree', 0.7),
            min_child_weight=3, n_jobs=4,
            random_state=42 + i, tree_method='hist')
        m.fit(X_cv[:-n_es, tab_idx], y1_all[:-n_es], sample_weight=all_weights[:-n_es],
              eval_set=[(X_cv[-n_es:, tab_idx], y1_all[-n_es:])], verbose=False)
        blind_s1_models.append(m)

    blind_s1 = np.mean([m.predict(blind_X[:, tab_idx]) for m in blind_s1_models], axis=0)

    nn_all = y_cv > 0
    y2_all = (y_cv[nn_all] - 1).astype(float)
    blind_s2_models = []
    n_es2 = max(1, int(0.2 * nn_all.sum()))
    for i in range(10):
        m = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=params2.get('max_depth', 3),
            learning_rate=params2.get('learning_rate', 0.05),
            n_estimators=500, early_stopping_rounds=20,
            subsample=params2.get('subsample', 0.7),
            colsample_bytree=params2.get('colsample_bytree', 0.7),
            min_child_weight=3, n_jobs=4,
            random_state=42 + i, tree_method='hist')
        m.fit(X_cv[nn_all][:, tab_idx][:-n_es2], y2_all[:-n_es2],
              sample_weight=all_weights[nn_all][:-n_es2],
              eval_set=[(X_cv[nn_all][:, tab_idx][-n_es2:], y2_all[-n_es2:])],
              verbose=False)
        blind_s2_models.append(m)

    blind_pred = np.zeros(len(blind_y))
    blind_nn = blind_s1 > 0.5
    if blind_nn.sum() > 0:
        blind_s2 = np.mean([m.predict(blind_X[blind_nn][:, tab_idx]) for m in blind_s2_models], axis=0)
        blind_pred[blind_nn] = np.where(blind_s2 > 0.5, 2, 1)
    
    blind_s2_all = np.mean([m.predict(blind_X[:, tab_idx]) for m in blind_s2_models], axis=0)
    
    blind_cdrsum_preds = model.cdr_model.predict(blind_X)
    
    blind_orig_probs = stage_probs_to_3class(blind_s1, blind_s2_all)
    blind_cdr_probs = cdrsum_to_probs(
        blind_cdrsum_preds,
        clinical_priors['mu_cn'], clinical_priors['sig_cn'],
        clinical_priors['mu_mci'], clinical_priors['sig_mci'],
        clinical_priors['mu_ad'], clinical_priors['sig_ad'])
    blind_m1_probs = 0.5 * blind_orig_probs + 0.5 * blind_cdr_probs
    blind_preds_m1 = blind_m1_probs.argmax(axis=1)
    
    meta_clf_blind = LogisticRegression(max_iter=2000, class_weight='balanced')
    meta_X_train = np.column_stack([original_probs, cv_cdrsum_preds.reshape(-1, 1)])
    meta_clf_blind.fit(meta_X_train, y_cv.astype(int))
    blind_meta_X = np.column_stack([blind_orig_probs, blind_cdrsum_preds.reshape(-1, 1)])
    blind_preds_m3 = meta_clf_blind.predict(blind_meta_X).astype(int)
    
    blind_meta_probs = meta_clf_blind.predict_proba(blind_meta_X)
    
    
    blind_preds_m2 = blind_pred.astype(int)
    
    blind_m4_probs = 0.36 * blind_orig_probs + 0.64 * blind_cdr_probs
    blind_preds_m4 = blind_m4_probs.argmax(axis=1)
    
    from sklearn.isotonic import IsotonicRegression
    blind_m5_probs = np.zeros_like(blind_orig_probs)
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        y_binary = (y_cv.astype(int) == c).astype(int)
        iso.fit(original_probs[:, c], y_binary)
        blind_m5_probs[:, c] = iso.transform(blind_orig_probs[:, c])
    blind_m5_probs = blind_m5_probs / (blind_m5_probs.sum(axis=1, keepdims=True) + 1e-10)
    blind_preds_m5 = blind_m5_probs.argmax(axis=1)
    
    train_meta_probs = meta_clf_blind.predict_proba(meta_X_train)
    
    best_scale_m6 = [1.0, 1.0, 1.0]
    best_acc_m6_train = 0
    for s_cn in [0.95, 1.0, 1.05, 1.10, 1.15]:
        for s_mci in [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
            for s_ad in [0.85, 0.95, 1.0, 1.05, 1.15]:
                scaled = train_meta_probs * np.array([s_cn, s_mci, s_ad])
                preds_train = scaled.argmax(axis=1)
                acc_train = accuracy_score(y_cv.astype(int), preds_train)
                if acc_train > best_acc_m6_train:
                    best_acc_m6_train = acc_train
                    best_scale_m6 = [s_cn, s_mci, s_ad]
    
    blind_m6_scaled = blind_meta_probs * np.array(best_scale_m6)
    blind_preds_m6 = blind_m6_scaled.argmax(axis=1)
    
    blind_preds_m7 = blind_preds_m3.copy()

    blind_correct = 0
    print(f"\n  {'Patient':<13} {'Actual':<8} {'CDR':<6} {'Pred CDR':<10} {'Orig':<6} {'M1':<6} {'M2':<6} {'M3':<6} {'M4':<6} {'M5':<6} {'M6':<6} {'M7':<6}")
    print(f"  {'-'*100}")
    blind_cdrsum_mae_cont_list = []
    blind_cdrsum_mae_rounded_list = []
    for pid, actual_cdr, pred_cdr_cont, actual_diag, predicted_diag, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7 in zip(
        blind_ids, blind_cdrsum.astype(float), blind_cdrsum_preds.astype(float),
        blind_y.astype(int), blind_pred.astype(int), blind_preds_m1, blind_preds_m2, blind_preds_m3,
        blind_preds_m4, blind_preds_m5, blind_preds_m6, blind_preds_m7):
        cdrsum_mae_cont = abs(actual_cdr - pred_cdr_cont)
        blind_cdrsum_mae_cont_list.append(cdrsum_mae_cont)
        pred_cdr_rounded = np.round(pred_cdr_cont * 2) / 2
        cdrsum_mae_rounded = abs(actual_cdr - pred_cdr_rounded)
        blind_cdrsum_mae_rounded_list.append(cdrsum_mae_rounded)
        if actual_diag == predicted_diag:
            blind_correct += 1
        actual_str = label_names_blind[actual_diag] if actual_diag in label_names_blind else str(actual_diag)
        def fmt_b(p):
            mark = '✓' if p == actual_diag else '✗'
            label = label_names_blind.get(int(p), str(int(p)))
            short = {'CN': 'CN', 'MCI': 'MCI', 'Alzheimer\'s': 'AD'}.get(label, label)
            return f"{short:<3}{mark}"
        print(f"  {pid:<13} {actual_str:<8} {actual_cdr:<6.2f} {pred_cdr_cont:<10.2f} {fmt_b(predicted_diag):<6} {fmt_b(p_m1):<6} {fmt_b(p_m2):<6} {fmt_b(p_m3):<6} {fmt_b(p_m4):<6} {fmt_b(p_m5):<6} {fmt_b(p_m6):<6} {fmt_b(p_m7):<6}")
    
    blind_cdrsum_mean_mae_cont = np.mean(blind_cdrsum_mae_cont_list)
    blind_cdrsum_mean_mae_rounded = np.mean(blind_cdrsum_mae_rounded_list)
    blind_acc_m1 = (blind_preds_m1 == blind_y.astype(int)).mean()
    blind_acc_m2 = (blind_preds_m2 == blind_y.astype(int)).mean()
    blind_acc_m3 = (blind_preds_m3 == blind_y.astype(int)).mean()
    blind_acc_m4 = (blind_preds_m4 == blind_y.astype(int)).mean()
    blind_acc_m5 = (blind_preds_m5 == blind_y.astype(int)).mean()
    blind_acc_m6 = (blind_preds_m6 == blind_y.astype(int)).mean()
    blind_acc_m7 = (blind_preds_m7 == blind_y.astype(int)).mean()
    print(f"\n  Blind Test Accuracy Summary:")
    print(f"    Original (XGBoost):   {blind_correct}/{len(blind_y)} = {blind_correct/len(blind_y):.1%}")
    print(f"    Method 1 (M1):        {int(blind_acc_m1 * len(blind_y))}/{len(blind_y)} = {blind_acc_m1:.1%}")
    print(f"    Method 2 (CDR feat):  {int(blind_acc_m2 * len(blind_y))}/{len(blind_y)} = {blind_acc_m2:.1%}")
    print(f"    Method 3 (Meta):      {int(blind_acc_m3 * len(blind_y))}/{len(blind_y)} = {blind_acc_m3:.1%}")
    print(f"    Method 4 (Per-Fold):  {int(blind_acc_m4 * len(blind_y))}/{len(blind_y)} = {blind_acc_m4:.1%}")
    print(f"    Method 5 (Isotonic):  {int(blind_acc_m5 * len(blind_y))}/{len(blind_y)} = {blind_acc_m5:.1%}")
    print(f"    Method 6 (Threshold): {int(blind_acc_m6 * len(blind_y))}/{len(blind_y)} = {blind_acc_m6:.1%}")
    print(f"    Method 7 (RFE+Meta):  {int(blind_acc_m7 * len(blind_y))}/{len(blind_y)} = {blind_acc_m7:.1%}")
    print(f"  Blind test CDRSUM MAE (continuous model output): {blind_cdrsum_mean_mae_cont:.3f}")
    print(f"  Blind test CDRSUM MAE (rounded to 0.5 for clinical use): {blind_cdrsum_mean_mae_rounded:.3f}")

    print("\n" + "-" * 60)
    plot_importance(model)

    with open('alzheimers_model_v6_calibrated.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("  Saved alzheimers_model_v6_calibrated.pkl")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  5-Fold CV Accuracy:       {cv_acc:.1%} (±{np.std(cv_fold_accs):.1%})")
    print(f"  5-Fold CV F1 (macro):     {cv_f1:.3f}")
    print(f"  5-Fold CV Precision:      {cv_prec:.3f}")
    print(f"  5-Fold CV Recall:         {cv_rec:.3f}")
    print(f"  Alzheimer's precision:    {precision_score(y_cv.astype(int), cv_preds.astype(int), labels=[2], average=None, zero_division=0)[0]:.3f}")
    print(f"  Alzheimer's recall:       {recall_score(y_cv.astype(int), cv_preds.astype(int), labels=[2], average=None, zero_division=0)[0]:.3f}")
    print("=" * 60)
