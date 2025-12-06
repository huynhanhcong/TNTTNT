import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# --- CẤU HÌNH ---
AUDIO_DIR = "C:/Users/08688/Downloads/Audio_Processed/Audio_Processed"  # <-- thay bằng thư mục chứa ~200 .wav của bạn
OUTPUT_CSV = "FULL_DATASET_edited.csv"
SR = 16000  # sampling rate để load (thay đổi nếu cần)
N_MFCC = 13

# --- HÀM HỖ TRỢ LẤY LABEL ---
def infer_label_from_path(path):
    """
    Cách suy đoán label:
    1) Nếu file nằm trong thư mục con (ví dụ `dataset/label_name/*.wav`) -> lấy tên thư mục con.
    2) Ngược lại, nếu filename bắt đầu bằng 'label_' hoặc 'label-' hoặc 'label.' -> lấy phần trước dấu phân tách.
    3) Nếu không có -> trả 'unknown'.
    """
    # 1) parent folder
    parent = os.path.basename(os.path.dirname(path))
    root = os.path.basename(os.path.abspath(AUDIO_DIR))
    if parent and parent != root:
        return parent

    fname = os.path.basename(path)
    # 2) tách theo '_' hoặc '-' hoặc '.'
    for sep in ['_', '-', '.']:
        if sep in fname:
            maybe = fname.split(sep)[0]
            # tránh trả 'audio' các từ chung -> nhưng ta cứ trả cái này; bạn có thể tinh chỉnh
            return maybe
    return "unknown"

# --- HÀM TRÍCH FEATURE CHO 1 FILE ---
def extract_features(file_path, sr=SR, n_mfcc=N_MFCC):
    try:
        # dùng soundfile để đọc (giữ nguyên dtype), librosa cho chuyển đổi
        x, file_sr = sf.read(file_path)
        if x.ndim > 1:
            x = np.mean(x, axis=1)  # convert to mono
        if file_sr != sr:
            x = librosa.resample(x.astype(float), file_sr, sr)
            file_sr = sr

        # đảm bảo float
        x = x.astype(float)

        # trim silence optional (uncomment nếu muốn)
        # x, _ = librosa.effects.trim(x)

        # cơ bản
        duration = len(x) / file_sr
        zcr = librosa.feature.zero_crossing_rate(x)[0].mean()
        rms = librosa.feature.rms(y=x)[0].mean()
        spec_cent = librosa.feature.spectral_centroid(y=x, sr=file_sr)[0].mean()
        spec_rolloff = librosa.feature.spectral_rolloff(y=x, sr=file_sr)[0].mean()
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=file_sr)[0].mean()
        # mfcc
        mfcc = librosa.feature.mfcc(y=x, sr=file_sr, n_mfcc=n_mfcc)
        mfcc_means = mfcc.mean(axis=1)
        # chroma
        chroma = librosa.feature.chroma_stft(y=x, sr=file_sr)
        chroma_means = chroma.mean(axis=1)
        # spectral contrast
        contrast = librosa.feature.spectral_contrast(y=x, sr=file_sr)
        contrast_means = contrast.mean(axis=1)
        # tonnetz (requires harmonic)
        y_harmonic = librosa.effects.harmonic(x)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=file_sr)
        tonnetz_means = tonnetz.mean(axis=1)

        # gom vao dict
        features = {
            "filename": os.path.basename(file_path),
            "filepath": os.path.abspath(file_path),
            "duration": duration,
            "zcr_mean": zcr,
            "rms_mean": rms,
            "spec_centroid_mean": spec_cent,
            "spec_rolloff_mean": spec_rolloff,
            "spec_bandwidth_mean": spec_bandwidth,
        }

        # add mfccs: mfcc_1 ... mfcc_13 (hoặc N_MFCC)
        for i, v in enumerate(mfcc_means, 1):
            features[f"mfcc_{i}"] = float(v)

        # chroma_1..12
        for i, v in enumerate(chroma_means, 1):
            features[f"chroma_{i}"] = float(v)

        # contrast
        for i, v in enumerate(contrast_means, 1):
            features[f"contrast_{i}"] = float(v)

        # tonnetz
        for i, v in enumerate(tonnetz_means, 1):
            features[f"tonnetz_{i}"] = float(v)

        return features
    except Exception as e:
        print(f"[ERROR] Cannot process {file_path}: {e}")
        return None

# --- DUYỆT THƯ MỤC, TRÍCH FEATURE CHO MỌI FILE .wav ---
rows = []
wav_files = []
for root_dir, dirs, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root_dir, f))

print(f"Found {len(wav_files)} .wav files. Processing...")

for fp in tqdm(sorted(wav_files)):
    feats = extract_features(fp)
    if feats is None:
        continue
    # infer label
    label = infer_label_from_path(fp)
    feats["label"] = label
    rows.append(feats)

# --- LƯU RA CSV ---
if len(rows) == 0:
    raise RuntimeError("No features extracted. Check your audio files and AUDIO_DIR path.")

df = pd.DataFrame(rows)
# sắp xếp cột cho đẹp: filename, label, filepath, rest...
cols = ["filename", "label", "filepath"] + [c for c in df.columns if c not in ("filename","label","filepath")]
df = df[cols]
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved features to {OUTPUT_CSV}. Shape: {df.shape}")
