#!/usr/bin/env python3
"""
Diagnostic: compare feature vectors from preprocessing, SMOTE, and inference extractors.

Usage:
    python3 tools/diagnose_feature_parity.py --cfg configs/default.yaml \
        --frame_labels path/to/frame_labels.csv --n_samples 200 --out_dir diagnostics

If frame_labels not provided, the script will try to run MABE preprocessing.load_data() to build labels.
"""
import argparse
import json
import logging
import random
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Insert src to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mabe.utils.logger import get_logger
from mabe.preprocessing import MABEDataPreprocessor, MouseBehaviorDataset
from mabe import smote_augmentation as smote_mod
from mabe.inference import TestMouseBehaviorDataset

logger = get_logger("diagnose_feature_parity")

DUMMY_VECTOR = None  # will be set to [0.0]*26 once we confirm feature dim


def load_config(cfg_path: Path) -> dict:
    import yaml
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def sample_frame_labels(frame_labels_df: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    if n_samples >= len(frame_labels_df):
        return frame_labels_df.copy().reset_index(drop=True)
    return frame_labels_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def is_dummy_vector(vec: np.ndarray, tol=1e-6) -> bool:
    if vec is None:
        return True
    if not isinstance(vec, np.ndarray):
        try:
            vec = np.array(vec, dtype=float)
        except Exception:
            return True
    return np.all(np.abs(vec) <= tol)


def safe_extract_preprocess(preprocessor: MABEDataPreprocessor, row) -> np.ndarray:
    try:
        # Use MouseBehaviorDataset.extract_tracking_features via a dataset instance to reuse existing logic
        ds = MouseBehaviorDataset(pd.DataFrame([row]), preprocessor.tracking_data, augment=False)
        features, _ = ds[0]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Preprocessing extractor failed for row {row.get('video_id','?')},{row.get('frame','?')}: {e}")
        return None


def safe_extract_smote(smote: smote_mod.BehavioralSMOTE, tracking_data: dict, row) -> np.ndarray:
    try:
        # smote._extract_tracking_features expects tracking_data, video_id, frame, agent_id, target_id
        feat = smote._extract_tracking_features(tracking_data, row['video_id'], int(row['frame']), row['agent_id'], row['target_id'])
        return np.array(feat, dtype=np.float32) if feat is not None else None
    except Exception as e:
        logger.warning(f"SMOTE extractor failed for row {row.get('video_id','?')},{row.get('frame','?')}: {e}")
        return None


def safe_extract_inference(test_ds: TestMouseBehaviorDataset, row) -> np.ndarray:
    try:
        # TestMouseBehaviorDataset.extract_tracking_features expects video_id, frame, agent_id, target_id
        feat = test_ds.extract_tracking_features(row['video_id'], int(row['frame']), row['agent_id'], row['target_id'])
        return np.array(feat, dtype=np.float32) if feat is not None else None
    except Exception as e:
        logger.warning(f"Inference extractor failed for row {row.get('video_id','?')},{row.get('frame','?')}: {e}")
        return None


def compare_vectors(v1: np.ndarray, v2: np.ndarray) -> dict:
    # Return L2, L1, max abs diff, exact equality
    out = {}
    if v1 is None or v2 is None:
        out.update({'l2': None, 'l1': None, 'max_abs': None, 'exact': False})
        return out
    if v1.shape != v2.shape:
        out.update({'l2': float('inf'), 'l1': float('inf'), 'max_abs': float('inf'), 'exact': False})
        return out
    diff = v1 - v2
    out['l2'] = float(np.linalg.norm(diff))
    out['l1'] = float(np.sum(np.abs(diff)))
    out['max_abs'] = float(np.max(np.abs(diff)))
    out['exact'] = bool(np.allclose(v1, v2, atol=1e-8))
    return out


def check_tracking_data_availability(cfg: dict, preprocessor: MABEDataPreprocessor, frame_labels_df: pd.DataFrame) -> dict:
    """Check tracking data availability and report issues"""
    logger.info("Checking tracking data availability...")
    
    availability_report = {
        'dataset_path': str(preprocessor.dataset_path),
        'requested_videos': len(frame_labels_df['video_id'].unique()),
        'tracking_data_loaded': len(preprocessor.tracking_data),
        'tracking_data_keys': list(preprocessor.tracking_data.keys()),
        'missing_videos': [],
        'file_availability': {},
        'column_variations': {},
        'type_mismatches': []
    }
    
    # Check for tracking files at expected locations
    keypoints_dir = preprocessor.dataset_path / "train_tracking" / "MABe22_keypoints"
    converted_dir = preprocessor.dataset_path / "converted_tracking"
    
    availability_report['keypoints_dir_exists'] = keypoints_dir.exists()
    availability_report['converted_dir_exists'] = converted_dir.exists()
    
    # Check individual video files
    unique_videos = frame_labels_df['video_id'].unique()
    for video_id in unique_videos:
        video_id_str = str(video_id)
        tracking_file = keypoints_dir / f"{video_id}.parquet"
        
        availability_report['file_availability'][video_id_str] = {
            'keypoints_file_exists': tracking_file.exists(),
            'in_tracking_data': video_id_str in preprocessor.tracking_data,
            'tracking_data_type': type(preprocessor.tracking_data.get(video_id_str, None)).__name__ if video_id_str in preprocessor.tracking_data else 'None'
        }
        
        if video_id_str not in preprocessor.tracking_data:
            availability_report['missing_videos'].append(video_id_str)
    
    # Check column variations in available tracking data
    for video_id, tracking_df in preprocessor.tracking_data.items():
        if tracking_df is not None and not tracking_df.empty:
            columns = list(tracking_df.columns)
            availability_report['column_variations'][video_id] = {
                'total_columns': len(columns),
                'has_frame_column': any(col.lower() == 'frame' for col in columns),
                'frame_columns': [col for col in columns if col.lower() == 'frame'],
                'mouse_columns': [col for col in columns if '_body_center_x' in col],
                'sample_columns': columns[:10]  # First 10 columns
            }
    
    # Check for type mismatches
    frame_labels_video_ids = set(str(vid) for vid in frame_labels_df['video_id'].unique())
    tracking_data_video_ids = set(preprocessor.tracking_data.keys())
    
    availability_report['type_mismatches'] = {
        'frame_labels_types': [type(vid).__name__ for vid in frame_labels_df['video_id'].unique()],
        'tracking_data_types': [type(k).__name__ for k in preprocessor.tracking_data.keys()],
        'intersection': len(frame_labels_video_ids & tracking_data_video_ids),
        'frame_labels_only': list(frame_labels_video_ids - tracking_data_video_ids),
        'tracking_data_only': list(tracking_data_video_ids - frame_labels_video_ids)
    }
    
    return availability_report


def run_diagnostic(cfg_path: Path, frame_labels_path: Path, n_samples: int, out_dir: Path, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(cfg_path)

    # Initialize preprocessor to load tracking data
    preprocessor = MABEDataPreprocessor(cfg)

    # Load or build frame labels
    if frame_labels_path and frame_labels_path.exists():
        logger.info(f"Loading frame labels from {frame_labels_path}")
        frame_labels_df = pd.read_csv(frame_labels_path)
    else:
        logger.info("No frame_labels path provided or file missing - calling preprocessor.load_data() to build frame labels")
        frame_labels_df = preprocessor.load_data()
    if frame_labels_df is None or frame_labels_df.empty:
        logger.error("Frame labels are empty; aborting diagnostic")
        return

    # Load tracking data (preprocessor.load_data does this when building labels)
    if not preprocessor.tracking_data:
        logger.info("Loading tracking data using preprocessor._load_tracking_data()")
        preprocessor._load_tracking_data(frame_labels_df, max_videos=cfg.get('training', {}).get('max_videos', 5))

    # Check tracking data availability
    availability_report = check_tracking_data_availability(cfg, preprocessor, frame_labels_df)
    
    # Save availability report
    availability_json_path = out_dir / f"tracking_availability_{int(time.time())}.json"
    with open(availability_json_path, 'w') as f:
        json.dump(availability_report, f, indent=2)
    logger.info(f"Saved tracking data availability report to {availability_json_path}")
    
    # Log key findings
    logger.info(f"Tracking data availability summary:")
    logger.info(f"  Requested videos: {availability_report['requested_videos']}")
    logger.info(f"  Loaded videos: {availability_report['tracking_data_loaded']}")
    logger.info(f"  Missing videos: {len(availability_report['missing_videos'])}")
    logger.info(f"  Keypoints dir exists: {availability_report['keypoints_dir_exists']}")
    logger.info(f"  Converted dir exists: {availability_report['converted_dir_exists']}")

    tracking_data = preprocessor.tracking_data

    # Initialize SMOTE object
    smote = smote_mod.BehavioralSMOTE(
        k_neighbors=cfg.get('training', {}).get('smote_k_neighbors', 5),
        target_ratio=cfg.get('training', {}).get('smote_target_ratio', 0.5),
        random_state=cfg.get('seed', 42)
    )

    # Initialize inference dataset using a small test_df built from unique videos in frame_labels
    # Create a minimal test_df structure expected by TestMouseBehaviorDataset
    unique_videos = frame_labels_df['video_id'].unique()[:10]
    # Build fake test_df with required columns: video_id, frames_per_second, video_duration_sec
    test_rows = []
    # Attempt to infer fps/duration from train.csv if available
    train_csv = Path(cfg['dataset']['path']) / "train.csv"
    train_df = None
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
    for vid in unique_videos:
        fps = 30.0
        duration_sec = 10.0
        if train_df is not None:
            m = train_df[train_df['video_id'] == vid]
            if not m.empty:
                fps = float(m.iloc[0].get('frames_per_second', fps))
                duration_sec = float(m.iloc[0].get('video_duration_sec', duration_sec))
        test_rows.append({
            'video_id': vid,
            'frames_per_second': fps,
            'video_duration_sec': duration_sec
        })
    test_df = pd.DataFrame(test_rows)
    test_ds = TestMouseBehaviorDataset(test_df, tracking_data)

    # Determine dummy vector (use a sample extraction to discover feature dim and dummy pattern)
    # Try to get feature_dim from MouseBehaviorDataset
    try:
        sample_ds = MouseBehaviorDataset(frame_labels_df.head(1), tracking_data, augment=False)
        sample_feat, _ = sample_ds[0]
        feature_dim = len(sample_feat)
    except Exception:
        feature_dim = 26
    global DUMMY_VECTOR
    DUMMY_VECTOR = np.zeros(feature_dim, dtype=np.float32)

    logger.info(f"Feature dimension assumed: {feature_dim}")

    # Select samples
    sampled_df = sample_frame_labels(frame_labels_df, n_samples, seed=seed)
    logger.info(f"Selected {len(sampled_df)} samples for diagnostics")

    # Build results list
    results = []
    start_time = time.time()
    for idx, row in sampled_df.iterrows():
        row_dict = row.to_dict()
        # normalize types
        row_dict['video_id'] = str(row_dict['video_id'])
        # agent_id/target_id may be ints or strings
        row_dict['agent_id'] = str(row_dict.get('agent_id'))
        row_dict['target_id'] = str(row_dict.get('target_id'))
        # Ensure frame is int
        row_dict['frame'] = int(row_dict.get('frame', 0))

        preprocess_vec = safe_extract_preprocess(preprocessor, row_dict)
        smote_vec = safe_extract_smote(smote, tracking_data, row_dict)
        inference_vec = safe_extract_inference(test_ds, row_dict)

        # Normalize None -> None, else array
        def norm(v):
            if v is None:
                return None
            try:
                arr = np.array(v, dtype=np.float32)
                if arr.size != feature_dim:
                    # pad or trim
                    if arr.size < feature_dim:
                        padded = np.zeros(feature_dim, dtype=np.float32)
                        padded[:arr.size] = arr
                        return padded
                    else:
                        return arr[:feature_dim].astype(np.float32)
                return arr
            except Exception:
                return None

        preprocess_vec = norm(preprocess_vec)
        smote_vec = norm(smote_vec)
        inference_vec = norm(inference_vec)

        pre_is_dummy = (preprocess_vec is None) or is_dummy_vector(preprocess_vec)
        smote_is_dummy = (smote_vec is None) or is_dummy_vector(smote_vec)
        inf_is_dummy = (inference_vec is None) or is_dummy_vector(inference_vec)

        # Compare vectors pairwise
        pre_vs_inf = compare_vectors(preprocess_vec, inference_vec)
        pre_vs_smote = compare_vectors(preprocess_vec, smote_vec)
        smote_vs_inf = compare_vectors(smote_vec, inference_vec)

        row_res = {
            'index': int(idx),
            'video_id': row_dict['video_id'],
            'frame': int(row_dict['frame']),
            'agent_id': row_dict['agent_id'],
            'target_id': row_dict['target_id'],
            'pre_is_dummy': bool(pre_is_dummy),
            'smote_is_dummy': bool(smote_is_dummy),
            'inf_is_dummy': bool(inf_is_dummy),
            'pre_vs_inf_l2': pre_vs_inf['l2'],
            'pre_vs_inf_l1': pre_vs_inf['l1'],
            'pre_vs_inf_maxabs': pre_vs_inf['max_abs'],
            'pre_vs_inf_exact': pre_vs_inf['exact'],
            'pre_vs_smote_l2': pre_vs_smote['l2'],
            'smote_vs_inf_l2': smote_vs_inf['l2'],
        }
        results.append(row_res)

    elapsed = time.time() - start_time
    logger.info(f"Diagnostics completed in {elapsed:.1f}s")

    results_df = pd.DataFrame(results)
    # Summary stats
    summary = {}
    for col in ['pre_is_dummy', 'smote_is_dummy', 'inf_is_dummy']:
        summary[col] = int(results_df[col].sum())
    summary['total_samples'] = len(results_df)
    # fraction exact matches pre vs inf
    summary['pre_vs_inf_exact_count'] = int(results_df['pre_vs_inf_exact'].sum())
    # mean L2 where available
    summary['pre_vs_inf_l2_mean'] = float(results_df['pre_vs_inf_l2'].replace({None: np.nan}).dropna().mean()) if not results_df['pre_vs_inf_l2'].dropna().empty else None
    summary['pre_vs_smote_l2_mean'] = float(results_df['pre_vs_smote_l2'].replace({None: np.nan}).dropna().mean()) if not results_df['pre_vs_smote_l2'].dropna().empty else None
    summary['smote_vs_inf_l2_mean'] = float(results_df['smote_vs_inf_l2'].replace({None: np.nan}).dropna().mean()) if not results_df['smote_vs_inf_l2'].dropna().empty else None

    # Save outputs
    csv_path = out_dir / f"feature_parity_results_{int(time.time())}.csv"
    json_path = out_dir / f"feature_parity_summary_{int(time.time())}.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump({'summary': summary}, f, indent=2)

    logger.info(f"Saved per-sample results to {csv_path}")
    logger.info(f"Saved summary to {json_path}")
    logger.info("Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    return results_df, summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose feature parity between preprocess/SMOTE/inference extractors")
    parser.add_argument('--cfg', type=str, required=True, help='Path to config YAML (e.g., configs/default.yaml)')
    parser.add_argument('--frame_labels', type=str, default='', help='Path to frame labels CSV (optional). If missing, script will call preprocessor.load_data()')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples to test')
    parser.add_argument('--out_dir', type=str, default='diagnostics', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    frame_labels_path = Path(args.frame_labels) if args.frame_labels else None
    out_dir = Path(args.out_dir)
    run_diagnostic(cfg_path, frame_labels_path, args.n_samples, out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
