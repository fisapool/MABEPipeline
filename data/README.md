# Data Directory

This directory contains data files for the MABE Pipeline.

## Expected Data Structure

The pipeline expects the following data structure:

```
data/
├── train.csv                    # Training video metadata
├── test.csv                     # Test video metadata
├── train_annotation/            # Training annotations
│   └── {lab_id}/
│       └── {video_id}.parquet
├── train_tracking/              # Training tracking data
│   └── {lab_id}/
│       └── {video_id}.parquet
├── test_tracking/               # Test tracking data
│   └── {lab_id}/
│       └── {video_id}.parquet
└── sample_frame_labels.csv      # Sample frame labels (optional)
```

## Data Sources

### Original MABE Dataset
The pipeline is designed to work with the MABE competition dataset structure:

- **train.csv**: Contains video metadata with columns: `video_id`, `lab_id`, `frames_per_second`, `video_duration_sec`
- **test.csv**: Contains test video metadata
- **train_annotation/**: Directory containing annotation files in parquet format
- **train_tracking/**: Directory containing tracking data files in parquet format
- **test_tracking/**: Directory containing test tracking data files in parquet format

### Frame Labels
The pipeline can work with preprocessed frame labels in CSV format:

```csv
video_id,frame,agent_id,target_id,behavior
video_001,100,agent1,target1,approach
video_001,101,agent1,target1,approach
...
```

## Data Configuration

Configure data paths in `configs/default.yaml`:

```yaml
dataset:
  path: C:/Users/MYPC/Documents/MABEDatasets/MABe-extracted
  train_csv: train.csv
  test_csv: test.csv
  frame_labels_sample: data/sample_frame_labels.csv
```

## Legacy Data Support

The pipeline provides backward compatibility for existing data formats:

### Legacy Frame Labels
If you have existing `mabe_frame_labels_*.csv` files, the pipeline can read them automatically. Use the migration script to convert them to the standardized format:

```bash
python scripts/migrate_frame_labels.py
```

### Data Validation
The pipeline includes data validation to ensure:
- Required files exist
- Data formats are correct
- Tracking data is properly loaded
- Feature extraction produces valid results

## Sample Data

For testing and development, you can create sample data using the data sampler:

```python
from mabe.data.data_sampler import create_sample
create_sample(cfg, 'data/sample_frame_labels.csv', n_frames=100)
```

## Data Processing

The pipeline processes data in the following stages:

1. **Data Loading**: Load video metadata and tracking data
2. **Feature Extraction**: Extract spatial, temporal, and interaction features
3. **Frame Label Creation**: Create frame-level labels from annotations
4. **Dataset Creation**: Create PyTorch datasets for training/inference

## Troubleshooting

### Common Data Issues

1. **Missing tracking data**: Ensure tracking files exist in the correct directory structure
2. **Invalid frame labels**: Check that frame labels have required columns
3. **Feature extraction errors**: Verify tracking data contains expected columns
4. **Memory issues**: Reduce `max_videos` in configuration for large datasets

### Data Validation

Run data validation to check your dataset:

```bash
python -c "from mabe.data.validate_dataset import validate_dataset; print(validate_dataset(cfg))"
```

### Performance Tips

- Use parquet format for large datasets (faster I/O)
- Enable data caching for repeated access
- Use data sampling for development and testing
- Monitor memory usage with large datasets
