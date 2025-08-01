# ML Pose Classifier Integration

## Overview
Your YOLOv8n-pose Viam module now supports **both ML-based and rules-based pose classification**:

- **ML Classifier**: Uses your trained `pose_classifier.joblib` model (81% accuracy)
- **Rules-based Fallback**: Original logic preserved for reference and backup

## Quick Start

### 1. Configuration
Add `pose_classifier_path` to your robot configuration:

```json
{
  "services": [
    {
      "name": "yolov8-pose-detector",
      "type": "vision",
      "model": "rig-guardian:yolov8n-pose:yolov8n-pose",
      "attributes": {
        "model_location": "yolov8n-pose.pt",
        "pose_classifier_path": "pose_classifier.joblib"
      }
    }
  ]
}
```

### 2. File Placement
Place `pose_classifier.joblib` in your module directory:
```
yolov8n-pose/
├── pose_classifier.joblib   ← Your trained model
├── src/models/yolov8n_pose.py
└── ...
```

### 3. Dependencies
Install required packages:
```bash
pip install scikit-learn numpy joblib
```

## Classification Methods

### ML Classifier (Primary)
- **Accuracy**: 81% on test data, 77% cross-validation
- **Features**: 13 engineered features from pose keypoints
- **Classes**: standing, sitting, crouching, fallen
- **Confidence**: Actual probability scores from Random Forest

### Rules-based Fallback
- Automatically used if ML classifier fails to load
- Original implementation preserved in `rules_based_pose_classifier.py`
- Logic-based scoring system

## API Response Format

```json
{
  "person_id": 1,
  "pose_class": "standing",
  "confidence": 0.85,
  "method": "ml",
  "reasoning": ["ML prediction with 13 features"],
  "probabilities": {
    "standing": 0.85,
    "sitting": 0.10,
    "crouching": 0.03,
    "fallen": 0.02
  }
}
```

## Performance Comparison

| Method | Accuracy | Reliability | Speed |
|--------|----------|-------------|-------|
| ML Classifier | 81% | High | Fast |
| Rules-based | ~60% | Variable | Fast |

## Model Performance (ML)
- **Standing**: 91% recall (excellent)
- **Sitting**: 75% recall (good) 
- **Crouching**: 67% recall (acceptable)
- **Fallen**: 43% recall (needs improvement)

## Troubleshooting

### ML Classifier Not Loading
1. Check file path: `pose_classifier_path` in config
2. Verify file exists: `pose_classifier.joblib`
3. Check dependencies: `pip install scikit-learn joblib`
4. Review logs for error messages

### Fallback to Rules-based
If you see `"method": "rules_based"` in responses:
- ML classifier failed to load or predict
- Check configuration and file paths
- Ensure all dependencies are installed

## Files Reference

- `pose_classifier.joblib` - Trained ML model (81% accuracy)
- `rules_based_pose_classifier.py` - Original rules logic (backup)
- `viam_config_ml.json` - Sample configuration with ML enabled
- `train_pose_classifier.py` - Training script (for retraining)

## Next Steps

1. **Test Integration**: Run with your robot configuration
2. **Monitor Performance**: Check logs for classification results
3. **Retrain if Needed**: Use more data to improve "fallen" detection
4. **Fine-tune Thresholds**: Adjust confidence thresholds per your needs

The ML classifier should significantly improve pose detection reliability compared to the rules-based approach!
