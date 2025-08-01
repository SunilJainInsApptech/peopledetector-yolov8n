import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def extract_pose_features(keypoints):
    """Extract meaningful features from COCO keypoints for classification."""
    features = []
    
    # COCO keypoint format: [x1, y1, v1, x2, y2, v2, ...]
    # Where v is visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    
    # Convert flat array to (x, y, visibility) tuples
    keypoint_data = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            keypoint_data.append((x, y, v))
        else:
            keypoint_data.append((0, 0, 0))
    
    # Ensure we have 17 keypoints (COCO format)
    while len(keypoint_data) < 17:
        keypoint_data.append((0, 0, 0))
    
    # COCO keypoint order
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Create keypoint dictionary for visible points only
    kp_dict = {}
    for i, name in enumerate(keypoint_names):
        if i < len(keypoint_data):
            x, y, v = keypoint_data[i]
            if v > 0 and x > 0 and y > 0:  # Visible and valid coordinates
                kp_dict[name] = (x, y)
    
    # Extract features if we have key body parts
    required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if all(point in kp_dict for point in required_points):
        # Basic body measurements
        head_y = kp_dict["nose"][1]
        shoulder_y = (kp_dict["left_shoulder"][1] + kp_dict["right_shoulder"][1]) / 2
        hip_y = (kp_dict["left_hip"][1] + kp_dict["right_hip"][1]) / 2
        
        # Vertical distances (key for pose classification)
        head_to_shoulder = head_y - shoulder_y
        shoulder_to_hip = shoulder_y - hip_y
        head_to_hip = head_y - hip_y
        
        # Body proportions
        shoulder_width = abs(kp_dict["left_shoulder"][0] - kp_dict["right_shoulder"][0])
        hip_width = abs(kp_dict["left_hip"][0] - kp_dict["right_hip"][0])
        
        features.extend([
            head_to_shoulder,    # Negative if head above shoulders (normal)
            shoulder_to_hip,     # Negative if shoulders above hips (normal)
            head_to_hip,         # Overall body height
            shoulder_width,      # Body width at shoulders
            hip_width,           # Body width at hips
        ])
        
        # Knee positions if available
        if "left_knee" in kp_dict and "right_knee" in kp_dict:
            knee_y = (kp_dict["left_knee"][1] + kp_dict["right_knee"][1]) / 2
            hip_to_knee = hip_y - knee_y
            features.extend([
                hip_to_knee,     # Negative if hips above knees (normal standing)
                knee_y           # Absolute knee position
            ])
        else:
            features.extend([0, 0])
            
        # Ankle positions if available
        if "left_ankle" in kp_dict and "right_ankle" in kp_dict:
            ankle_y = (kp_dict["left_ankle"][1] + kp_dict["right_ankle"][1]) / 2
            if "left_knee" in kp_dict and "right_knee" in kp_dict:
                knee_to_ankle = knee_y - ankle_y
                features.extend([knee_to_ankle, ankle_y])
            else:
                features.extend([0, ankle_y])
        else:
            features.extend([0, 0])
            
        # Body aspect ratio
        if shoulder_to_hip != 0:
            features.append(abs(head_to_hip) / abs(shoulder_to_hip))
        else:
            features.append(0)
            
        # Normalized positions (relative to body height)
        body_height = abs(head_to_hip) if abs(head_to_hip) > 0 else 1
        features.extend([
            head_y / body_height,
            shoulder_y / body_height, 
            hip_y / body_height
        ])
        
    else:
        # If missing key points, return zeros
        features = [0] * 13
        
    return features

def load_coco_dataset(coco_file):
    """Load COCO annotations and extract pose features and labels."""
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    X = []  # Features
    y = []  # Labels
    
    print(f"Loading dataset from {coco_file}")
    print(f"Found {len(coco_data['annotations'])} annotations")
    
    # Track label distribution
    label_counts = {}
    
    for annotation in coco_data['annotations']:
        # Check if this annotation has keypoints and pose label
        if 'keypoints' in annotation and 'attributes' in annotation:
            keypoints = annotation['keypoints']
            
            # Get pose label from attributes
            if 'pose_label' in annotation['attributes']:
                pose_label = annotation['attributes']['pose_label']
                
                # Extract features
                features = extract_pose_features(keypoints)
                
                # Only use if we got valid features (not all zeros)
                if any(f != 0 for f in features):
                    X.append(features)
                    y.append(pose_label)
                    
                    # Count labels
                    label_counts[pose_label] = label_counts.get(pose_label, 0) + 1
    
    print(f"\nDataset summary:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")
    print(f"Total valid samples: {len(X)}")
    
    return np.array(X), np.array(y)

def train_pose_classifier(coco_file, output_file):
    """Train pose classifier from COCO dataset."""
    
    # Load dataset
    X, y = load_coco_dataset(coco_file)
    
    if len(X) == 0:
        print("Error: No valid training data found!")
        return None
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    feature_names = [
        'head_to_shoulder', 'shoulder_to_hip', 'head_to_hip',
        'shoulder_width', 'hip_width', 'hip_to_knee', 'knee_y',
        'knee_to_ankle', 'ankle_y', 'aspect_ratio', 
        'norm_head_y', 'norm_shoulder_y', 'norm_hip_y'
    ]
    
    print("\nTop 5 most important features:")
    feature_importance = list(zip(feature_names, clf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for name, importance in feature_importance[:5]:
        print(f"  {name}: {importance:.3f}")
    
    # Save model
    joblib.dump(clf, output_file)
    print(f"\nModel saved to {output_file}")
    
    return clf

if __name__ == "__main__":
    coco_file = "coco_annotations.json"
    output_file = "pose_classifier.joblib"
    
    if not os.path.exists(coco_file):
        print(f"Error: {coco_file} not found!")
        exit(1)
    
    # Train the classifier
    model = train_pose_classifier(coco_file, output_file)
    
    if model:
        print(f"\n✅ Training complete! Use {output_file} in your Viam module.")
        print("\nTo use in your module, add to robot config:")
        print('{')
        print('  "attributes": {')
        print('    "model_location": "yolov8n.pt",')
        print(f'    "pose_classifier_path": "{output_file}"')
        print('  }')
        print('}')
