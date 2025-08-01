"""
Quick test script to verify ML pose classifier is working correctly.
This helps debug if the issue is with the classifier or the integration.
"""

import joblib
import numpy as np
import os

def test_ml_classifier():
    """Test the ML classifier with sample data."""
    
    classifier_path = "pose_classifier.joblib"
    
    if not os.path.exists(classifier_path):
        print(f"❌ Classifier file not found: {classifier_path}")
        return
    
    try:
        # Load the classifier
        print(f"Loading classifier from {classifier_path}...")
        classifier = joblib.load(classifier_path)
        
        print(f"✅ Classifier loaded successfully!")
        print(f"   Type: {type(classifier)}")
        print(f"   Classes: {classifier.classes_}")
        print(f"   Expected features: {classifier.n_features_in_}")
        
        # Test with some sample feature vectors
        # These are typical values for different poses from the training data
        test_cases = {
            "standing_sample": [
                -50, -80, -130,  # head_to_shoulder, shoulder_to_hip, head_to_hip (negative = proper order)
                40, 35,          # shoulder_width, hip_width
                -60, 250,        # hip_to_knee (negative = hips above knees), knee_y
                -40, 290,        # knee_to_ankle, ankle_y
                1.6,             # aspect_ratio
                0.3, 0.6, 1.0    # normalized positions
            ],
            "sitting_sample": [
                -30, -40, -70,   # shorter distances (more compact)
                35, 40,          # similar widths
                -5, 180,         # knees closer to hip level
                -20, 200,        # ankles
                1.8,             # different aspect ratio
                0.4, 0.6, 1.0    # different proportions
            ]
        }
        
        print("\n🧪 Testing classifier predictions:")
        
        for pose_name, features in test_cases.items():
            if len(features) != classifier.n_features_in_:
                print(f"⚠️  Feature count mismatch for {pose_name}: {len(features)} vs {classifier.n_features_in_}")
                continue
                
            features_array = np.array(features).reshape(1, -1)
            prediction = classifier.predict(features_array)[0]
            probabilities = classifier.predict_proba(features_array)[0]
            
            print(f"\n  {pose_name}:")
            print(f"    Prediction: {prediction}")
            print(f"    Confidence: {max(probabilities):.3f}")
            
            # Show all class probabilities
            prob_dict = {classifier.classes_[i]: probabilities[i] for i in range(len(classifier.classes_))}
            for class_name, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"    {class_name}: {prob:.3f}")
        
        print("\n✅ Classifier test completed!")
        
    except Exception as e:
        print(f"❌ Error testing classifier: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== ML Pose Classifier Test ===")
    test_ml_classifier()
