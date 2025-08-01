"""
Rules-based Pose Classification (Original Implementation)
========================================================

This file contains the original rules-based pose classification logic
for reference and comparison with the ML-based classifier.

Created: July 2025
Purpose: Backup of original classification rules
"""

def classify_poses_rules_based(keypoints_data):
    """Classify poses as standing, sitting, crouching, or fallen based on rules."""
    classifications = []
    
    for person_data in keypoints_data:
        keypoints = {kp["name"]: (kp["x"], kp["y"]) for kp in person_data["keypoints"] if kp["visible"]}
        
        classification = {
            "person_id": person_data["person_id"],
            "pose_class": "unknown",
            "confidence": 0.0,
            "reasoning": [],
            "method": "rules_based"
        }
        
        # Check if we have enough keypoints for classification
        required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        missing_points = [point for point in required_points if point not in keypoints]
        
        if missing_points:
            classification["reasoning"].append(f"Missing keypoints: {missing_points}")
            classifications.append(classification)
            continue
        
        # Calculate body metrics
        head_y = keypoints["nose"][1]
        shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
        hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
        
        # Get knee and ankle positions if available
        knee_y = None
        ankle_y = None
        
        if "left_knee" in keypoints and "right_knee" in keypoints:
            knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
        if "left_ankle" in keypoints and "right_ankle" in keypoints:
            ankle_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
        
        # Calculate body orientation (vertical distance ratios)
        head_to_shoulder = abs(head_y - shoulder_y)
        shoulder_to_hip = abs(shoulder_y - hip_y)
        
        # Rule-based classification
        score_standing = 0
        score_sitting = 0  
        score_crouching = 0
        score_fallen = 0
        
        # FALLEN: Check if person is horizontal (head and hip at similar y-level)
        if abs(head_y - hip_y) < shoulder_to_hip * 0.5:
            score_fallen += 3
            classification["reasoning"].append("Head and hip at similar level (horizontal)")
        
        # STANDING: Head significantly above hips, knees below hips
        if head_y < hip_y - shoulder_to_hip * 1.5:  # Head well above hips
            score_standing += 2
            classification["reasoning"].append("Head well above hips")
            
            if knee_y and knee_y > hip_y:  # Knees below hips
                score_standing += 2
                classification["reasoning"].append("Knees below hips")
                
            if ankle_y and ankle_y > knee_y:  # Ankles below knees
                score_standing += 1
                classification["reasoning"].append("Ankles below knees")
        
        # SITTING: Hip and knee at similar level, head above hips
        if knee_y and abs(hip_y - knee_y) < shoulder_to_hip * 0.8:
            score_sitting += 2
            classification["reasoning"].append("Hips and knees at similar level")
            
            if head_y < hip_y:  # Head above hips
                score_sitting += 1
                classification["reasoning"].append("Head above hips")
        
        # CROUCHING: Knees significantly above hips, but head still visible
        if knee_y and knee_y < hip_y - shoulder_to_hip * 0.5:
            score_crouching += 2
            classification["reasoning"].append("Knees above hips")
            
            if head_y < shoulder_y:  # Head above shoulders
                score_crouching += 1
                classification["reasoning"].append("Head above shoulders")
        
        # Determine final classification
        scores = {
            "standing": score_standing,
            "sitting": score_sitting,
            "crouching": score_crouching,
            "fallen": score_fallen
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            pose_class = max(scores.keys(), key=lambda k: scores[k])
            classification["pose_class"] = pose_class
            classification["confidence"] = min(max_score / 5.0, 1.0)  # Normalize to 0-1
        else:
            classification["pose_class"] = "unknown"
            classification["confidence"] = 0.0
            classification["reasoning"].append("Insufficient evidence for any pose class")
        
        # Add raw metrics for debugging
        classification["metrics"] = {
            "head_y": float(head_y),
            "shoulder_y": float(shoulder_y), 
            "hip_y": float(hip_y),
            "knee_y": float(knee_y) if knee_y else None,
            "ankle_y": float(ankle_y) if ankle_y else None,
            "scores": scores
        }
        
        classifications.append(classification)
    
    return classifications

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points (point2 is the vertex)."""
    import math
    
    # Vector from point2 to point1
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    # Vector from point2 to point3  
    v2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return float(angle_deg)
