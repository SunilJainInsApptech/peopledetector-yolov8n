import os
from pathlib import Path
from typing import ClassVar, Mapping, Any, Optional, List, cast
from typing_extensions import Self
from urllib.request import urlretrieve

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes


from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.logging import getLogger
from viam.utils import struct_to_dict

from ultralytics.engine.results import Results
from ultralytics import YOLO
import torch
import joblib
import numpy as np

# Import fall detection alerts
try:
    from ..fall_detection_alerts import FallDetectionAlerts
except ImportError:
    # When running as local module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from fall_detection_alerts import FallDetectionAlerts

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
)

class Yolov8nPose(Vision, EasyResource):
    """
    Vision represents a Vision service.
    """
        
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("rig-guardian", "yolov8n-pose"), "yolov8n-pose"
    )

    MODEL_FILE = ""
    MODEL_REPO = ""
    MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_REPO))

    model: YOLO
    pose_classifier: Optional[Any] = None
    fall_alerts: Optional[FallDetectionAlerts] = None
    device: str

    # Constructor
    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        LOGGER.error("=== NEW METHOD CALLED - CREATING YOLOV8 INSTANCE ===")
        LOGGER.error(f"Config received in new(): {config}")
        LOGGER.error(f"Dependencies received in new(): {dependencies}")
        try:
            result = super().new(config, dependencies)
            LOGGER.error("=== NEW METHOD COMPLETED SUCCESSFULLY ===")
            return result
        except Exception as e:
            LOGGER.error(f"=== NEW METHOD FAILED: {e} ===")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            raise

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ComponentConfig):
        LOGGER.error("=== VALIDATE_CONFIG METHOD CALLED ===")
        LOGGER.error(f"Validating config: {config}")
        LOGGER.error(f"Config attributes: {config.attributes}")
        try:
            attrs = struct_to_dict(config.attributes)
            LOGGER.error(f"Parsed attributes: {attrs}")
            model_location = attrs.get("model_location")
            if not model_location:
                LOGGER.error("ERROR: No model_location found in config")
                raise Exception("A model_location must be defined")
            LOGGER.error(f"Validation successful for model_location: {model_location}")
            return []
        except Exception as e:
            LOGGER.error(f"VALIDATION FAILED: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            raise e

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        LOGGER.error("=== RECONFIGURE METHOD CALLED ===")
        LOGGER.error(f"RAW CONFIG OBJECT TYPE: {type(config)}")
        LOGGER.error(f"RAW CONFIG.ATTRIBUTES TYPE: {type(config.attributes)}")
        LOGGER.error(f"RAW CONFIG.ATTRIBUTES: {config.attributes}")
        
        attrs = struct_to_dict(config.attributes)
        model_location = str(attrs.get("model_location"))
        pose_classifier_path = attrs.get("pose_classifier_path")

        LOGGER.error(f"FULL CONFIG ATTRIBUTES RECEIVED: {attrs}")
        LOGGER.error(f"POSE CLASSIFIER PATH RAW: {pose_classifier_path}")
        LOGGER.error(f"POSE CLASSIFIER PATH TYPE: {type(pose_classifier_path)}")
        LOGGER.error(f"POSE CLASSIFIER PATH STR: {str(pose_classifier_path)}")
        LOGGER.error(f"AVAILABLE KEYS: {list(attrs.keys())}")
        
        # Convert to string but preserve None detection
        if pose_classifier_path is not None:
            pose_classifier_path_str = str(pose_classifier_path)
            LOGGER.error(f"POSE CLASSIFIER PATH CONVERTED: {pose_classifier_path_str}")
        else:
            pose_classifier_path_str = None
            LOGGER.error("POSE CLASSIFIER PATH IS NONE")
        
        self.DEPS = dependencies
        self.task = str(attrs.get("task")) or None

        # Debug: Log available camera dependencies
        if dependencies:
            LOGGER.error(f"TOTAL DEPENDENCIES RECEIVED: {len(dependencies)}")
            for dep_name, dep_resource in dependencies.items():
                LOGGER.error(f"  - {dep_name} ({type(dep_resource)})")
            
            camera_names = self.get_available_camera_names()
            LOGGER.error(f"AVAILABLE CAMERA DEPENDENCIES: {camera_names}")
            if camera_names:
                LOGGER.error(f"PRIMARY CAMERA: {camera_names[0]}")
            else:
                LOGGER.error("⚠️ NO CAMERA DEPENDENCIES FOUND - check 'depends_on' in config")
        else:
            LOGGER.error("⚠️ NO DEPENDENCIES PROVIDED TO VISION SERVICE")

        # Load pose classifier if specified
        if pose_classifier_path_str and pose_classifier_path_str != "None":
            try:
                LOGGER.error(f"ATTEMPTING TO LOAD ML CLASSIFIER FROM: {pose_classifier_path_str}")
                import joblib
                classifier_path = os.path.abspath(pose_classifier_path_str)
                LOGGER.error(f"ABSOLUTE PATH: {classifier_path}")
                
                if os.path.exists(classifier_path):
                    LOGGER.error(f"FILE EXISTS - LOADING ML POSE CLASSIFIER")
                    try:
                        LOGGER.error(f"BEFORE JOBLIB.LOAD - ABOUT TO LOAD: {classifier_path}")
                        
                        # Check numpy version for compatibility warning
                        import numpy as np
                        LOGGER.error(f"Current numpy version: {np.__version__}")
                        
                        self.pose_classifier = joblib.load(classifier_path)
                        LOGGER.error(f"AFTER JOBLIB.LOAD - CHECKING RESULT")
                        if self.pose_classifier is not None:
                            LOGGER.error(f"SUCCESS - ML POSE CLASSIFIER LOADED!")
                            LOGGER.error(f"   Model type: {type(self.pose_classifier)}")
                            LOGGER.error(f"   Classes: {getattr(self.pose_classifier, 'classes_', 'Unknown')}")
                            LOGGER.error(f"   Feature count: {getattr(self.pose_classifier, 'n_features_in_', 'Unknown')}")
                        else:
                            LOGGER.error(f"JOBLIB.LOAD RETURNED NONE")
                    except ModuleNotFoundError as numpy_error:
                        if "numpy._core" in str(numpy_error):
                            LOGGER.error(f"NUMPY VERSION COMPATIBILITY ERROR: {numpy_error}")
                            LOGGER.error("The ML classifier was saved with a newer numpy version (2.x) but current environment has numpy 1.x")
                            LOGGER.error("SOLUTION: Update numpy with: pip install --upgrade numpy>=2.0")
                            LOGGER.error("Or retrain the classifier with current numpy version")
                        else:
                            LOGGER.error(f"MODULE NOT FOUND ERROR: {numpy_error}")
                        self.pose_classifier = None
                    except Exception as joblib_error:
                        LOGGER.error(f"JOBLIB.LOAD FAILED WITH EXCEPTION: {joblib_error}")
                        LOGGER.error(f"Exception type: {type(joblib_error)}")
                        import traceback
                        LOGGER.error(f"Full joblib traceback: {traceback.format_exc()}")
                        self.pose_classifier = None
                else:
                    LOGGER.error(f"FILE NOT FOUND: {classifier_path}")
                    # Try to list directory contents for debugging
                    try:
                        parent_dir = os.path.dirname(classifier_path)
                        if os.path.exists(parent_dir):
                            files = os.listdir(parent_dir)
                            LOGGER.error(f"Directory contents of {parent_dir}: {files}")
                        else:
                            LOGGER.error(f"Parent directory does not exist: {parent_dir}")
                    except Exception as list_err:
                        LOGGER.error(f"Could not list directory: {list_err}")
                    self.pose_classifier = None
            except ImportError:
                LOGGER.error("JOBLIB NOT INSTALLED - cannot load ML pose classifier")
                self.pose_classifier = None
            except Exception as e:
                LOGGER.error(f"FAILED TO LOAD POSE CLASSIFIER: {e}")
                import traceback
                LOGGER.error(f"Full traceback: {traceback.format_exc()}")
                self.pose_classifier = None
        else:
            LOGGER.error("NO POSE CLASSIFIER PATH SPECIFIED - ML classification disabled")
            self.pose_classifier = None

        # Initialize fall detection alerts if configured
        # Support both environment variables (secure) and config attributes
        alert_config = {}
        
        # Check if we should use environment variables (prioritize env vars)
        # Default to True if use_env_for_twilio is set or if TWILIO_ACCOUNT_SID env var exists
        use_env = attrs.get('use_env_for_twilio', False) or bool(os.environ.get('TWILIO_ACCOUNT_SID'))
        
        if use_env:
            LOGGER.error("🔒 LOADING TWILIO CREDENTIALS FROM ENVIRONMENT VARIABLES (secure method)")
            # Debug: Show what environment variables we can actually see (without exposing sensitive data)
            env_status = {
                'TWILIO_ACCOUNT_SID': 'SET' if os.environ.get('TWILIO_ACCOUNT_SID') else 'NOT_SET',
                'TWILIO_AUTH_TOKEN': 'SET' if os.environ.get('TWILIO_AUTH_TOKEN') else 'NOT_SET',
                'TWILIO_FROM_PHONE': 'SET' if os.environ.get('TWILIO_FROM_PHONE') else 'NOT_SET',
                'TWILIO_TO_PHONES': 'SET' if os.environ.get('TWILIO_TO_PHONES') else 'NOT_SET',
                'TWILIO_WEBHOOK_URL': 'SET' if os.environ.get('TWILIO_WEBHOOK_URL') else 'NOT_SET'
            }
            LOGGER.error(f"Environment variable status: {env_status}")
            
            # Load from environment (secure method via viam agent configuration)
            alert_config = {
                'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
                'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
                'twilio_from_phone': os.environ.get('TWILIO_FROM_PHONE'),
                'twilio_to_phones': os.environ.get('TWILIO_TO_PHONES', '').split(',') if os.environ.get('TWILIO_TO_PHONES') else [],
                'webhook_url': os.environ.get('TWILIO_WEBHOOK_URL'),
                # Non-sensitive settings can still come from config
                'fall_confidence_threshold': attrs.get('fall_confidence_threshold', 0.7),
                'alert_cooldown_seconds': attrs.get('alert_cooldown_seconds', 300)
            }
        else:
            LOGGER.error("⚠️ LOADING TWILIO CREDENTIALS FROM ROBOT CONFIGURATION (less secure)")
            # Load from config attributes (less secure - for backwards compatibility)
            alert_config = {
                'twilio_account_sid': attrs.get('twilio_account_sid'),
                'twilio_auth_token': attrs.get('twilio_auth_token'), 
                'twilio_from_phone': attrs.get('twilio_from_phone'),
                'twilio_to_phones': attrs.get('twilio_to_phones', []),
                'fall_confidence_threshold': attrs.get('fall_confidence_threshold', 0.7),
                'alert_cooldown_seconds': attrs.get('alert_cooldown_seconds', 300),
                'webhook_url': attrs.get('webhook_url')
            }
        
        # Check if all required Twilio settings are provided
        if all([alert_config['twilio_account_sid'], 
                alert_config['twilio_auth_token'], 
                alert_config['twilio_from_phone'],
                alert_config['twilio_to_phones']]):
            try:
                LOGGER.error("INITIALIZING FALL DETECTION ALERTS")
                self.fall_alerts = FallDetectionAlerts(alert_config)
                LOGGER.error("✅ FALL DETECTION ALERTS INITIALIZED SUCCESSFULLY")
            except Exception as e:
                LOGGER.error(f"❌ FAILED TO INITIALIZE FALL DETECTION ALERTS: {e}")
                self.fall_alerts = None
        else:
            LOGGER.error("FALL DETECTION ALERTS DISABLED - Missing Twilio configuration")
            LOGGER.error(f"Account SID present: {bool(alert_config['twilio_account_sid'])}")
            LOGGER.error(f"Auth Token present: {bool(alert_config['twilio_auth_token'])}")
            LOGGER.error(f"From Phone present: {bool(alert_config['twilio_from_phone'])}")
            LOGGER.error(f"To Phones present: {bool(alert_config['twilio_to_phones'])}")
            self.fall_alerts = None

        # Initialize data manager reference for doCommand data capture
        data_manager_name = attrs.get("data_manager")
        if data_manager_name:
            LOGGER.error(f"🔄 Setting up data manager reference: {data_manager_name}")
            self.data_manager_name = data_manager_name
            # Note: The actual data manager service will be accessed via the robot's resource manager
            LOGGER.error("✅ Data manager reference configured for fall detection data capture")
        else:
            LOGGER.error("⚠️ No data_manager specified in config - doCommand data capture may not work")
            self.data_manager_name = None

        if "/" in model_location:
            if self.is_path(model_location):
                self.MODEL_PATH = model_location
            else:
                model_name = str(attrs.get("model_name", ""))
                if model_name == "":
                    raise Exception(
                        "model_name attribute is required for downloading models from HuggingFace."
                    )
                self.MODEL_REPO = model_location
                self.MODEL_FILE = model_name
                self.MODEL_PATH = os.path.abspath(
                    os.path.join(
                        MODEL_DIR,
                        f"{self.MODEL_REPO.replace('/', '_')}_{self.MODEL_FILE}",
                    )
                )

                self.get_model()

            self.model = YOLO(self.MODEL_PATH, task=self.task)
        else:
            self.model = YOLO(model_location, task=self.task)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        return

    def get_available_camera_names(self) -> List[str]:
        """Get list of available camera names from dependencies."""
        camera_names = []
        if hasattr(self, 'DEPS') and self.DEPS:
            try:
                for resource_name in self.DEPS.keys():
                    # Check if this is a camera resource
                    if hasattr(resource_name, 'resource_type') and hasattr(resource_name.resource_type, 'resource_type'):
                        if resource_name.resource_type.resource_type == "camera":
                            camera_names.append(resource_name.name)
                    # Fallback: check resource name string
                    elif "camera" in str(resource_name).lower():
                        # Extract name from resource string
                        name_part = str(resource_name).split('/')[-1] if '/' in str(resource_name) else str(resource_name)
                        camera_names.append(name_part)
                        
                LOGGER.info(f"Found camera dependencies: {camera_names}")
            except Exception as e:
                LOGGER.error(f"Error getting camera names from dependencies: {e}")
                # Fallback: try to extract from DEPS keys as strings
                try:
                    for key in self.DEPS.keys():
                        key_str = str(key)
                        if "camera" in key_str.lower():
                            # Try to extract just the name part
                            if '/' in key_str:
                                name = key_str.split('/')[-1]
                            else:
                                name = key_str
                            camera_names.append(name)
                    LOGGER.info(f"Fallback camera names: {camera_names}")
                except Exception as fallback_error:
                    LOGGER.error(f"Fallback camera name extraction failed: {fallback_error}")
        return camera_names

    def get_primary_camera_name(self) -> str:
        """Get the primary camera name from dependencies."""
        camera_names = self.get_available_camera_names()
        if camera_names:
            # Log all available cameras for debugging
            LOGGER.error(f"Available cameras in dependencies: {camera_names}")
            return camera_names[0]  # Return first available camera
        return "unknown_camera"

    def get_camera_name_from_image_request(self, image: ViamImage) -> str:
        """Try to determine camera name from image metadata or context."""
        # This is a placeholder - in practice, we'd need to track which camera
        # the image came from through the call chain
        return "unknown_camera"

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return cam_image

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name))

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        detections = []
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
        
        if len(results) >= 1:
            result = results[0]
            
            # Debug logging
            LOGGER.info(f"Model result attributes: {dir(result)}")
            LOGGER.info(f"Has boxes: {hasattr(result, 'boxes')}")
            LOGGER.info(f"Has keypoints: {hasattr(result, 'keypoints')}")
            if hasattr(result, 'boxes') and result.boxes is not None:
                LOGGER.info(f"Number of boxes: {len(result.boxes)}")
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                LOGGER.info(f"Keypoints shape: {result.keypoints.xy.shape if result.keypoints.xy is not None else 'None'}")
            
            # Handle pose detection with both bounding boxes and keypoints
            if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result, 'keypoints') and result.keypoints is not None:
                LOGGER.info(f"Found {len(result.boxes)} people with keypoints")
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i]
                    confidence = result.boxes.conf[i].item()
                    class_id = int(result.boxes.cls[i].item())
                    class_name = result.names[class_id]
                    
                    # Extract keypoints for logging
                    keypoints = result.keypoints.xy[i].cpu().numpy()  # Shape: (17, 2) for COCO pose
                    visible_keypoints = sum(1 for x, y in keypoints if x > 0 and y > 0)
                    LOGGER.info(f"Person {i}: {visible_keypoints}/17 keypoints visible")
                    
                    # Create standard Detection object (no keypoints field)
                    detection = Detection(
                        x_min=int(box[0].item()),
                        y_min=int(box[1].item()),
                        x_max=int(box[2].item()),
                        y_max=int(box[3].item()),
                        confidence=confidence,
                        class_name=class_name
                    )
                    detections.append(detection)
                    
            # Fallback to regular bounding box detection if no keypoints
            elif hasattr(result, 'boxes') and result.boxes is not None:
                LOGGER.info(f"Found {len(result.boxes)} detections (no keypoints)")
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i]
                    confidence = result.boxes.conf[i].item()
                    class_id = int(result.boxes.cls[i].item())
                    class_name = result.names[class_id]
                    
                    detection = Detection(
                        x_min=int(box[0].item()),
                        y_min=int(box[1].item()),
                        x_max=int(box[2].item()),
                        y_max=int(box[3].item()),
                        confidence=confidence,
                        class_name=class_name
                    )
                    detections.append(detection)

        return detections

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        # Pass camera name in extra for fall alert context
        extra_dict = dict(extra) if extra else {}
        extra_dict["camera_name"] = camera_name
        return await self.get_classifications(await self.get_cam_image(camera_name), count, extra=extra_dict)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int = 0,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        
        # Get pose classifications instead of trying to extract from YOLO classification output
        keypoints = await self.extract_keypoints(image)
        pose_classifications = await self.classify_poses(keypoints)
        
        # Convert pose classifications to Viam Classification format AND check for fall alerts
        for pose_class in pose_classifications:
            if pose_class["pose_class"] != "unknown":
                classification = Classification(
                    class_name=pose_class["pose_class"],
                    confidence=pose_class["confidence"]
                )
                classifications.append(classification)
                
                # 🚨 TRIGGER FALL DETECTION ALERT IF NEEDED 🚨
                if pose_class.get("_trigger_fall_alert") and self.fall_alerts:
                    # Get camera name with improved fallback logic
                    camera_name = "unknown_camera"
                    
                    # Priority 1: Use camera_name from extra (most reliable)
                    if extra and "camera_name" in extra:
                        camera_name = extra["camera_name"]
                        LOGGER.error(f"✅ Using camera name from extra: {camera_name}")
                    else:
                        # Priority 2: Try to get from dependencies
                        available_cameras = self.get_available_camera_names()
                        if available_cameras:
                            if len(available_cameras) == 1:
                                # Only one camera - safe to use it
                                camera_name = available_cameras[0]
                                LOGGER.error(f"✅ Using single camera from dependencies: {camera_name}")
                            else:
                                # Multiple cameras - this is the problem case
                                LOGGER.error(f"⚠️ Multiple cameras in dependencies: {available_cameras}")
                                LOGGER.error(f"⚠️ Cannot determine which camera detected the fall!")
                                LOGGER.error(f"⚠️ Using first camera as fallback: {available_cameras[0]}")
                                LOGGER.error(f"🔧 SOLUTION: Use separate vision services for each camera")
                                camera_name = available_cameras[0]
                        else:
                            LOGGER.error(f"❌ No cameras found in dependencies")
                    
                    person_id = pose_class["person_id"]
                    confidence = pose_class["confidence"]
                    alert_metadata = pose_class.get("_alert_metadata", {})
                    
                    LOGGER.error(f"🚨 TRIGGERING FALL ALERT for person {person_id} on camera {camera_name}")
                    
                    # Send fall alert asynchronously (don't block the vision service)
                    try:
                        import asyncio
                        
                        # Get data manager instance if available
                        data_manager = None
                        if hasattr(self, 'data_manager_name') and self.data_manager_name:
                            try:
                                # Access data manager from robot dependencies if available
                                # Note: This is a simplified approach - in production you'd access via robot.resource_manager
                                LOGGER.error(f"🔄 Attempting to access data manager: {self.data_manager_name}")
                                data_manager = None  # Will use file fallback for now
                            except Exception as dm_error:
                                LOGGER.error(f"⚠️ Could not access data manager {self.data_manager_name}: {dm_error}")
                                data_manager = None
                        
                        asyncio.create_task(
                            self.fall_alerts.send_fall_alert(
                                camera_name=camera_name,
                                person_id=person_id,
                                confidence=confidence,
                                image=image,
                                metadata=alert_metadata,
                                data_manager=data_manager,
                                vision_service=self
                            )
                        )
                        LOGGER.error("✅ Fall alert task created successfully")
                    except Exception as alert_error:
                        LOGGER.error(f"❌ Failed to create fall alert task: {alert_error}")
        
        return classifications

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        return []

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None
    ) -> Mapping[str, ValueTypes]:
        """Handle custom commands like getting keypoints or pose analysis."""
        cmd_name = command.get("command")
        
        if cmd_name == "get_keypoints":
            # Get keypoints from camera
            camera_name = command.get("camera_name", "")
            if not camera_name:
                camera_name = self.get_primary_camera_name()
            if camera_name and camera_name != "unknown_camera":
                image = await self.get_cam_image(camera_name)
                keypoints = await self.extract_keypoints(image)
                return {"keypoints": keypoints, "camera_name": camera_name}
            else:
                available_cameras = self.get_available_camera_names()
                return {"error": "camera_name required or no cameras available", "available_cameras": available_cameras}
                
        elif cmd_name == "get_pose_analysis":
            # Get pose analysis (angles, distances, etc.)
            camera_name = command.get("camera_name", "")
            if not camera_name:
                camera_name = self.get_primary_camera_name()
            if camera_name and camera_name != "unknown_camera":
                image = await self.get_cam_image(camera_name)
                analysis = await self.analyze_pose(image)
                return {"pose_analysis": analysis, "camera_name": camera_name}
            else:
                available_cameras = self.get_available_camera_names()
                return {"error": "camera_name required or no cameras available", "available_cameras": available_cameras}
                
        elif cmd_name == "get_pose_classifications":
            # Get detections + keypoints + pose classifications in one call
            camera_name = command.get("camera_name", "")
            if not camera_name:
                camera_name = self.get_primary_camera_name()
            if camera_name and camera_name != "unknown_camera":
                image = await self.get_cam_image(camera_name)
                
                # Get both detections and keypoints
                detections = await self.get_detections(image)
                keypoints = await self.extract_keypoints(image)
                
                # Classify poses
                pose_classifications = await self.classify_poses(keypoints)
                
                return {
                    "detections": [
                        {
                            "x_min": det.x_min,
                            "y_min": det.y_min, 
                            "x_max": det.x_max,
                            "y_max": det.y_max,
                            "confidence": det.confidence,
                            "class_name": det.class_name
                        } for det in detections
                    ],
                    "keypoints": keypoints,
                    "pose_classifications": pose_classifications,
                    "camera_name": camera_name
                }
            else:
                available_cameras = self.get_available_camera_names()
                return {"error": "camera_name required or no cameras available", "available_cameras": available_cameras}
        
        elif cmd_name == "list_cameras":
            # List available cameras
            available_cameras = self.get_available_camera_names()
            return {"available_cameras": available_cameras, "primary_camera": self.get_primary_camera_name()}
        
        return {"supported_commands": ["get_keypoints", "get_pose_analysis", "get_pose_classifications", "list_cameras"]}

    async def extract_keypoints(self, image: ViamImage) -> List[dict]:
        """Extract only keypoints from detected persons."""
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
        
        all_keypoints = []
        if len(results) >= 1:
            result = results[0]
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]
                
                for person_idx in range(len(result.keypoints)):
                    keypoints = result.keypoints.xy[person_idx].cpu().numpy()
                    keypoint_conf = result.keypoints.conf[person_idx].cpu().numpy() if result.keypoints.conf is not None else None
                    
                    person_keypoints = {
                        "person_id": person_idx,
                        "keypoints": []
                    }
                    
                    for j, (x, y) in enumerate(keypoints):
                        keypoint_data = {
                            "name": keypoint_names[j] if j < len(keypoint_names) else f"keypoint_{j}",
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(keypoint_conf[j]) if keypoint_conf is not None else 1.0,
                            "visible": float(x) > 0 and float(y) > 0
                        }
                        person_keypoints["keypoints"].append(keypoint_data)
                    
                    all_keypoints.append(person_keypoints)
        
        return all_keypoints

    async def analyze_pose(self, image: ViamImage) -> List[dict]:
        """Analyze pose for basic metrics like angles and distances."""
        keypoints_data = await self.extract_keypoints(image)
        
        analyses = []
        for person_data in keypoints_data:
            keypoints = {kp["name"]: (kp["x"], kp["y"]) for kp in person_data["keypoints"] if kp["visible"]}
            
            analysis = {
                "person_id": person_data["person_id"],
                "pose_metrics": {}
            }
            
            # Calculate arm angles if shoulder, elbow, wrist are visible
            if all(joint in keypoints for joint in ["left_shoulder", "left_elbow", "left_wrist"]):
                left_arm_angle = self.calculate_angle(
                    keypoints["left_shoulder"], 
                    keypoints["left_elbow"], 
                    keypoints["left_wrist"]
                )
                analysis["pose_metrics"]["left_arm_angle"] = left_arm_angle
            
            if all(joint in keypoints for joint in ["right_shoulder", "right_elbow", "right_wrist"]):
                right_arm_angle = self.calculate_angle(
                    keypoints["right_shoulder"], 
                    keypoints["right_elbow"], 
                    keypoints["right_wrist"]
                )
                analysis["pose_metrics"]["right_arm_angle"] = right_arm_angle
            
            # Calculate body posture
            if all(joint in keypoints for joint in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                shoulder_width = abs(keypoints["left_shoulder"][0] - keypoints["right_shoulder"][0])
                hip_width = abs(keypoints["left_hip"][0] - keypoints["right_hip"][0])
                analysis["pose_metrics"]["shoulder_width"] = float(shoulder_width)
                analysis["pose_metrics"]["hip_width"] = float(hip_width)
            
            analyses.append(analysis)
        
        return analyses

    async def classify_poses(self, keypoints_data: List[dict]) -> List[dict]:
        """Classify poses as standing, sitting, crouching, or fallen based on keypoints."""
        classifications = []
        
        for person_data in keypoints_data:
            keypoints = {kp["name"]: (kp["x"], kp["y"]) for kp in person_data["keypoints"] if kp["visible"]}
            
            classification = {
                "person_id": person_data["person_id"],
                "pose_class": "unknown",
                "confidence": 0.0,
                "reasoning": [],
                "method": "ml" if self.pose_classifier else "rules_based"
            }
            
            # Check if we have enough keypoints for classification
            required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            missing_points = [point for point in required_points if point not in keypoints]
            
            if missing_points:
                classification["reasoning"].append(f"Missing keypoints: {missing_points}")
                classifications.append(classification)
                continue
            
            # Use ML classifier if available - NO FALLBACK TO RULES
            if self.pose_classifier:
                try:
                    features = self.extract_pose_features_for_ml(keypoints)
                    if features is not None:
                        import numpy as np
                        features_array = np.array(features).reshape(1, -1)
                        prediction = self.pose_classifier.predict(features_array)[0]
                        probabilities = self.pose_classifier.predict_proba(features_array)[0]
                        
                        # Get class names from classifier
                        class_names = self.pose_classifier.classes_
                        confidence = max(probabilities)
                        
                        classification["pose_class"] = prediction
                        classification["confidence"] = float(confidence)
                        classification["reasoning"].append(f"ML prediction with {len(features)} features")
                        
                        # Add probability distribution for debugging
                        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                        classification["probabilities"] = prob_dict
                        
                        # Add feature values for debugging
                        feature_names = [
                            'head_to_shoulder', 'shoulder_to_hip', 'head_to_hip',
                            'shoulder_width', 'hip_width', 'hip_to_knee', 'knee_y',
                            'knee_to_ankle', 'ankle_y', 'aspect_ratio', 
                            'norm_head_y', 'norm_shoulder_y', 'norm_hip_y'
                        ]
                        feature_debug = {feature_names[i]: float(features[i]) for i in range(len(features))}
                        classification["features"] = feature_debug
                        
                        LOGGER.info(f"ML Classification for person {person_data['person_id']}: {prediction} ({confidence:.3f})")
                        LOGGER.info(f"Probabilities: {prob_dict}")
                        LOGGER.info(f"Features used: {feature_debug}")
                        
                        # 🚨 FALL DETECTION TRIGGER 🚨
                        if prediction == "fallen" and self.fall_alerts:
                            # Trigger fall detection alert
                            LOGGER.error(f"🚨 FALL DETECTED for person {person_data['person_id']} with confidence {confidence:.3f}")
                            
                            # We need the original image for the alert - store it for later use
                            classification["_trigger_fall_alert"] = True
                            classification["_alert_metadata"] = {
                                "probabilities": prob_dict,
                                "features": feature_debug,
                                "confidence": confidence
                            }
                        
                        classifications.append(classification)
                        continue
                    else:
                        classification["reasoning"].append("Failed to extract ML features - insufficient keypoints")
                        classification["pose_class"] = "unknown"
                        classification["confidence"] = 0.0
                        LOGGER.warning(f"Could not extract features for person {person_data['person_id']}")
                        classifications.append(classification)
                        continue
                except Exception as e:
                    LOGGER.error(f"ML classification failed: {e}")
                    classification["reasoning"].append(f"ML classification error: {e}")
                    classification["pose_class"] = "error"
                    classification["confidence"] = 0.0
                    classifications.append(classification)
                    continue
            
            # NO ML CLASSIFIER AVAILABLE
            classification["method"] = "no_classifier"
            classification["pose_class"] = "unknown"
            classification["confidence"] = 0.0
            classification["reasoning"].append("No ML classifier loaded - rules-based disabled")
            LOGGER.warning("No ML classifier available and rules-based classification disabled")
            classifications.append(classification)
        
        return classifications

    def extract_pose_features_for_ml(self, keypoints: dict) -> Optional[List[float]]:
        """Extract features for ML classifier (matches training format)."""
        try:
            features = []
            
            # Check required points
            required_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            if not all(point in keypoints for point in required_points):
                return None
            
            # Basic body measurements
            head_y = keypoints["nose"][1]
            shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
            hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
            
            # Vertical distances (key for pose classification)
            head_to_shoulder = head_y - shoulder_y
            shoulder_to_hip = shoulder_y - hip_y
            head_to_hip = head_y - hip_y
            
            # Body proportions
            shoulder_width = abs(keypoints["left_shoulder"][0] - keypoints["right_shoulder"][0])
            hip_width = abs(keypoints["left_hip"][0] - keypoints["right_hip"][0])
            
            features.extend([
                head_to_shoulder,    # Negative if head above shoulders (normal)
                shoulder_to_hip,     # Negative if shoulders above hips (normal)
                head_to_hip,         # Overall body height
                shoulder_width,      # Body width at shoulders
                hip_width,           # Body width at hips
            ])
            
            # Knee positions if available
            if "left_knee" in keypoints and "right_knee" in keypoints:
                knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
                hip_to_knee = hip_y - knee_y
                features.extend([
                    hip_to_knee,     # Negative if hips above knees (normal standing)
                    knee_y           # Absolute knee position
                ])
            else:
                features.extend([0, 0])
                
            # Ankle positions if available
            if "left_ankle" in keypoints and "right_ankle" in keypoints:
                ankle_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
                if "left_knee" in keypoints and "right_knee" in keypoints:
                    knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
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
            
            return features
            
        except Exception as e:
            LOGGER.error(f"Feature extraction failed: {e}")
            return None

    async def classify_pose_rules_based(self, keypoints: dict, classification: dict) -> dict:
        """Rules-based pose classification (fallback method)."""
        # Calculate body metrics
        head_y = keypoints["nose"][1]
        shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
        hip_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
        
        # Debug logging for pose classification
        LOGGER.info(f"Person {classification['person_id']} pose metrics:")
        LOGGER.info(f"  head_y: {head_y}, shoulder_y: {shoulder_y}, hip_y: {hip_y}")
        
        # Get knee and ankle positions if available
        knee_y = None
        ankle_y = None
        
        if "left_knee" in keypoints and "right_knee" in keypoints:
            knee_y = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
            LOGGER.info(f"  knee_y: {knee_y}")
        if "left_ankle" in keypoints and "right_ankle" in keypoints:
            ankle_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2
            LOGGER.info(f"  ankle_y: {ankle_y}")
        
        # Calculate body orientation (vertical distance ratios)
        head_to_shoulder = abs(head_y - shoulder_y)
        shoulder_to_hip = abs(shoulder_y - hip_y)
        
        LOGGER.info(f"  head_to_shoulder: {head_to_shoulder}, shoulder_to_hip: {shoulder_to_hip}")
        
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
        
        # Debug logging for scores
        LOGGER.info(f"  Classification scores: {scores}")
        LOGGER.info(f"  Reasoning so far: {classification['reasoning']}")
        
        max_score = max(scores.values())
        if max_score > 0:
            pose_class = max(scores.keys(), key=lambda k: scores[k])
            classification["pose_class"] = pose_class
            classification["confidence"] = min(max_score / 5.0, 1.0)  # Normalize to 0-1
            LOGGER.info(f"  Final classification: {pose_class} (confidence: {classification['confidence']})")
        else:
            classification["pose_class"] = "unknown"
            classification["confidence"] = 0.0
            classification["reasoning"].append("Insufficient evidence for any pose class")
            LOGGER.info(f"  Final classification: unknown")
        
        # Add raw metrics for debugging
        classification["metrics"] = {
            "head_y": float(head_y),
            "shoulder_y": float(shoulder_y), 
            "hip_y": float(hip_y),
            "knee_y": float(knee_y) if knee_y else None,
            "ankle_y": float(ankle_y) if ankle_y else None,
            "scores": scores
        }
        
        return classification

    def calculate_angle(self, point1, point2, point3):
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

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        result.detections = await self.get_detections(result.image)
        result.classifications = await self.get_classifications(result.image, 1)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    def is_path(self, path: str) -> bool:
        try:
            Path(path)
            return os.path.exists(path)
        except ValueError:
            return False

    def get_model(self):
        if not os.path.exists(self.MODEL_PATH):
            MODEL_URL = f"https://huggingface.co/{self.MODEL_REPO}/resolve/main/{self.MODEL_FILE}"
            LOGGER.debug(f"Fetching model {self.MODEL_FILE} from {MODEL_URL}")
            urlretrieve(MODEL_URL, self.MODEL_PATH, self.log_progress)

    def log_progress(self, count: int, block_size: int, total_size: int) -> None:
        percent = count * block_size * 100 // total_size
        LOGGER.debug(f"\rDownloading {self.MODEL_FILE}: {percent}%")


# vendored and updated from ultralyticsplus library
def postprocess_classify_output(model: YOLO, result: Results) -> dict:
    """
    Postprocesses the output of classification models

    Args:
        model (YOLO): YOLO model
        prob (np.ndarray): output of the model

    Returns:
        dict: dictionary of outputs with labels
    """
    output = {}
    if isinstance(model.names, list):
        names = model.names
    elif isinstance(model.names, dict):
        names = model.names.values()
    else:
        raise ValueError("Model names must be either a list or a dict")

    if result.probs:
        for i, label in enumerate(names):
            output[label] = result.probs[i].item()
        return output
    else:
        return {}