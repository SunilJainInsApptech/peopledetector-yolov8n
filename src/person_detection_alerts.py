"""
Person Detection Alert Service using Twilio
Sends SMS alerts when a person is detected during nighttime hours (11:00 PM - 6:00 AM)
"""

import os
import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Dict, Any
import base64
import tempfile
from twilio.rest import Client
from viam.media.video import ViamImage

# Try to import Viam DataManager service
try:
    from viam.services.data_manager import DataManager
    VIAM_DATA_AVAILABLE = True
except ImportError:
    VIAM_DATA_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

class PersonDetectionAlerts:
    """Service for sending person detection alerts via Twilio during nighttime hours"""
    
    def __init__(self, config: dict):
        """Initialize Twilio client and alert configuration"""
        # Try to load from environment variables first, then config
        self.account_sid = (
            os.environ.get('TWILIO_ACCOUNT_SID') or 
            config.get('twilio_account_sid')
        )
        self.auth_token = (
            os.environ.get('TWILIO_AUTH_TOKEN') or 
            config.get('twilio_auth_token')
        )
        self.from_phone = (
            os.environ.get('TWILIO_FROM_PHONE') or 
            config.get('twilio_from_phone')
        )
        
        # Handle phone numbers from environment (comma-separated) or config (list)
        env_phones = os.environ.get('TWILIO_TO_PHONES')
        if env_phones:
            self.to_phones = [phone.strip() for phone in env_phones.split(',')]
        else:
            self.to_phones = config.get('twilio_to_phones', [])
        
        self.webhook_url = (
            os.environ.get('TWILIO_WEBHOOK_URL') or 
            config.get('webhook_url')
        )
        
        # Alert settings
        self.min_confidence = config.get('detection_confidence_threshold', 0.5)
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alert_time = {}  # Track last alert time per camera
        
        # Nighttime hours (11:00 PM to 6:00 AM)
        self.night_start = time(23, 0)  # 11:00 PM
        self.night_end = time(6, 0)     # 6:00 AM
        
        # Push notification settings
        self.push_notification_url = (
            os.environ.get('RIGGUARDIAN_WEBHOOK_URL') or
            config.get('rigguardian_webhook_url', 'https://building-sensor-platform-production.up.railway.app/webhook/person-alert')
        )
        
        # Log what source we're using (without exposing credentials)
        if os.environ.get('TWILIO_ACCOUNT_SID'):
            LOGGER.info("✅ Using Twilio credentials from environment variables")
        else:
            LOGGER.info("⚠️ Using Twilio credentials from robot configuration")
        
        # Validate required config
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            raise ValueError("Missing required Twilio configuration: account_sid, auth_token, from_phone")
        
        if not self.to_phones:
            raise ValueError("No alert phone numbers configured")
        
        # Initialize Twilio client
        try:
            self.client = Client(self.account_sid, self.auth_token)
            LOGGER.info("✅ Twilio client initialized successfully")
        except Exception as e:
            LOGGER.error(f"❌ Failed to initialize Twilio client: {e}")
            raise
    
    def is_nighttime(self) -> bool:
        """Check if current time is within nighttime hours (11:00 PM - 6:00 AM)"""
        current_time = datetime.now().time()
        
        # Handle the case where night spans midnight
        if self.night_start > self.night_end:  # e.g., 23:00 to 06:00
            return current_time >= self.night_start or current_time <= self.night_end
        else:  # e.g., 22:00 to 05:00 (doesn't span midnight)
            return self.night_start <= current_time <= self.night_end
    
    def should_send_alert(self, camera_name: str, confidence: float) -> bool:
        """Check if we should send an alert based on time, confidence and cooldown"""
        # Check if it's nighttime
        if not self.is_nighttime():
            current_time = datetime.now().strftime("%H:%M")
            LOGGER.debug(f"Not nighttime ({current_time}) - no alert needed")
            return False
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            LOGGER.debug(f"Detection confidence {confidence:.3f} below threshold {self.min_confidence}")
            return False
        
        # Check cooldown period per camera
        now = datetime.now()
        if camera_name in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[camera_name]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                LOGGER.debug(f"Alert cooldown active for {camera_name} ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                return False
        
        return True
    
    async def save_image_locally(self, image: ViamImage, camera_name: str) -> str:
        """Save image to local temporary file and return path"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"person_detection_{camera_name}_{timestamp}.jpg"
            
            # Save to temporary directory
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, filename)
            
            # Convert ViamImage to bytes and save
            with open(image_path, 'wb') as f:
                f.write(image.data)
            
            LOGGER.info(f"📸 Person detection image saved: {image_path}")
            return image_path
            
        except Exception as e:
            LOGGER.error(f"❌ Failed to save image: {e}")
            return ""
    
    def format_alert_message(self, 
                           camera_name: str, 
                           person_count: int,
                           confidence: float,
                           timestamp: datetime,
                           image_path: str = "") -> str:
        """Format the alert message for SMS"""
        
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 PERSON DETECTED 🚨\n"
        message += f"Camera: {camera_name}\n"
        message += f"People Count: {person_count}\n"
        message += f"Detection Confidence: {confidence:.1%}\n"
        message += f"Time: {timestamp_str}\n"
        
        if image_path:
            message += f"Image: {os.path.basename(image_path)}\n"
        
        message += "\nNighttime security alert - please verify."
        
        return message
    
    async def send_person_alert(self, 
                              camera_name: str,
                              person_count: int,
                              confidence: float,
                              image: ViamImage,
                              data_manager=None,
                              vision_service=None) -> bool:
        """Send person detection alert via Twilio SMS during nighttime hours"""
        
        try:
            # Check if we should send alert
            if not self.should_send_alert(camera_name, confidence):
                return False
            
            # Record alert time
            timestamp = datetime.now()
            self.last_alert_time[camera_name] = timestamp
            
            LOGGER.info(f"🌙 Nighttime person detection on {camera_name}: {person_count} people detected")
            
            # Save image to Viam-monitored directory for automatic sync
            await self.save_detection_image(camera_name, person_count, confidence, image, data_manager, vision_service)
            
            # Save image locally for SMS reference
            image_path = await self.save_image_locally(image, camera_name)
            
            # Format alert message
            message = self.format_alert_message(
                camera_name=camera_name,
                person_count=person_count,
                confidence=confidence,
                timestamp=timestamp,
                image_path=image_path
            )
            
            # Send SMS to all configured phone numbers
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    # Send SMS
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Person alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send SMS to {phone_number}: {e}")
            
            # Send push notification to rigguardian.com app
            push_success = await self.send_webhook_notification(
                camera_name=camera_name,
                person_count=person_count,
                confidence=confidence,
                timestamp=timestamp,
                image=image
            )
            
            if push_success:
                LOGGER.info("📱 Push notification sent to rigguardian.com successfully")
            else:
                LOGGER.warning("⚠️ Push notification failed - SMS alert still sent")
            
            if success_count > 0:
                LOGGER.info(f"✅ Person alert sent successfully to {success_count}/{len(self.to_phones)} recipients")
                return True
            else:
                LOGGER.error("❌ Failed to send person alert to any recipients")
                return False
                
        except Exception as e:
            LOGGER.error(f"❌ Error sending person alert: {e}")
            return False
    
    async def send_test_alert(self, camera_name: str = "test_camera") -> bool:
        """Send a test alert to verify Twilio configuration"""
        try:
            message = f"🧪 TEST ALERT 🧪\n"
            message += f"Person detection system is active\n"
            message += f"Camera: {camera_name}\n"
            message += f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
            message += f"Nighttime monitoring: {self.night_start.strftime('%I:%M %p')} - {self.night_end.strftime('%I:%M %p')}"
            
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Test alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send test SMS to {phone_number}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            LOGGER.error(f"❌ Error sending test alert: {e}")
            return False
    
    async def send_webhook_notification(self, camera_name: str, person_count: int, confidence: float, timestamp: datetime, image: Optional[ViamImage] = None) -> bool:
        """Send notification via webhook to Railway server"""
        try:
            import aiohttp
            import json
            import base64
            
            # Create webhook payload
            webhook_data = {
                "alert_type": "person_detection",
                "camera_name": camera_name,
                "location": f"Camera {camera_name}",
                "person_count": person_count,
                "confidence": confidence,
                "severity": "warning",
                "title": "Nighttime Person Detection",
                "message": f"Person detected on {camera_name} during nighttime hours",
                "requires_immediate_attention": False,
                "notification_type": "person_detection",
                "timestamp": timestamp.isoformat(),
                "metadata": {
                    "detection_time": timestamp.strftime("%H:%M:%S"),
                    "is_nighttime": True,
                    "person_count": person_count
                },
                "actions": [
                    {"action": "view_camera", "title": "View Camera"},
                    {"action": "acknowledge", "title": "Acknowledge"}
                ]
            }
            
            # Add image data if available
            if image:
                try:
                    # Convert image to base64
                    image_b64 = base64.b64encode(image.data).decode('utf-8')
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    
                    webhook_data["image"] = image_b64
                    webhook_data["image_filename"] = f"person_alert_{camera_name}_{timestamp_str}.jpg"
                    
                    LOGGER.info(f"📷 Added image data to webhook ({len(image.data)} bytes)")
                except Exception as e:
                    LOGGER.warning(f"⚠️ Failed to encode image for webhook: {e}")
            
            LOGGER.info(f"🔄 Sending webhook to Railway server")
            LOGGER.info(f"📊 Person alert: {camera_name} at {timestamp.strftime('%H:%M:%S')} ({person_count} people, {confidence:.1%} confidence)")
            
            # Send to Railway server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.push_notification_url,
                    json=webhook_data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'PersonDetectionSystem/1.0',
                        'X-Alert-Type': 'person_detection',
                        'X-Sensor-Type': 'person_detection',
                        'Accept': 'application/json'
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_text = await response.text()
                    
                    LOGGER.info(f"📡 Railway server response: {response.status}")
                    
                    if response.status == 200:
                        LOGGER.info("✅ Webhook sent successfully to Railway server")
                        return True
                    else:
                        LOGGER.error(f"❌ Railway server webhook failed with status {response.status}")
                        LOGGER.error(f"📄 Response: {response_text}")
                        return False
                        
        except ImportError:
            LOGGER.error("❌ aiohttp not installed - install with: pip install aiohttp")
            return False
        except aiohttp.ClientTimeout:
            LOGGER.error("❌ Webhook request to Railway server timed out")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Railway server webhook error: {e}")
            return False
    
    async def save_detection_image(self, camera_name: str, person_count: int, confidence: float, image: ViamImage, data_manager=None, vision_service=None):
        """Save person detection image using data manager camera capture with PersonDetection tag"""
        try:
            LOGGER.info(f"🔄 Triggering person detection image capture for camera: {camera_name}")
            LOGGER.info(f"📊 Image size: {len(image.data)} bytes, People: {person_count}, Confidence: {confidence:.3f}")
            
            # Use data manager to capture from camera with PersonDetection tag
            if data_manager and vision_service:
                try:
                    LOGGER.info(f"🏷️ Triggering data manager capture with PersonDetection tag for camera: {camera_name}")
                    
                    # Use data manager's capture functionality with tags
                    capture_result = await data_manager.capture(
                        component_name=camera_name,
                        method_name="ReadImage",
                        tags=["PersonDetection", "Nighttime"],
                        additional_metadata={
                            "person_count": person_count,
                            "confidence": f"{confidence:.3f}",
                            "event_type": "person_detected",
                            "detection_time": datetime.now().strftime("%H:%M:%S"),
                            "is_nighttime": True,
                            "vision_service": "yolov8n-pose"
                        }
                    )
                    
                    LOGGER.info(f"✅ Data manager capture completed: {capture_result}")
                    LOGGER.info(f"🎯 Component: {camera_name}, Tags: PersonDetection, Nighttime")
                    
                    return {"status": "success", "method": "data_manager_capture", "result": capture_result}
                    
                except Exception as dm_error:
                    LOGGER.error(f"❌ Data manager capture failed: {dm_error}")
                    LOGGER.info("🔄 Falling back to file-based method")
                    
                    # Fallback: Save to the data manager's capture directory
                    return await self._save_detection_image_to_file(camera_name, person_count, confidence, image)
                    
            else:
                LOGGER.warning("⚠️ No data manager or vision service provided - using file-based fallback")
                return await self._save_detection_image_to_file(camera_name, person_count, confidence, image)
                
        except Exception as e:
            LOGGER.error(f"❌ Error in save_detection_image: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return {"status": "error", "method": "save_detection_image", "error": str(e)}
    
    async def _save_detection_image_to_file(self, camera_name: str, person_count: int, confidence: float, image: ViamImage):
        """Fallback method to save image directly to data manager's capture directory"""
        try:
            from datetime import datetime
            import os
            
            # Use the data manager's capture directory
            capture_dir = "/home/sunil/Documents/viam_captured_images"
            timestamp = datetime.utcnow()
            
            # Create filename with proper Viam naming convention for data manager to recognize
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            filename = f"{timestamp_str}_{camera_name}_ReadImage.jpg"
            filepath = os.path.join(capture_dir, filename)
            
            # Ensure directory exists
            os.makedirs(capture_dir, exist_ok=True)
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image.data)
            
            # Create metadata file with PersonDetection tag for data manager to process
            metadata_filename = f"{timestamp_str}_{camera_name}_ReadImage.json"
            metadata_filepath = os.path.join(capture_dir, metadata_filename)
            
            import json
            metadata_content = {
                "component_name": camera_name,
                "method_name": "ReadImage",
                "tags": ["PersonDetection", "Nighttime"],
                "timestamp": timestamp.isoformat(),
                "additional_metadata": {
                    "person_count": person_count,
                    "confidence": f"{confidence:.3f}",
                    "event_type": "person_detected",
                    "detection_time": datetime.now().strftime("%H:%M:%S"),
                    "is_nighttime": True,
                    "vision_service": "yolov8n-pose"
                }
            }
            
            with open(metadata_filepath, 'w') as meta_f:
                json.dump(metadata_content, meta_f, indent=2)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                LOGGER.info(f"✅ Person detection image saved: {filename} ({file_size} bytes)")
                LOGGER.info(f"📋 Metadata saved: {metadata_filename}")
                LOGGER.info(f"🎯 Component: {camera_name}, Tags: ['PersonDetection', 'Nighttime']")
                LOGGER.info("🔄 Files will sync to Viam within 1 minute")
                
                return {"status": "success", "method": "file_fallback", "filename": filename, "path": filepath}
            else:
                LOGGER.error(f"❌ Failed to save: {filepath}")
                return {"status": "error", "method": "file_fallback", "error": "File not saved"}
                
        except Exception as e:
            LOGGER.error(f"❌ Error in file fallback: {e}")
            return {"status": "error", "method": "file_fallback", "error": str(e)}
