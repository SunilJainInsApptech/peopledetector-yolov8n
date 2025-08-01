import asyncio
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
logger = logging.getLogger(__name__)

logger.info("Starting module...")

from viam.module.module import Module
from viam.services.vision import Vision
logger.info("Imports successful, importing yolov8...")

try:
    from models.yolov8n_pose import Yolov8nPose
    logger.info("Yolov8nPose imported successfully.")
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.yolov8n_pose import Yolov8nPose
    logger.info("Yolov8nPose imported successfully from local module.")

if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
