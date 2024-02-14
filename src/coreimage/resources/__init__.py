from pathlib import Path

RESOURCES_ROOT = Path(__file__).parent
UPSCALE_BSRGANx2: Path = RESOURCES_ROOT / "BSRGANx2.pth"
UPSCALE_REALESGRAN_x4PLUS = RESOURCES_ROOT / "RealESRGAN_x4plus.pth"
MEDIAPIPE_BLAZE_SHORT = RESOURCES_ROOT / "blaze_face_short_range.tflite"
MEDIAPIPE_FACE_LANDMARKER = RESOURCES_ROOT / "face_landmarker.task"