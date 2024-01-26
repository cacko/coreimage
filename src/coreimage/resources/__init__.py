from pathlib import Path

RESOURCES_ROOT = Path(__file__).parent
HAARCASCADE_XML = RESOURCES_ROOT / "haarcascade_frontalface_alt2.xml"
CASCADE_BACK = RESOURCES_ROOT / "cascade_back.xml"
CASCADE_FRONT = RESOURCES_ROOT / "cascade_front.xml"
CASCADE_SIDE = RESOURCES_ROOT / "cascade_side.xml"