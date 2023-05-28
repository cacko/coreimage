__name__ = "coreimage"

from corelog import register
import os

register(os.environ.get("COREIMAGE_LOG_LEVEL", "INFO"))
