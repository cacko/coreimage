__name__ = "coreimage"
import warnings
from importlib.metadata import version, PackageNotFoundError
__version__ = version(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import corelog
import os

corelog.register(os.environ.get("COREIMAGE_LOG_LEVEL", "INFO"), corelog.Handlers.RICH)
