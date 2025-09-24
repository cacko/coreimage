__name__ = "coreimage"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import corelog
import os

corelog.register(os.environ.get("COREIMAGE_LOG_LEVEL", "INFO"), corelog.Handlers.RICH)
