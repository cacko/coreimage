__name__ = "coreimage"

import corelog
import os

corelog.register(os.environ.get("COREIMAGE_LOG_LEVEL", "INFO"), corelog.Handlers.RICH)
