from loguru import logger
from datetime import datetime
import os


# remove the default logger
# logger.remove()

# add a new logger with a custom format
prefix = f"{os.environ['HOST']}_{os.environ['PORT']}"
time = datetime.now().strftime("%Y-%m-%d")
filename = f"{prefix}_{time}"
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
logger.add(
    f"logs/{filename}.log",
    format=logger_format,
    level="INFO",
    rotation="1 day",
    compression="zip",
    retention="3 months",
)

# if you want to log to stdout as well, you can add another logger
# logger.add(sys.stdout, format=logger_format, level="INFO")
