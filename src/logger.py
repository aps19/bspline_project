import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # format: month_day_year_hour_minute_second.log
logs_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE) # format: current_dir/logs/month_day_year_hour_minute_second.log
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH= os.path.join(logs_dir, LOG_FILE) # This is the path where the log file will be saved

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)s %(name)s - %(levelname)s - %(message)s", # format: [time] line_number module_name - log_level - log_message
    level=logging.INFO, # Change this to DEBUG if you want to see all the logs
)
