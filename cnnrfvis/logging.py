# logging.py
import logging

def setup_logging(log_file='app.log', log_level=logging.INFO):
    """
    Configure the logger

    Args:
        log_file (str): log file name, default is app.log。
        log_level (int): log level, default is logging.INFO。
    """
    logging.basicConfig(
        level=log_level,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file)  # 输出到文件
        ]
    )

# 获取日志记录器
logger = logging.getLogger(__name__)