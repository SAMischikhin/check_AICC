import logging

_log_format = u"%(asctime)s - [%(levelname)s] - %(message)s"

def get_file_handler(log_path):
    file_handler = logging.FileHandler(log_path, mode='w', encoding="UTF-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(log_path, name = __name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_file_handler(log_path))
    logger.addHandler(get_stream_handler())
    return logger
