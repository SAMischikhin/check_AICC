import configparser
import sys
from subprocess import Popen

"""Сообщения, идущие и в лог и в консоль. logger создается в main.py,
поскольку ему необходимо передать путь на конкретную папку решения"""

class MyInfo(Exception):
    def __init__(self, logger):
         self.logger = logger

    def send (self, message):
        super().__init__(message)
        self.logger.info(message)
        
class MyWarning(Exception):
    def __init__(self, logger):
         self.logger = logger

    def send (self, message):
        super().__init__(message)
        self.logger.warning(message)
        
        
class MyError(Exception):
    def __init__(self, logger):
         self.logger = logger
         
    def _open_log_file(self):
        """Открытие log файла"""
        log_path = self.logger.handlers[0].baseFilename
        cmd = r'notepad {}'.format(log_path)
        Popen(cmd)

    def send(self, message):
        super().__init__(message)
        self.logger.error(message)
        self._open_log_file()
        sys.exit()


class MyParsingError(configparser.ParsingError):
    """Parsing error -> Log"""
    def __init__(self, source, logger):
        super().__init__(source)
        logger.error(self.message)
        sys.exit()
   