import logging


class Logger:
    def __init__(self, component, target, level):
        self._component = component
        self._target = target
        self._level = level
        self._logger = logging.getLogger(component)
        self._logger.setLevel(level)
        fh = logging.FileHandler(target)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def set_level(self, level):
        self._level = level

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)

    def exception(self, msg):
        self._logger.exception(msg)

    def info(self, msg):
        self._logger.info(msg)

    @classmethod
    def logger(cls, component, target='error.log', level=logging.WARNING):
        return cls(component, target, level)

    @classmethod
    def info_logger(cls, component, target='error.log', level=logging.INFO):
        return cls(component, target, level)
