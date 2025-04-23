"""
Модуль настройки логирования для системы ArcticCyclone.

Предоставляет функции и классы для унифицированной настройки журналирования
в разных компонентах системы с поддержкой различных уровней детализации и форматов.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import os
import json
from datetime import datetime

class MeteorologicalLogFormatter(logging.Formatter):
    """
    Форматтер для метеорологических журналов с дополнительным контекстом.
    
    Расширяет стандартный форматтер для включения дополнительной метеорологической информации,
    такой как параметры анализа, регион исследования и временные рамки.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирует запись журнала с дополнительным метеорологическим контекстом.
        
        Аргументы:
            record: Запись журнала для форматирования.
            
        Возвращает:
            Отформатированное сообщение журнала.
        """
        # Добавляем дополнительную информацию, если она есть
        meteо_context = getattr(record, 'meteo_context', None)
        formatted_record = super().format(record)
        
        if meteо_context:
            formatted_record += f"\nМетеорологический контекст: {json.dumps(meteо_context, indent=2, ensure_ascii=False)}"
            
        return formatted_record


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    include_meteo_context: bool = True
) -> None:
    """
    Настраивает глобальную систему логирования для ArcticCyclone.
    
    Аргументы:
        log_level: Уровень детализации журнала (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Путь к файлу для сохранения журнала. Если None, журнал выводится в stdout.
        log_format: Формат сообщений журнала.
        include_meteo_context: Включать ли дополнительный метеорологический контекст.
        
    Примечание:
        Создает директорию для журнала, если она не существует.
    """
    # Преобразуем уровень логирования из строки, если необходимо
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Удаляем существующие обработчики, если они есть
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создаем форматтер
    if include_meteo_context:
        formatter = MeteorologicalLogFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)
    
    # Настраиваем вывод в файл, если указан
    if log_file:
        log_path = Path(log_file)
        log_dir = log_path.parent
        
        # Создаем директорию для журнала, если она не существует
        if not log_dir.exists():
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Добавляем вывод в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Логируем начало сеанса
    logging.info(f"Инициализация журналирования ArcticCyclone. Уровень: {logging.getLevelName(log_level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер с заданным именем.
    
    Аргументы:
        name: Имя логгера, обычно имя модуля (__name__).
        
    Возвращает:
        Настроенный логгер.
    """
    return logging.getLogger(name)


class MeteorologicalLogger(logging.Logger):
    """
    Расширенный логгер для метеорологических приложений.
    
    Добавляет методы для журналирования событий с метеорологическим контекстом.
    """
    
    def __init__(self, name: str):
        """
        Инициализирует метеорологический логгер.
        
        Аргументы:
            name: Имя логгера.
        """
        super().__init__(name)
    
    def meteo_info(self, msg: str, meteo_context: Dict[str, Any] = None, **kwargs) -> None:
        """
        Логирует информационное сообщение с метеорологическим контекстом.
        
        Аргументы:
            msg: Сообщение для записи в журнал.
            meteo_context: Словарь с дополнительной метеорологической информацией.
            **kwargs: Дополнительные аргументы для метода info.
        """
        extra = kwargs.get('extra', {})
        extra['meteo_context'] = meteo_context
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def meteo_warning(self, msg: str, meteo_context: Dict[str, Any] = None, **kwargs) -> None:
        """
        Логирует предупреждение с метеорологическим контекстом.
        
        Аргументы:
            msg: Сообщение для записи в журнал.
            meteo_context: Словарь с дополнительной метеорологической информацией.
            **kwargs: Дополнительные аргументы для метода warning.
        """
        extra = kwargs.get('extra', {})
        extra['meteo_context'] = meteo_context
        kwargs['extra'] = extra
        self.warning(msg, **kwargs)
    
    def meteo_error(self, msg: str, meteo_context: Dict[str, Any] = None, **kwargs) -> None:
        """
        Логирует ошибку с метеорологическим контекстом.
        
        Аргументы:
            msg: Сообщение для записи в журнал.
            meteo_context: Словарь с дополнительной метеорологической информацией.
            **kwargs: Дополнительные аргументы для метода error.
        """
        extra = kwargs.get('extra', {})
        extra['meteo_context'] = meteo_context
        kwargs['extra'] = extra
        self.error(msg, **kwargs)


# Регистрируем метеорологический логгер
logging.setLoggerClass(MeteorologicalLogger)