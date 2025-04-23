"""
Модуль публикации данных для системы ArcticCyclone.

Предоставляет классы для публикации и распространения
результатов обнаружения и отслеживания циклонов.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import tempfile
import json
import shutil
import datetime
import ftplib
import requests

from models.cyclone import Cyclone
from core.exceptions import ExportError
from export.formats.csv_exporter import CycloneCSVExporter
from export.formats.netcdf_exporter import CycloneNetCDFExporter

# Инициализация логгера
logger = logging.getLogger(__name__)

class DataPublisher:
    """
    Базовый класс для публикации данных о циклонах.
    
    Предоставляет общий интерфейс для различных методов публикации
    результатов обнаружения и отслеживания циклонов.
    """
    
    def __init__(self, export_format: str = 'csv',
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализирует публикатор данных.
        
        Аргументы:
            export_format: Формат экспорта данных ('csv', 'netcdf', 'geojson', 'shapefile').
            metadata: Метаданные для включения в экспортированные данные.
        """
        self.export_format = export_format
        self.metadata = metadata or {}
        
        # Добавляем стандартные метаданные
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.datetime.now().isoformat()
        if 'generator' not in self.metadata:
            self.metadata['generator'] = 'ArcticCyclone'
        
        # Создаем экспортер в зависимости от формата
        self._create_exporter()
        
        logger.debug(f"Инициализирован публикатор данных формата {export_format}")
    
    def _create_exporter(self) -> None:
        """
        Создает экспортер для выбранного формата данных.
        
        Вызывает:
            ExportError: Если указан неподдерживаемый формат экспорта.
        """
        if self.export_format == 'csv':
            self.exporter = CycloneCSVExporter()
        elif self.export_format == 'netcdf':
            self.exporter = CycloneNetCDFExporter()
        elif self.export_format == 'geojson':
            from formats.geojson_exporter import CycloneGeoJSONExporter
            self.exporter = CycloneGeoJSONExporter()
        elif self.export_format == 'shapefile':
            from formats.shapefile_exporter import CycloneShapefileExporter
            self.exporter = CycloneShapefileExporter()
        else:
            raise ExportError(f"Неподдерживаемый формат экспорта: {self.export_format}")
    
    def publish(self, *args, **kwargs) -> Any:
        """
        Публикует данные о циклонах.
        
        Аргументы и результат зависят от конкретной реализации в дочерних классах.
        
        Вызывает:
            NotImplementedError: Если метод не переопределен в дочернем классе.
        """
        raise NotImplementedError("Метод publish должен быть переопределен в дочернем классе")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Добавляет метаданные для включения в экспортированные данные.
        
        Аргументы:
            key: Ключ метаданных.
            value: Значение метаданных.
        """
        self.metadata[key] = value
        
    def set_export_format(self, export_format: str) -> None:
        """
        Изменяет формат экспорта данных.
        
        Аргументы:
            export_format: Новый формат экспорта данных.
            
        Вызывает:
            ExportError: Если указан неподдерживаемый формат экспорта.
        """
        if export_format != self.export_format:
            self.export_format = export_format
            self._create_exporter()
            logger.debug(f"Формат экспорта изменен на {export_format}")


class FilePublisher(DataPublisher):
    """
    Публикатор данных в файл.
    
    Экспортирует данные о циклонах в файл заданного формата.
    """
    
    def __init__(self, output_dir: Union[str, Path],
                export_format: str = 'csv',
                metadata: Optional[Dict[str, Any]] = None,
                filename_template: Optional[str] = None):
        """
        Инициализирует публикатор данных в файл.
        
        Аргументы:
            output_dir: Директория для сохранения экспортированных файлов.
            export_format: Формат экспорта данных.
            metadata: Метаданные для включения в экспортированные данные.
            filename_template: Шаблон имени файла. Если None, используется шаблон по умолчанию.
        """
        super().__init__(export_format, metadata)
        
        self.output_dir = Path(output_dir)
        self.filename_template = filename_template or "arctic_cyclones_{date}_{format}"
        
        # Создаем директорию, если она не существует
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug(f"Инициализирован публикатор в файл. Директория: {self.output_dir}")
    
    def publish(self, cyclones: List[Cyclone], 
               filename: Optional[str] = None) -> Path:
        """
        Публикует данные о циклонах в файл.
        
        Аргументы:
            cyclones: Список циклонов для публикации.
            filename: Имя файла. Если None, генерируется автоматически.
            
        Возвращает:
            Путь к созданному файлу.
            
        Вызывает:
            ExportError: При ошибке экспорта данных.
        """
        try:
            # Генерируем имя файла, если не указано
            if filename is None:
                date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.filename_template.format(
                    date=date_str, format=self.export_format)
                
            # Добавляем расширение, если отсутствует
            if not filename.endswith(f".{self.export_format}") and self.export_format != 'shapefile':
                filename = f"{filename}.{self.export_format}"
                
            # Полный путь к файлу
            file_path = self.output_dir / filename
            
            # Экспортируем данные в файл
            if self.export_format == 'csv':
                self.exporter.export_cyclone_tracks(cyclones, file_path)
            elif self.export_format == 'netcdf':
                self.exporter.export_to_netcdf(cyclones, file_path, self.metadata)
            elif self.export_format == 'geojson':
                self.exporter.export_to_geojson(cyclones, file_path, self.metadata)
            elif self.export_format == 'shapefile':
                # Для shapefile имя файла не должно содержать расширение
                shp_path = self.output_dir / filename
                self.exporter.export_to_shapefile(cyclones, shp_path, self.metadata)
            else:
                raise ExportError(f"Неподдерживаемый формат экспорта: {self.export_format}")
            
            logger.info(f"Данные успешно экспортированы в файл: {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Ошибка при публикации данных в файл: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def publish_statistics(self, cyclones: List[Cyclone], 
                         filename: Optional[str] = None) -> Path:
        """
        Публикует статистику циклонов в файл CSV.
        
        Аргументы:
            cyclones: Список циклонов для публикации статистики.
            filename: Имя файла. Если None, генерируется автоматически.
            
        Возвращает:
            Путь к созданному файлу.
            
        Вызывает:
            ExportError: При ошибке экспорта данных.
        """
        try:
            # Генерируем имя файла, если не указано
            if filename is None:
                date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"arctic_cyclones_stats_{date_str}.csv"
                
            # Полный путь к файлу
            file_path = self.output_dir / filename
            
            # Экспортируем статистику в CSV
            self.exporter.export_cyclone_statistics(cyclones, file_path)
            
            logger.info(f"Статистика успешно экспортирована в файл: {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Ошибка при публикации статистики в файл: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)


class WebPublisher(DataPublisher):
    """
    Публикатор данных на веб-ресурс.
    
    Экспортирует данные о циклонах на удаленный сервер
    через HTTP, FTP или другие протоколы.
    """
    
    def __init__(self, server_url: str,
                username: Optional[str] = None,
                password: Optional[str] = None,
                export_format: str = 'csv',
                metadata: Optional[Dict[str, Any]] = None,
                protocol: str = 'http'):
        """
        Инициализирует публикатор данных на веб-ресурс.
        
        Аргументы:
            server_url: URL сервера для публикации.
            username: Имя пользователя для аутентификации.
            password: Пароль для аутентификации.
            export_format: Формат экспорта данных.
            metadata: Метаданные для включения в экспортированные данные.
            protocol: Протокол передачи ('http', 'ftp').
        """
        super().__init__(export_format, metadata)
        
        self.server_url = server_url
        self.username = username
        self.password = password
        self.protocol = protocol.lower()
        
        # Проверяем поддерживаемые протоколы
        if self.protocol not in ['http', 'ftp']:
            raise ValueError(f"Неподдерживаемый протокол: {protocol}")
        
        logger.debug(f"Инициализирован веб-публикатор. Сервер: {server_url}, протокол: {protocol}")
    
    def publish(self, cyclones: List[Cyclone], 
               remote_path: Optional[str] = None) -> str:
        """
        Публикует данные о циклонах на веб-ресурс.
        
        Аргументы:
            cyclones: Список циклонов для публикации.
            remote_path: Путь на сервере для публикации. Если None, используется корневой путь.
            
        Возвращает:
            URL опубликованных данных.
            
        Вызывает:
            ExportError: При ошибке публикации данных.
        """
        try:
            # Создаем временный файл для экспорта
            with tempfile.NamedTemporaryFile(suffix=f".{self.export_format}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            # Экспортируем данные во временный файл
            if self.export_format == 'csv':
                self.exporter.export_cyclone_tracks(cyclones, temp_path)
            elif self.export_format == 'netcdf':
                self.exporter.export_to_netcdf(cyclones, temp_path, self.metadata)
            elif self.export_format == 'geojson':
                self.exporter.export_to_geojson(cyclones, temp_path, self.metadata)
            else:
                raise ExportError(f"Неподдерживаемый формат экспорта для веб-публикации: {self.export_format}")
            
            # Публикуем файл на сервер
            if self.protocol == 'http':
                url = self._publish_http(temp_path, remote_path)
            elif self.protocol == 'ftp':
                url = self._publish_ftp(temp_path, remote_path)
            else:
                raise ExportError(f"Неподдерживаемый протокол: {self.protocol}")
            
            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {temp_path}: {str(e)}")
            
            logger.info(f"Данные успешно опубликованы по URL: {url}")
            return url
            
        except Exception as e:
            error_msg = f"Ошибка при публикации данных на веб-ресурс: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def _publish_http(self, file_path: Path, remote_path: Optional[str] = None) -> str:
        """
        Публикует файл через HTTP(S).
        
        Аргументы:
            file_path: Путь к файлу для публикации.
            remote_path: Путь на сервере.
            
        Возвращает:
            URL опубликованного файла.
            
        Вызывает:
            ExportError: При ошибке публикации.
        """
        # Формируем URL для публикации
        url = self.server_url
        if remote_path:
            url = f"{url.rstrip('/')}/{remote_path.lstrip('/')}"
        
        # Открываем файл для чтения
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f)}
            
            # Аутентификация, если требуется
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            # Отправляем запрос
            response = requests.post(url, files=files, auth=auth)
            
            # Проверяем результат
            if response.status_code not in [200, 201, 202]:
                raise ExportError(f"Ошибка HTTP при публикации: {response.status_code}, {response.text}")
            
            # Возвращаем URL публикации
            try:
                result_url = response.json().get('url')
            except:
                # Если сервер не вернул URL, формируем его из имени файла
                result_url = f"{url}/{file_path.name}"
            
            return result_url
    
    def _publish_ftp(self, file_path: Path, remote_path: Optional[str] = None) -> str:
        """
        Публикует файл через FTP.
        
        Аргументы:
            file_path: Путь к файлу для публикации.
            remote_path: Путь на сервере.
            
        Возвращает:
            URL опубликованного файла.
            
        Вызывает:
            ExportError: При ошибке публикации.
        """
        # Извлекаем хост из URL
        if self.server_url.startswith('ftp://'):
            host = self.server_url[6:]
        else:
            host = self.server_url
        
        # Удаляем порт из хоста, если есть
        if ':' in host:
            host = host.split(':')[0]
        
        try:
            # Подключаемся к FTP-серверу
            ftp = ftplib.FTP(host)
            
            # Аутентификация
            if self.username and self.password:
                ftp.login(self.username, self.password)
            else:
                ftp.login()
            
            # Переходим в удаленную директорию, если указана
            if remote_path:
                try:
                    ftp.cwd(remote_path)
                except:
                    # Создаем директорию, если не существует
                    dirs = remote_path.split('/')
                    for d in dirs:
                        if d:
                            try:
                                ftp.cwd(d)
                            except:
                                ftp.mkd(d)
                                ftp.cwd(d)
            
            # Открываем файл для чтения и загружаем на сервер
            with open(file_path, 'rb') as f:
                ftp.storbinary(f"STOR {file_path.name}", f)
            
            # Получаем URL публикации
            if remote_path:
                url = f"ftp://{host}/{remote_path.strip('/')}/{file_path.name}"
            else:
                url = f"ftp://{host}/{file_path.name}"
            
            # Закрываем соединение
            ftp.quit()
            
            return url
            
        except Exception as e:
            raise ExportError(f"Ошибка FTP при публикации: {str(e)}")


class EmailPublisher(DataPublisher):
    """
    Публикатор данных по электронной почте.
    
    Отправляет данные о циклонах по электронной почте.
    """
    
    def __init__(self, smtp_server: str,
                smtp_port: int = 587,
                username: str = None,
                password: str = None,
                sender: str = None,
                export_format: str = 'csv',
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализирует публикатор данных по электронной почте.
        
        Аргументы:
            smtp_server: Адрес SMTP-сервера.
            smtp_port: Порт SMTP-сервера.
            username: Имя пользователя для аутентификации.
            password: Пароль для аутентификации.
            sender: Адрес отправителя.
            export_format: Формат экспорта данных.
            metadata: Метаданные для включения в экспортированные данные.
        """
        super().__init__(export_format, metadata)
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender or username
        
        logger.debug(f"Инициализирован публикатор по электронной почте. Сервер: {smtp_server}:{smtp_port}")
    
    def publish(self, cyclones: List[Cyclone], 
               recipients: List[str],
               subject: str = "Arctic Cyclone Data",
               message: str = None) -> bool:
        """
        Публикует данные о циклонах по электронной почте.
        
        Аргументы:
            cyclones: Список циклонов для публикации.
            recipients: Список адресов получателей.
            subject: Тема сообщения.
            message: Текст сообщения. Если None, используется стандартное сообщение.
            
        Возвращает:
            True, если отправка успешна, иначе False.
            
        Вызывает:
            ExportError: При ошибке отправки данных.
        """
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.mime.application import MIMEApplication
            
            # Создаем временный файл для экспорта
            with tempfile.NamedTemporaryFile(suffix=f".{self.export_format}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            # Экспортируем данные во временный файл
            if self.export_format == 'csv':
                self.exporter.export_cyclone_tracks(cyclones, temp_path)
                attachment_name = f"arctic_cyclones_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            elif self.export_format == 'netcdf':
                self.exporter.export_to_netcdf(cyclones, temp_path, self.metadata)
                attachment_name = f"arctic_cyclones_{datetime.datetime.now().strftime('%Y%m%d')}.nc"
            elif self.export_format == 'geojson':
                self.exporter.export_to_geojson(cyclones, temp_path, self.metadata)
                attachment_name = f"arctic_cyclones_{datetime.datetime.now().strftime('%Y%m%d')}.geojson"
            else:
                raise ExportError(f"Неподдерживаемый формат экспорта для отправки по почте: {self.export_format}")
            
            # Создаем сообщение
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            # Добавляем текст сообщения
            if message is None:
                message = (
                    "Уважаемый пользователь,\n\n"
                    "Во вложении представлены данные об арктических циклонах, "
                    "обнаруженных системой ArcticCyclone.\n\n"
                    f"Формат данных: {self.export_format}\n"
                    f"Дата создания: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    "С уважением,\nСистема ArcticCyclone"
                )
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Добавляем вложение
            with open(temp_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), Name=attachment_name)
            
            attachment['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            msg.attach(attachment)
            
            # Отправляем сообщение
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {temp_path}: {str(e)}")
            
            logger.info(f"Данные успешно отправлены по электронной почте получателям: {', '.join(recipients)}")
            return True
            
        except Exception as e:
            error_msg = f"Ошибка при отправке данных по электронной почте: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)