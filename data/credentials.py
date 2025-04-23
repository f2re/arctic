"""
Модуль управления учетными данными для системы ArcticCyclone.

Предоставляет безопасное хранение и доступ к учетным данным для различных
источников метеорологических данных.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import json
import logging
from ..core.exceptions import CredentialError

# Инициализация логгера
logger = logging.getLogger(__name__)

class CredentialManager:
    """
    Управляет учетными данными для доступа к источникам данных.
    
    Обеспечивает безопасное хранение и доступ к API-ключам и другим
    учетным данным, необходимым для получения метеорологических данных.
    """
    
    def __init__(self, keyring_service: str = "arctic_cyclone",
                credentials_file: Optional[Path] = None):
        """
        Инициализирует менеджер учетных данных.
        
        Аргументы:
            keyring_service: Имя службы для хранения учетных данных в системном keyring.
            credentials_file: Путь к файлу с учетными данными. Если None, используется
                             файл ~/.config/arctic_cyclone/credentials.json.
        """
        self.service = keyring_service
        self._credentials = {}
        self.credentials_file = credentials_file or Path.home() / ".config" / "arctic_cyclone" / "credentials.json"
        
        # Загружаем учетные данные при инициализации
        self._load_credentials()
        
    def _load_credentials(self) -> None:
        """
        Загружает учетные данные из файла или системного keyring.
        
        Примечание:
            Сначала пытается загрузить из файла, затем из системного keyring.
            Если оба метода не дают результатов, создает пустой словарь учетных данных.
        """
        # Попытка загрузки из файла
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    self._credentials = json.load(f)
                logger.info(f"Учетные данные загружены из {self.credentials_file}")
                return
            except Exception as e:
                logger.warning(f"Не удалось загрузить учетные данные из файла: {str(e)}")
        
        # Попытка загрузки из системного keyring
        try:
            import keyring
            
            # Получаем список источников данных
            sources_str = keyring.get_password(self.service, "sources")
            if sources_str:
                sources = json.loads(sources_str)
                
                # Загружаем учетные данные для каждого источника
                for source in sources:
                    username = keyring.get_password(self.service, f"{source}_user")
                    api_key = keyring.get_password(self.service, f"{source}_key")
                    
                    if username or api_key:
                        self._credentials[source] = {
                            "username": username,
                            "api_key": api_key
                        }
                
                logger.info("Учетные данные загружены из системного keyring")
                return
                
        except ImportError:
            logger.warning("Модуль keyring не установлен, невозможно загрузить учетные данные из системного хранилища")
        except Exception as e:
            logger.warning(f"Не удалось загрузить учетные данные из keyring: {str(e)}")
        
        # Если ничего не загружено, создаем пустой словарь
        self._credentials = {}
        logger.info("Учетные данные не найдены, создан пустой словарь")
    
    def get(self, source: str) -> Dict[str, str]:
        """
        Получает учетные данные для указанного источника данных.
        
        Аргументы:
            source: Имя источника данных.
            
        Возвращает:
            Словарь с учетными данными для источника.
            
        Вызывает:
            CredentialError: Если учетные данные для источника не найдены.
        """
        if source not in self._credentials:
            # Попытка загрузки из переменных окружения
            if self._load_from_env(source):
                return self._credentials[source]
                
            raise CredentialError(f"Учетные данные для источника {source} не найдены")
            
        return self._credentials[source]
    
    def _load_from_env(self, source: str) -> bool:
        """
        Загружает учетные данные из переменных окружения.
        
        Аргументы:
            source: Имя источника данных.
            
        Возвращает:
            True, если учетные данные успешно загружены, иначе False.
            
        Примечание:
            Ищет переменные окружения вида SOURCE_USERNAME и SOURCE_API_KEY,
            где SOURCE - имя источника данных в верхнем регистре.
        """
        env_prefix = source.upper()
        username = os.environ.get(f"{env_prefix}_USERNAME")
        api_key = os.environ.get(f"{env_prefix}_API_KEY")
        
        if username or api_key:
            self._credentials[source] = {
                "username": username,
                "api_key": api_key
            }
            logger.info(f"Учетные данные для источника {source} загружены из переменных окружения")
            return True
            
        return False
    
    def set(self, source: str, username: Optional[str] = None, 
           api_key: Optional[str] = None, save: bool = True) -> None:
        """
        Устанавливает учетные данные для источника данных.
        
        Аргументы:
            source: Имя источника данных.
            username: Имя пользователя для источника.
            api_key: API-ключ для источника.
            save: Сохранять ли учетные данные в файл или keyring.
        """
        if source not in self._credentials:
            self._credentials[source] = {}
            
        if username:
            self._credentials[source]["username"] = username
            
        if api_key:
            self._credentials[source]["api_key"] = api_key
            
        logger.info(f"Установлены учетные данные для источника {source}")
        
        if save:
            self._save_credentials()
    
    def _save_credentials(self) -> None:
        """
        Сохраняет учетные данные в файл и/или системный keyring.
        
        Примечание:
            Пытается сохранить в файл и в системный keyring, если доступно.
        """
        # Сохранение в файл
        try:
            # Создаем директорию, если она не существует
            os.makedirs(self.credentials_file.parent, exist_ok=True)
            
            with open(self.credentials_file, 'w') as f:
                json.dump(self._credentials, f)
                
            # Устанавливаем права доступа только для владельца
            os.chmod(self.credentials_file, 0o600)
            
            logger.info(f"Учетные данные сохранены в {self.credentials_file}")
            
        except Exception as e:
            logger.error(f"Не удалось сохранить учетные данные в файл: {str(e)}")
        
        # Сохранение в системный keyring
        try:
            import keyring
            
            # Сохраняем список источников
            sources = list(self._credentials.keys())
            keyring.set_password(self.service, "sources", json.dumps(sources))
            
            # Сохраняем учетные данные для каждого источника
            for source, creds in self._credentials.items():
                if "username" in creds:
                    keyring.set_password(self.service, f"{source}_user", creds["username"])
                if "api_key" in creds:
                    keyring.set_password(self.service, f"{source}_key", creds["api_key"])
            
            logger.info("Учетные данные сохранены в системный keyring")
            
        except ImportError:
            logger.warning("Модуль keyring не установлен, невозможно сохранить учетные данные в системное хранилище")
        except Exception as e:
            logger.warning(f"Не удалось сохранить учетные данные в keyring: {str(e)}")
    
    def delete(self, source: str, save: bool = True) -> None:
        """
        Удаляет учетные данные для источника данных.
        
        Аргументы:
            source: Имя источника данных.
            save: Сохранять ли изменения в файл или keyring.
        """
        if source in self._credentials:
            del self._credentials[source]
            logger.info(f"Удалены учетные данные для источника {source}")
            
            if save:
                self._save_credentials()
                
                # Удаляем из системного keyring
                try:
                    import keyring
                    keyring.delete_password(self.service, f"{source}_user")
                    keyring.delete_password(self.service, f"{source}_key")
                    
                    # Обновляем список источников
                    sources = list(self._credentials.keys())
                    keyring.set_password(self.service, "sources", json.dumps(sources))
                    
                except (ImportError, Exception) as e:
                    logger.warning(f"Не удалось удалить учетные данные из keyring: {str(e)}")
    
    def list_sources(self) -> Dict[str, Dict]:
        """
        Возвращает список источников данных с учетными данными.
        
        Возвращает:
            Словарь с именами источников и маскированными учетными данными.
        """
        result = {}
        
        for source, creds in self._credentials.items():
            masked = {}
            
            if "username" in creds:
                username = creds["username"]
                masked["username"] = username[:2] + "*" * (len(username) - 2) if username else None
                
            if "api_key" in creds:
                api_key = creds["api_key"]
                masked["api_key"] = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if api_key else None
                
            result[source] = masked
            
        return result