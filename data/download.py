# Функции загрузки данных ERA5
"""
Функции для загрузки данных ERA5.
"""

import os
import xarray as xr
import cdsapi

def setup_cdsapirc(arctic_dir):
    """
    Настройка файла .cdsapirc для доступа к API CDS.
    
    Параметры:
    ----------
    arctic_dir : str
        Директория для хранения файла конфигурации
    """
    # Реализация без изменений из оригинального файла
    # Путь к .cdsapirc в домашнем каталоге (для API)
    home_cdsapirc_path = os.path.expanduser("~/.cdsapirc")
    # Путь к копии файла в Google Drive
    drive_cdsapirc_path = os.path.join(arctic_dir, '.cdsapirc')

    # Проверяем наличие файла в Google Drive
    if os.path.exists(drive_cdsapirc_path):
        print(f"Найден файл .cdsapirc в Google Drive: {drive_cdsapirc_path}")

        # Копируем файл из Drive в домашний каталог
        with open(drive_cdsapirc_path, 'r') as f_drive:
            content = f_drive.read()

        with open(home_cdsapirc_path, 'w') as f_home:
            f_home.write(content)

        print(f"Файл .cdsapirc скопирован из Google Drive в {home_cdsapirc_path}")

    elif not os.path.exists(home_cdsapirc_path):
        print("\nФайл .cdsapirc не найден. Необходимо создать его для доступа к API CDS.")
        print("Согласно официальной документации файл должен содержать:")
        print("url: https://cds.climate.copernicus.eu/api")
        print("key: ваш-api-ключ")

        create_file = input("\nСоздать файл .cdsapirc? (y/n): ")

        if create_file.lower() == 'y':
            api_key = input("Введите ваш API-ключ: ")

            # Создаем содержимое файла
            content = f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}"

            # Записываем в домашний каталог
            with open(home_cdsapirc_path, 'w') as f:
                f.write(content)

            # Сохраняем копию в Google Drive
            with open(drive_cdsapirc_path, 'w') as f:
                f.write(content)

            print(f"Файл .cdsapirc создан в {home_cdsapirc_path}")
            print(f"Копия файла сохранена в Google Drive: {drive_cdsapirc_path}")
        else:
            print("Необходимо создать файл .cdsapirc для доступа к API CDS.")
            sys.exit(1)
    else:
        # Проверяем содержимое файла в домашнем каталоге
        with open(home_cdsapirc_path, 'r') as f:
            content = f.read()

        # Проверяем URL в файле
        if "url: https://cds.climate.copernicus.eu/api/v2" in content:
            print("\nВНИМАНИЕ: В файле .cdsapirc указан устаревший URL.")
            print("Согласно официальной документации, необходимо использовать:")
            print("url: https://cds.climate.copernicus.eu/api")

            update_url = input("\nОбновить URL в файле? (y/n): ")

            if update_url.lower() == 'y':
                # Обновляем URL
                updated_content = content.replace(
                    "url: https://cds.climate.copernicus.eu/api/v2",
                    "url: https://cds.climate.copernicus.eu/api"
                )

                # Сохраняем копию старого файла
                backup_path = home_cdsapirc_path + ".backup"
                with open(backup_path, 'w') as f:
                    f.write(content)
                print(f"Резервная копия сохранена в: {backup_path}")

                # Записываем обновленный файл
                with open(home_cdsapirc_path, 'w') as f:
                    f.write(updated_content)

                # Обновляем копию в Google Drive
                with open(drive_cdsapirc_path, 'w') as f:
                    f.write(updated_content)

                print("URL успешно обновлен в файле .cdsapirc")
                print(f"Копия файла обновлена в Google Drive: {drive_cdsapirc_path}")

        elif "url: https://cds.climate.copernicus.eu/api" not in content:
            print("\nВНИМАНИЕ: В файле .cdsapirc может быть указан некорректный URL.")
            print("Согласно официальной документации, необходимо использовать:")
            print("url: https://cds.climate.copernicus.eu/api")

            show_content = input("\nПоказать текущее содержимое файла? (y/n): ")
            if show_content.lower() == 'y':
                print("\nТекущее содержимое файла .cdsapirc:")
                print(content)

            update_file = input("\nОбновить файл .cdsapirc? (y/n): ")
            if update_file.lower() == 'y':
                api_key = ""
                if "key:" in content:
                    # Извлекаем ключ из существующего файла
                    for line in content.split('\n'):
                        if line.strip().startswith("key:"):
                            api_key = line.strip()[4:].strip()

                if not api_key:
                    api_key = input("Введите ваш API-ключ: ")

                # Создаем новое содержимое
                updated_content = f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}"

                # Записываем обновленный файл
                with open(home_cdsapirc_path, 'w') as f:
                    f.write(updated_content)

                # Обновляем копию в Google Drive
                with open(drive_cdsapirc_path, 'w') as f:
                    f.write(updated_content)

                print("Файл .cdsapirc успешно обновлен")
                print(f"Копия файла обновлена в Google Drive: {drive_cdsapirc_path}")
        else:
            print("Файл .cdsapirc имеет корректный URL согласно документации.")

            # Создаем копию в Google Drive, если её еще нет
            if not os.path.exists(drive_cdsapirc_path):
                with open(drive_cdsapirc_path, 'w') as f:
                    f.write(content)
                print(f"Копия файла .cdsapirc сохранена в Google Drive: {drive_cdsapirc_path}")


def download_era5_data_extended(start_date, end_date, data_dir, output_file='era5_arctic_data.nc', 
                              region=None, era5_data_info=None, detection_methods=None, analysis_level='basic'):
    """
    Загрузка расширенного набора данных ERA5 для арктического региона с поддержкой
    раздельной загрузки однослойных и многослойных переменных.
    
    Параметры:
    ----------
    start_date : str
        Начальная дата в формате 'YYYY-MM-DD'
    end_date : str
        Конечная дата в формате 'YYYY-MM-DD'
    data_dir : str
        Директория для сохранения данных
    output_file : str, optional
        Имя выходного файла
    region : list, optional
        Регион в формате [север, запад, юг, восток]
    era5_data_info : dict, optional
        Информация о необходимых данных ERA5 (результат функции determine_required_era5_variables).
        Если None, информация будет определена автоматически на основе detection_methods.
    detection_methods : list, optional
        Список методов обнаружения циклонов. Используется, если era5_data_info не задан.
    analysis_level : str, optional
        Уровень детализации анализа. Используется, если era5_data_info не задан.
        
    Возвращает:
    -----------
    str
        Путь к загруженному файлу
    """
    import cdsapi
    import os
    
    # Установка региона по умолчанию, если не указан
    if region is None:
        region = [90, -180, 65, 180]  # [север, запад, юг, восток]
    
    # Если информация о данных не предоставлена, определяем её на основе методов обнаружения
    if era5_data_info is None:
        if detection_methods is None:
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient']
        era5_data_info = determine_required_era5_variables(detection_methods, analysis_level)
    
    # Получаем списки переменных и уровней
    single_level_vars = era5_data_info['single_level_vars']
    pressure_level_vars = era5_data_info['pressure_level_vars']
    pressure_levels = era5_data_info['pressure_levels']
    
    print(f"Запрос данных ERA5 для периода {start_date} - {end_date}")
    print(f"Регион: {region}")
    print(f"Необходимые переменные:")
    print(f"  Однослойные: {', '.join(single_level_vars)}")
    if pressure_level_vars:
        print(f"  Многослойные: {', '.join(pressure_level_vars)}")
        print(f"  Уровни давления: {', '.join(pressure_levels)}")
    
    # Создаем директорию, если она не существует
    os.makedirs(data_dir, exist_ok=True)
    
    # Полный путь к файлу
    file_path = os.path.join(data_dir, output_file)
    
    # Проверяем, существует ли уже файл
    if os.path.exists(file_path):
        print(f"Файл {file_path} уже существует, используем его")
        return file_path
    
    # Создаем клиент CDS API
    try:
        c = cdsapi.Client()
    except Exception as e:
        print(f"Ошибка при создании клиента CDS API: {e}")
        print("Убедитесь, что файл .cdsapirc настроен корректно")
        return None
    
    # Создаем пути для временных файлов
    single_level_file = os.path.join(data_dir, f"temp_sl_{os.path.basename(file_path)}")
    pressure_level_file = os.path.join(data_dir, f"temp_pl_{os.path.basename(file_path)}")
    
    # Флаги для отслеживания загрузки
    single_level_downloaded = False
    pressure_level_downloaded = False
    
    # Загружаем однослойные переменные, если они есть
    if single_level_vars:
        try:
            print(f"Загрузка однослойных переменных: {', '.join(single_level_vars)}")
            request_params = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': single_level_vars,
                'year': [start_date[:4], end_date[:4]],
                'month': [start_date[5:7], end_date[5:7]],
                'day': list(range(1, 32)),
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': region,  # [север, запад, юг, восток]
            }
            
            c.retrieve('reanalysis-era5-single-levels', request_params, single_level_file)
            single_level_downloaded = True
            print(f"Однослойные переменные успешно загружены в {single_level_file}")
        except Exception as e:
            print(f"Ошибка при загрузке однослойных переменных: {e}")
    
    # Загружаем многослойные переменные, если они есть
    if pressure_level_vars and pressure_levels:
        try:
            print(f"Загрузка многослойных переменных: {', '.join(pressure_level_vars)} на уровнях {', '.join(pressure_levels)}")
            request_params = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': pressure_level_vars,
                'pressure_level': pressure_levels,
                'year': [start_date[:4], end_date[:4]],
                'month': [start_date[5:7], end_date[5:7]],
                'day': list(range(1, 32)),
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': region,  # [север, запад, юг, восток]
            }
            
            c.retrieve('reanalysis-era5-pressure-levels', request_params, pressure_level_file)
            pressure_level_downloaded = True
            print(f"Многослойные переменные успешно загружены в {pressure_level_file}")
        except Exception as e:
            print(f"Ошибка при загрузке многослойных переменных: {e}")
    
    # Объединяем файлы, если необходимо
    if single_level_downloaded and pressure_level_downloaded:
        try:
            import xarray as xr
            
            print(f"Объединение файлов однослойных и многослойных переменных...")
            ds_sl = xr.open_dataset(single_level_file)
            ds_pl = xr.open_dataset(pressure_level_file)
            
            # Объединяем данные
            ds_combined = xr.merge([ds_sl, ds_pl])
            
            # Сохраняем объединенный датасет
            ds_combined.to_netcdf(file_path)
            
            # Закрываем датасеты
            ds_sl.close()
            ds_pl.close()
            
            # Удаляем временные файлы
            import os
            os.remove(single_level_file)
            os.remove(pressure_level_file)
            
            print(f"Данные успешно объединены и сохранены в {file_path}")
        except Exception as e:
            print(f"Ошибка при объединении файлов: {e}")
            # Если не удалось объединить, используем один из загруженных файлов
            if single_level_downloaded:
                import shutil
                shutil.copy(single_level_file, file_path)
                print(f"Используем только файл с однослойными переменными: {file_path}")
                os.remove(single_level_file)
                if pressure_level_downloaded:
                    os.remove(pressure_level_file)
            elif pressure_level_downloaded:
                import shutil
                shutil.copy(pressure_level_file, file_path)
                print(f"Используем только файл с многослойными переменными: {file_path}")
                os.remove(pressure_level_file)
    elif single_level_downloaded:
        import shutil
        shutil.move(single_level_file, file_path)
        print(f"Используем только однослойные переменные: {file_path}")
    elif pressure_level_downloaded:
        import shutil
        shutil.move(pressure_level_file, file_path)
        print(f"Используем только многослойные переменные: {file_path}")
    else:
        print(f"Ошибка: не удалось загрузить данные ERA5")
        return None
    
    print(f"Данные успешно загружены и сохранены в {file_path}")
    return file_path
    

def inspect_netcdf(file_path):
    """
    Подробная проверка структуры NetCDF файла.
    
    Параметры:
    ----------
    file_path : str
        Путь к файлу NetCDF

    Возвращает:
    -----------
    dict
        Словарь с информацией о структуре файла
    """
    # Реализация без изменений из оригинального файла
    if not file_path or file_path is None:
        print("ОШИБКА: Не указан путь к файлу")
        return None
    
    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл не существует: {file_path}")
        return None
    
    print(f"\nИсследование структуры файла: {file_path}")
    
    # Проверка наличия необходимых библиотек
    missing_deps = []
    try:
        import xarray as xr
    except ImportError:
        missing_deps.append("xarray")
    
    try:
        import netCDF4
    except ImportError:
        missing_deps.append("netCDF4")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        import h5py
    except ImportError:
        missing_deps.append("h5py")
    
    try:
        import cfgrib
    except ImportError:
        missing_deps.append("cfgrib")
    

    # Определяем тип файла
    file_type = None
    engines = []
    
    # Проверяем размер файла
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Размер файла: {file_size_mb:.2f} МБ")
    
    # Пытаемся определить тип файла по расширению и сигнатуре
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.nc':
        file_type = "NetCDF"
        engines = ['netcdf4', 'h5netcdf', 'scipy']
    elif file_ext == '.grib' or file_ext == '.grb' or file_ext == '.grib2' or file_ext == '.grb2':
        file_type = "GRIB"
        engines = ['cfgrib']
    elif file_ext == '.h5' or file_ext == '.hdf5':
        file_type = "HDF5"
        engines = ['h5netcdf']
    else:
        # Пытаемся определить тип по первым байтам файла
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if header[:4] == b'CDF\x01' or header[:4] == b'CDF\x02':
                file_type = "NetCDF"
                engines = ['netcdf4', 'h5netcdf', 'scipy']
            elif header[:4] == b'GRIB' or header[:4] == b'\x47\x52\x49\x42':
                file_type = "GRIB"
                engines = ['cfgrib']
            elif header[:8] == b'\x89\x48\x44\x46\x0d\x0a\x1a\x0a':
                file_type = "HDF5"
                engines = ['h5netcdf']
    
    if file_type:
        print(f"Определенный тип файла: {file_type}")
    else:
        print("Не удалось определить тип файла по расширению или сигнатуре")
        # Пробуем все доступные движки
        engines = ['netcdf4', 'h5netcdf', 'scipy', 'cfgrib']
    
    # Пробуем открыть файл с разными движками
    ds = None
    used_engine = None
    
    for engine in engines:
        try:
            print(f"Попытка открыть файл с движком: {engine}...")
            ds = xr.open_dataset(file_path, engine=engine)
            used_engine = engine
            print(f"Успешно открыт файл с движком: {engine}")
            break
        except Exception as e:
            print(f"Не удалось открыть с движком {engine}: {e}")
    
    if ds is None:
        print("\nНе удалось открыть файл с доступными движками. Проверьте формат файла.")
        print("Рекомендации:")
        print("1. Убедитесь, что файл не поврежден")
        print("2. Проверьте, что файл соответствует заявленному формату")
        print("3. Установите дополнительные библиотеки: pip install netcdf4 h5netcdf cfgrib zarr")
        return None

    # Получаем информацию о структуре файла
    info = {
        "file_type": file_type,
        "used_engine": used_engine,
        "file_size_mb": file_size_mb,
        "dimensions": list(ds.dims),
        "variables": list(ds.variables),
        "coordinates": list(ds.coords),
        "global_attrs": {k: str(v) for k, v in ds.attrs.items()}
    }
    print("==============================")
    print(list(ds.variables))
    print("==============================")
    # Проверка наличия временного измерения
    time_dims = [dim for dim in info["dimensions"] if "time" in dim.lower()]
    if time_dims:
        info["time_dim"] = time_dims[0]
        time_var = ds[time_dims[0]]
        info["time_values"] = time_var.values
        info["time_count"] = len(time_var)
        info["time_min"] = str(time_var.values[0])
        info["time_max"] = str(time_var.values[-1])
    else:
        print("ВНИМАНИЕ: В файле не обнаружено измерение времени!")

        # Проверяем другие возможные временные измерения
        for dim in info["dimensions"]:
            if dim in ds and hasattr(ds[dim], 'units'):
                if 'since' in ds[dim].units:
                    print(f"Возможное временное измерение: {dim}")
                    info["time_dim"] = dim
                    time_var = ds[dim]
                    info["time_values"] = time_var.values
                    info["time_count"] = len(time_var)
                    info["time_min"] = str(time_var.values[0])
                    info["time_max"] = str(time_var.values[-1])
                    break

    # Проверка переменных давления
    pressure_keywords = ['pressure', 'msl', 'slp', 'mslp', 'pres']
    pressure_vars = []
    
    for var in info["variables"]:
        var_lower = var.lower()
        if any(keyword in var_lower for keyword in pressure_keywords):
            pressure_vars.append(var)
    
    if pressure_vars:
        info["pressure_vars"] = pressure_vars

        # Исследуем первую переменную давления
        pressure_var = ds[pressure_vars[0]]
        info["pressure_dims"] = list(pressure_var.dims)
        info["pressure_attrs"] = {k: str(v) for k, v in pressure_var.attrs.items()}

        # Проверяем диапазон значений давления
        try:
            pressure_range = float(pressure_var.max().values) - float(pressure_var.min().values)
            info["pressure_range"] = pressure_range
            info["pressure_min"] = float(pressure_var.min().values)
            info["pressure_max"] = float(pressure_var.max().values)

            if pressure_range < 5000:
                info["pressure_unit"] = "гПа (предположительно)"
            else:
                info["pressure_unit"] = "Па (предположительно)"
        except Exception as e:
            print(f"Не удалось вычислить диапазон давления: {e}")
    else:
        print("ВНИМАНИЕ: В файле не обнаружены переменные давления!")
        # Выводим список всех переменных для диагностики
        print("Доступные переменные:")
        for var in info["variables"]:
            if var in ds:
                print(f"  - {var}: {ds[var].dims}")

    # Проверка координат широты и долготы
    lat_keywords = ['lat', 'latitude']
    lon_keywords = ['lon', 'longitude']
    
    lat_vars = []
    lon_vars = []
    
    for var in info["variables"]:
        var_lower = var.lower()
        if any(keyword == var_lower for keyword in lat_keywords):
            lat_vars.append(var)
        if any(keyword == var_lower for keyword in lon_keywords):
            lon_vars.append(var)
    
    if lat_vars:
        info["lat_var"] = lat_vars[0]
    else:
        print("ВНИМАНИЕ: В файле не обнаружена координата широты!")
    
    if lon_vars:
        info["lon_var"] = lon_vars[0]
    else:
        print("ВНИМАНИЕ: В файле не обнаружена координата долготы!")
    
    # Проверка наличия переменных ветра
    wind_keywords = ['wind', 'u10', 'v10', '10m']
    wind_vars = []
    
    for var in info["variables"]:
        var_lower = var.lower()
        if any(keyword in var_lower for keyword in wind_keywords):
            wind_vars.append(var)
    
    if wind_vars:
        info["wind_vars"] = wind_vars
        print(f"Найдены переменные ветра: {', '.join(wind_vars)}")
    else:
        print("ВНИМАНИЕ: В файле не обнаружены переменные ветра!")
    
    # Проверка наличия переменных завихренности
    vort_keywords = ['vort', 'vo', 'rvor']
    vort_vars = []
    
    for var in info["variables"]:
        var_lower = var.lower()
        if any(keyword in var_lower for keyword in vort_keywords):
            vort_vars.append(var)
    
    if vort_vars:
        info["vort_vars"] = vort_vars
        print(f"Найдены переменные завихренности: {', '.join(vort_vars)}")
    else:
        print("ВНИМАНИЕ: В файле не обнаружены переменные завихренности!")

    # Выводим базовую информацию
    print(f"\nРазмерности файла: {', '.join(info['dimensions'])}")
    print(f"Переменные: {', '.join(info['variables'])}")
    print(f"Координаты: {', '.join(info['coordinates'])}")

    if 'time_dim' in info:
        print(f"\nИзмерение времени: {info['time_dim']}")
        print(f"Количество временных шагов: {info['time_count']}")
        print(f"Первый временной шаг: {info['time_min']}")
        print(f"Последний временной шаг: {info['time_max']}")

    if 'pressure_vars' in info:
        print(f"\nПеременные давления: {', '.join(info['pressure_vars'])}")
        if 'pressure_dims' in info:
            print(f"Размерности переменной {info['pressure_vars'][0]}: {', '.join(info['pressure_dims'])}")
        if 'pressure_min' in info and 'pressure_max' in info:
            print(f"Диапазон значений: {info['pressure_min']} - {info['pressure_max']} ({info.get('pressure_unit', 'неизвестные единицы')})")

    # Закрываем датасет
    ds.close()
    return info