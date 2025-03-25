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


def download_era5_data_extended(start_date, end_date, data_dir, output_file='era5_arctic_data.nc'):
    """
    Загрузка расширенного набора данных ERA5 для арктического региона.
    
    Параметры:
    ----------
    start_date, end_date : str
        Начальная и конечная даты в формате 'YYYY-MM-DD'
    data_dir : str
        Директория для сохранения данных
    output_file : str
        Имя выходного файла

    Возвращает:
    -----------
    str
        Путь к загруженному файлу
    """
    # Реализация без изменений из оригинального файла
    import netCDF4
    import cfgrib
    
    # Проверка наличия cdsapi
    import cdsapi

    # Полный путь для сохранения файла
    import os
    output_path = os.path.join(data_dir, output_file)

    # Проверяем существование файла перед загрузкой
    if os.path.exists(output_path):
        print(f"Файл данных уже существует: {output_path}")
        use_existing = input("Использовать существующий файл? (y/n): ")
        if use_existing.lower() == 'y':
            print(f"Используется существующий файл: {output_path}")
            return output_path

    print(f"Загрузка данных ERA5 для периода {start_date} - {end_date}...")
    print(f"Файл будет сохранен в: {output_path}")

    # Преобразуем строки дат в объекты datetime
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Формируем списки для параметров запроса
    years = []
    months = []
    days = []

    # Генерируем список дат в указанном диапазоне
    current_dt = start_dt
    while current_dt <= end_dt:
        years.append(str(current_dt.year))
        months.append(f"{current_dt.month:02d}")
        days.append(f"{current_dt.day:02d}")
        current_dt += timedelta(days=1)

    # Удаляем дубликаты, сохраняя порядок
    years = list(dict.fromkeys(years))
    months = list(dict.fromkeys(months))
    days = list(dict.fromkeys(days))

    try:
        # Создаем клиент CDS API
        client = cdsapi.Client()
        
        # Временные файлы для разных наборов данных
        surface_file = os.path.join(data_dir, "temp_surface.nc")
        pressure_file = os.path.join(data_dir, "temp_pressure.nc")
        vorticity_file = os.path.join(data_dir, "temp_vorticity.nc")
        # Запрос для приземных данных с корректными именами переменных
        print("\nЗагрузка приземных данных...")
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'mean_sea_level_pressure',  # Приземное давление
                    '10m_u_component_of_wind',  # U-компонента ветра на 10м
                    '10m_v_component_of_wind',  # V-компонента ветра на 10м
                ],
                'year': years,
                'month': months,
                'day': days,
                'time': ['00:00', '12:00'],
                'area': [90, -180, 70, 180],    # Северный полярный регион выше 70°N
                'format': 'netcdf',             # Формат выходных данных
            },
            surface_file
        )
        print(f"Приземные данные успешно загружены в {surface_file}")
        
        # Запрос для данных на уровне давления 700 гПа
        print("\nЗагрузка данных на уровне давления...")
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'temperature',      # Температура
                    'relative_humidity' # Относительная влажность
                ],
                'pressure_level': '700',
                'year': years,
                'month': months,
                'day': days,
                'time': ['00:00', '12:00'],
                'area': [90, -180, 70, 180],
                'format': 'netcdf',
            },
            pressure_file
        )
        print(f"Данные на уровне давления успешно загружены в {pressure_file}")
        
        # Отдельный запрос для завихренности на уровне 850 гПа
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'vo',   # Завихренность
                ],
                'pressure_level': '850',
                'year': years,
                'month': months,
                'day': days,
                'time': ['00:00', '12:00'],
                'area': [90, -180, 70, 180],
                'format': 'netcdf',
            },
            vorticity_file
        )
        print(f"Данные на уровне 850 гПа успешно загружены в {vorticity_file}")
        # Объединяем наборы данных с использованием xarray
        print("\nОбъединение наборов данных...")
        import xarray as xr
        
        # Явно указываем движок netCDF4 при открытии файлов
        ds_surface = xr.open_dataset(surface_file, engine='netcdf4')
        ds_pressure = xr.open_dataset(pressure_file, engine='netcdf4')
        ds_vorticity = xr.open_dataset(vorticity_file, engine='netcdf4')
        # Обеспечиваем совместимость координат
        ds_pressure = ds_pressure.assign_coords(
            longitude=ds_pressure.longitude,
            latitude=ds_pressure.latitude
        )
        ds_vorticity = ds_vorticity.assign_coords(    
            longitude=ds_vorticity.longitude,
            latitude=ds_vorticity.latitude
        )
        # Объединяем в один набор данных и сохраняем
        ds_combined = xr.merge([ds_surface, ds_pressure, ds_vorticity])
        ds_combined.to_netcdf(output_path)
        
        # Вычисляем скорость ветра из компонент U и V
        if '10m_u_component_of_wind' in ds_combined and '10m_v_component_of_wind' in ds_combined:
            print("Вычисление скорости ветра из компонент U и V...")
            import numpy as np
            
            # Создаем новый датасет для сохранения
            ds_with_wind = ds_combined.copy()
            
            # Вычисляем скорость ветра
            u10 = ds_combined['10m_u_component_of_wind']
            v10 = ds_combined['10m_v_component_of_wind']
            wind_speed = np.sqrt(u10**2 + v10**2)
            
            # Добавляем новую переменную в датасет
            ds_with_wind['10m_wind_speed'] = wind_speed
            
            # Сохраняем обновленный датасет
            ds_with_wind.to_netcdf(output_path)
            print(f"Добавлена вычисленная переменная: скорость ветра")
            
            # Используем обновленный датасет для дальнейшей работы
            ds_combined = ds_with_wind
        
        # Закрываем датасеты
        ds_surface.close()
        ds_pressure.close()
        ds_vorticity.close()
        # Удаляем временные файлы
        try:
            os.remove(surface_file)
            os.remove(pressure_file)
            os.remove(vorticity_file)
            print("Временные файлы удалены")
        except Exception as e:
            print(f"Предупреждение: не удалось удалить временные файлы: {e}")
        
        print(f"\nДанные успешно загружены и объединены в файл: {output_path}")
        return output_path

    except Exception as e:
        print(f"\nОшибка при загрузке данных: {e}")
        
        # Предлагаем альтернативные действия
        print("\nАльтернативные действия:")
        print("1. Проверьте правильность API ключа в файле .cdsapirc")
        print("2. Проверьте соединение с интернетом")
        print("3. Возможно, сервер CDS перегружен, попробуйте позже")
        print("4. Используйте веб-интерфейс CDS для загрузки данных")
        
        # Предлагаем использовать локальный файл
        use_local = input("\nИспользовать существующий файл данных на Google Drive? (y/n): ")
        if use_local.lower() == 'y':
            # Показываем список nc-файлов в папке data
            nc_files = [f for f in os.listdir(data_dir) if f.endswith('.nc')]

            if nc_files:
                print("\nДоступные файлы .nc в директории данных:")
                for i, file in enumerate(nc_files):
                    print(f"{i+1}. {file}")

                choice = input("\nВыберите номер файла (или нажмите Enter для ввода другого пути): ")

                if choice.isdigit() and 1 <= int(choice) <= len(nc_files):
                    file_path = os.path.join(data_dir, nc_files[int(choice)-1])
                else:
                    file_path = input("Введите полный путь к файлу с данными ERA5: ")
            else:
                print("В директории данных не найдены файлы .nc")
                file_path = input("Введите полный путь к файлу с данными ERA5: ")

            if os.path.exists(file_path):
                return file_path
        return None

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