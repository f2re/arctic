"""
Модуль процессора данных ERA5 для системы ArcticCyclone.

Специализированные методы обработки данных реанализа ERA5 для исследования
арктических мезоциклонов.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import logging
from datetime import datetime
import scipy.ndimage as ndimage

# Инициализация логгера
logger = logging.getLogger(__name__)

class ERA5Processor:
    """
    Процессор данных реанализа ERA5.
    
    Предоставляет специализированные методы для обработки и анализа
    данных ERA5 в контексте исследования арктических мезоциклонов.
    """
    
    def __init__(self, dataset: xr.Dataset):
        """
        Инициализирует процессор данных ERA5.
        
        Аргументы:
            dataset: Набор данных ERA5 в формате xarray.
        """
        self.dataset = dataset
        self.verify_dataset()
        
    def verify_dataset(self) -> None:
        """
        Проверяет корректность набора данных ERA5.
        
        Вызывает:
            ValueError: Если набор данных не соответствует требованиям.
        """
        # Проверяем наличие необходимых координат
        required_coords = ['time', 'latitude', 'longitude']
        missing_coords = [coord for coord in required_coords if coord not in self.dataset.coords]
        
        if missing_coords:
            raise ValueError(f"Отсутствуют необходимые координаты: {missing_coords}")
        
        # Проверяем наличие данных о давлении
        if 'pressure_level' in self.dataset.dims:
            logger.info("Обнаружены данные на уровнях давления")
        elif 'mean_sea_level_pressure' in self.dataset:
            logger.info("Обнаружены данные о давлении на уровне моря")
        else:
            logger.warning("В наборе данных отсутствует информация о давлении")
        
        # Проверяем тип данных ERA5 по атрибутам
        if 'source' in self.dataset.attrs and self.dataset.attrs['source'] == 'ERA5':
            logger.info("Набор данных подтвержден как ERA5")
        else:
            logger.warning("Набор данных не содержит атрибута source='ERA5'")
            
    def extract_polar_region(self, min_latitude: float = 70.0) -> xr.Dataset:
        """
        Извлекает данные для полярного региона.
        
        Аргументы:
            min_latitude: Минимальная широта для фильтрации (по умолчанию 70° с.ш.).
            
        Возвращает:
            Отфильтрованный набор данных.
        """
        return self.dataset.sel(latitude=slice(min_latitude, 90))
    
    def calculate_geopotential_height(self) -> xr.Dataset:
        """
        Рассчитывает геопотенциальную высоту из геопотенциала.
        
        Возвращает:
            Набор данных с добавленной переменной геопотенциальной высоты.
        """
        if 'geopotential' not in self.dataset:
            raise ValueError("Отсутствует переменная 'geopotential' в наборе данных")
        
        result = self.dataset.copy()
        
        # Проверяем единицы измерения
        if result.geopotential.max() > 100000:  # Предполагаем, что единицы - м²/с²
            # Конвертируем геопотенциал в геопотенциальную высоту (м)
            result['geopotential_height'] = result.geopotential / 9.80665
            result.geopotential_height.attrs['units'] = 'm'
            result.geopotential_height.attrs['long_name'] = 'Геопотенциальная высота'
        else:
            # Если значения маленькие, возможно, это уже геопотенциальная высота в км
            result['geopotential_height'] = result.geopotential * 1000  # км -> м
            result.geopotential_height.attrs['units'] = 'm'
            result.geopotential_height.attrs['long_name'] = 'Геопотенциальная высота'
            
        return result
    
    def calculate_relative_vorticity(self) -> xr.Dataset:
        """
        Рассчитывает относительную завихренность, если она отсутствует в данных.
        
        Возвращает:
            Набор данных с добавленной переменной относительной завихренности.
        """
        # Проверяем, есть ли уже завихренность в данных
        if 'vorticity' in self.dataset or 'vo' in self.dataset:
            return self.dataset
        
        # Проверяем наличие компонентов ветра
        wind_components = [
            ('u_component_of_wind', 'v_component_of_wind'),
            ('u', 'v'),
            ('10m_u_component_of_wind', '10m_v_component_of_wind'),
            ('10u', '10v')
        ]
        
        u_var, v_var = None, None
        for u, v in wind_components:
            if u in self.dataset and v in self.dataset:
                u_var, v_var = u, v
                break
                
        if not u_var or not v_var:
            raise ValueError("Отсутствуют компоненты ветра для расчета завихренности")
        
        result = self.dataset.copy()
        
        # Вычисляем сетку координат в метрах
        lat_rad = np.radians(result.latitude)
        R = 6371000  # радиус Земли в метрах
        
        # Рассчитываем расстояния между точками сетки
        dy = R * np.gradient(lat_rad, axis=0)
        dx = R * np.cos(lat_rad) * np.gradient(np.radians(result.longitude), axis=1)
        
        # Рассчитываем градиенты компонентов ветра
        dudx = np.gradient(result[u_var].values, axis=2) / dx[..., np.newaxis]
        dvdy = np.gradient(result[v_var].values, axis=1) / dy[..., np.newaxis]
        dudy = np.gradient(result[u_var].values, axis=1) / dy[..., np.newaxis]
        dvdx = np.gradient(result[v_var].values, axis=2) / dx[..., np.newaxis]
        
        # Вычисляем завихренность как curl(V) = dv/dx - du/dy
        vorticity = dvdx - dudy
        
        # Создаем переменную с такими же размерностями, как у входных данных
        coords = {dim: result[dim] for dim in result[u_var].dims}
        result['vorticity'] = xr.DataArray(vorticity, coords=coords, dims=result[u_var].dims)
        
        # Добавляем атрибуты
        result.vorticity.attrs['units'] = 's^-1'
        result.vorticity.attrs['long_name'] = 'Относительная завихренность'
        
        return result
    
    def calculate_thermal_front_parameter(self) -> xr.Dataset:
        """
        Рассчитывает параметр термического фронта (TFP).
        
        Возвращает:
            Набор данных с добавленной переменной TFP.
        """
        if 'temperature' not in self.dataset and 't' not in self.dataset:
            raise ValueError("Отсутствует переменная температуры в наборе данных")
            
        temp_var = 'temperature' if 'temperature' in self.dataset else 't'
        
        result = self.dataset.copy()
        
        # Выбираем уровень 850 гПа, если доступны данные на уровнях давления
        if 'pressure_level' in result.dims:
            if 850 in result.pressure_level.values:
                temp = result[temp_var].sel(pressure_level=850)
            else:
                # Берем ближайший доступный уровень к 850 гПа
                level = min(result.pressure_level.values, key=lambda x: abs(x - 850))
                logger.warning(f"Уровень 850 гПа недоступен, используется ближайший: {level} гПа")
                temp = result[temp_var].sel(pressure_level=level)
        else:
            # Используем приземную температуру, если нет данных по уровням давления
            temp = result[temp_var]
            
        # Рассчитываем градиент температуры
        grad_temp = xr.apply_ufunc(
            lambda x: np.gradient(x, axis=(0, 1)),
            temp,
            input_core_dims=[['latitude', 'longitude']],
            output_core_dims=[['latitude', 'longitude'], ['latitude', 'longitude']],
            vectorize=True
        )
        
        # Вычисляем величину градиента температуры
        grad_temp_magnitude = np.sqrt(grad_temp[0]**2 + grad_temp[1]**2)
        
        # Рассчитываем градиент величины градиента температуры
        grad_temp_mag_grad = xr.apply_ufunc(
            lambda x: np.gradient(x, axis=(0, 1)),
            grad_temp_magnitude,
            input_core_dims=[['latitude', 'longitude']],
            output_core_dims=[['latitude', 'longitude'], ['latitude', 'longitude']],
            vectorize=True
        )
        
        # Нормализуем градиент температуры
        norm_grad_temp = [grad_temp[0] / grad_temp_magnitude, grad_temp[1] / grad_temp_magnitude]
        
        # Вычисляем TFP как скалярное произведение нормализованного градиента и градиента величины
        tfp = -(norm_grad_temp[0] * grad_temp_mag_grad[0] + norm_grad_temp[1] * grad_temp_mag_grad[1])
        
        # Добавляем TFP в результат
        result['thermal_front_parameter'] = tfp
        result.thermal_front_parameter.attrs['units'] = 'K/m²'
        result.thermal_front_parameter.attrs['long_name'] = 'Параметр термического фронта'
        
        return result
    
    def calculate_potential_vorticity(self) -> xr.Dataset:
        """
        Рассчитывает потенциальную завихренность (PV).
        
        Возвращает:
            Набор данных с добавленной переменной потенциальной завихренности.
        """
        if 'pressure_level' not in self.dataset.dims:
            raise ValueError("Для расчета потенциальной завихренности требуются данные на уровнях давления")
            
        if 'temperature' not in self.dataset and 't' not in self.dataset:
            raise ValueError("Отсутствует переменная температуры в наборе данных")
            
        # Проверяем наличие компонентов ветра
        if ('u_component_of_wind' not in self.dataset and 'u' not in self.dataset) or \
           ('v_component_of_wind' not in self.dataset and 'v' not in self.dataset):
            raise ValueError("Отсутствуют компоненты ветра для расчета потенциальной завихренности")
            
        result = self.dataset.copy()
        
        # Определяем имена переменных
        temp_var = 'temperature' if 'temperature' in result else 't'
        u_var = 'u_component_of_wind' if 'u_component_of_wind' in result else 'u'
        v_var = 'v_component_of_wind' if 'v_component_of_wind' in result else 'v'
        
        # Константы
        g = 9.80665  # ускорение свободного падения, м/с²
        omega = 7.292e-5  # угловая скорость вращения Земли, рад/с
        R = 287.0  # газовая постоянная для сухого воздуха, Дж/(кг·К)
        cp = 1004.0  # удельная теплоемкость воздуха при постоянном давлении, Дж/(кг·К)
        
        # Вычисляем параметр Кориолиса
        f = 2 * omega * np.sin(np.radians(result.latitude))
        
        # Рассчитываем потенциальную температуру
        # θ = T * (1000/p)^(R/cp)
        p_levels = result.pressure_level.values
        theta = result[temp_var] * (1000 / p_levels[:, np.newaxis, np.newaxis])**(R/cp)
        
        # Вертикальный градиент потенциальной температуры
        dtheta_dp = theta.differentiate('pressure_level')
        
        # Относительная завихренность
        if 'vorticity' in result:
            rel_vort = result.vorticity
        else:
            # Рассчитываем относительную завихренность
            dudy = result[u_var].differentiate('latitude')
            dvdx = result[v_var].differentiate('longitude')
            
            # Корректируем dvdx для учета сферичности Земли
            corr_factor = 1 / (R * np.cos(np.radians(result.latitude)))
            dvdx = dvdx * corr_factor
            
            rel_vort = dvdx - dudy
        
        # Абсолютная завихренность (ζ_a = ζ + f)
        abs_vort = rel_vort + f
        
        # Потенциальная завихренность (PV = -g * ζ_a * ∂θ/∂p)
        pv = -g * abs_vort * dtheta_dp
        
        # Добавляем PV в результат
        result['potential_vorticity'] = pv
        result.potential_vorticity.attrs['units'] = 'K·m²/(kg·s)'
        result.potential_vorticity.attrs['long_name'] = 'Потенциальная завихренность'
        
        return result
    
    def calculate_wind_speed(self) -> xr.Dataset:
        """
        Рассчитывает скорость ветра на основе компонентов.
        
        Возвращает:
            Набор данных с добавленной переменной скорости ветра.
        """
        # Проверяем наличие компонентов ветра
        wind_components = [
            ('u_component_of_wind', 'v_component_of_wind'),
            ('u', 'v'),
            ('10m_u_component_of_wind', '10m_v_component_of_wind'),
            ('10u', '10v')
        ]
        
        result = self.dataset.copy()
        
        for u_var, v_var in wind_components:
            if u_var in result and v_var in result:
                # Рассчитываем скорость ветра как sqrt(u² + v²)
                if 'pressure_level' in result.dims and '10m' not in u_var:
                    # Для ветра на уровнях давления
                    wind_speed_name = 'wind_speed'
                    suffix = ''
                else:
                    # Для приземного ветра
                    wind_speed_name = '10m_wind_speed'
                    suffix = '_10m'
                
                result[wind_speed_name] = np.sqrt(result[u_var]**2 + result[v_var]**2)
                result[wind_speed_name].attrs['units'] = 'm/s'
                result[wind_speed_name].attrs['long_name'] = f'Скорость ветра{suffix}'
                
                logger.info(f"Рассчитана скорость ветра из компонентов {u_var} и {v_var}")
        
        return result
    
    def calculate_slp_anomaly(self, climatology: Optional[xr.DataArray] = None) -> xr.Dataset:
        """
        Рассчитывает аномалию давления на уровне моря.
        
        Аргументы:
            climatology: Климатологическое среднее давление на уровне моря.
                         Если None, вычисляется из имеющихся данных.
            
        Возвращает:
            Набор данных с добавленной переменной аномалии давления.
        """
        # Определяем переменную давления
        if 'mean_sea_level_pressure' in self.dataset:
            slp_var = 'mean_sea_level_pressure'
        elif 'msl' in self.dataset:
            slp_var = 'msl'
        else:
            raise ValueError("Отсутствует переменная давления на уровне моря")
        
        result = self.dataset.copy()
        
        # Если климатология не предоставлена, рассчитываем из имеющихся данных
        if climatology is None:
            # Рассчитываем среднее по времени
            slp_mean = result[slp_var].mean(dim='time')
            logger.warning("Климатология рассчитана из текущего набора данных, что может быть неточно")
        else:
            slp_mean = climatology
        
        # Рассчитываем аномалию
        result['slp_anomaly'] = result[slp_var] - slp_mean
        result.slp_anomaly.attrs['units'] = result[slp_var].attrs.get('units', 'hPa')
        result.slp_anomaly.attrs['long_name'] = 'Аномалия давления на уровне моря'
        
        return result
    
    def calculate_thermal_structure(self) -> xr.Dataset:
        """
        Определяет термическую структуру циклонов (теплая или холодная).
        
        Возвращает:
            Набор данных с добавленными переменными термической структуры.
        """
        if 'pressure_level' not in self.dataset.dims:
            raise ValueError("Для определения термической структуры требуются данные на уровнях давления")
            
        if 'temperature' not in self.dataset and 't' not in self.dataset:
            raise ValueError("Отсутствует переменная температуры в наборе данных")
        
        temp_var = 'temperature' if 'temperature' in self.dataset else 't'
        
        # Проверяем наличие необходимых уровней
        required_levels = [500, 850]
        available_levels = self.dataset.pressure_level.values
        
        if not all(level in available_levels for level in required_levels):
            raise ValueError(f"Требуются уровни давления {required_levels}, доступны {available_levels}")
        
        result = self.dataset.copy()
        
        # Рассчитываем толщину слоя 500-850 гПа
        if 'geopotential_height' in result:
            z500 = result.geopotential_height.sel(pressure_level=500)
            z850 = result.geopotential_height.sel(pressure_level=850)
        elif 'geopotential' in result:
            # Преобразуем геопотенциал в высоту
            z500 = result.geopotential.sel(pressure_level=500) / 9.80665
            z850 = result.geopotential.sel(pressure_level=850) / 9.80665
        else:
            raise ValueError("Отсутствуют данные о геопотенциале или геопотенциальной высоте")
        
        # Толщина слоя 500-850 гПа
        result['thickness_500_850'] = z500 - z850
        result.thickness_500_850.attrs['units'] = 'm'
        result.thickness_500_850.attrs['long_name'] = 'Толщина слоя 500-850 гПа'
        
        # Температурная аномалия на 500 и 850 гПа
        t500 = result[temp_var].sel(pressure_level=500)
        t850 = result[temp_var].sel(pressure_level=850)
        
        # Рассчитываем зональные средние
        t500_zonal_mean = t500.mean(dim='longitude')
        t850_zonal_mean = t850.mean(dim='longitude')
        
        # Рассчитываем аномалии
        result['t500_anomaly'] = t500 - t500_zonal_mean
        result.t500_anomaly.attrs['units'] = 'K'
        result.t500_anomaly.attrs['long_name'] = 'Аномалия температуры на 500 гПа'
        
        result['t850_anomaly'] = t850 - t850_zonal_mean
        result.t850_anomaly.attrs['units'] = 'K'
        result.t850_anomaly.attrs['long_name'] = 'Аномалия температуры на 850 гПа'
        
        return result
    
    def detect_cyclone_centers(self, 
                              min_latitude: float = 70.0,
                              slp_var: str = None,
                              min_slp_gradient: float = 0.5,  # гПа/100км
                              min_vorticity: float = 1e-5,  # с^-1
                              smooth_sigma: float = 1.0) -> pd.DataFrame:
        """
        Обнаруживает центры циклонов на основе минимумов давления и максимумов завихренности.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения циклонов.
            slp_var: Имя переменной давления на уровне моря. Если None, определяется автоматически.
            min_slp_gradient: Минимальный градиент давления для идентификации циклона.
            min_vorticity: Минимальное значение завихренности для идентификации циклона.
            smooth_sigma: Параметр сглаживания для фильтрации шума.
            
        Возвращает:
            DataFrame с информацией о центрах циклонов.
        """
        # Определяем переменную давления
        if slp_var is None:
            if 'mean_sea_level_pressure' in self.dataset:
                slp_var = 'mean_sea_level_pressure'
            elif 'msl' in self.dataset:
                slp_var = 'msl'
            else:
                raise ValueError("Отсутствует переменная давления на уровне моря")
                
        # Подготавливаем данные
        polar_data = self.dataset.sel(latitude=slice(min_latitude, 90))
        
        # Добавляем завихренность, если она отсутствует
        if 'vorticity' not in polar_data and 'vo' not in polar_data:
            processor = ERA5Processor(polar_data)
            polar_data = processor.calculate_relative_vorticity()
            
        vo_var = 'vorticity' if 'vorticity' in polar_data else 'vo'
        
        # Сглаживаем поля для уменьшения шума
        slp_smoothed = ndimage.gaussian_filter(polar_data[slp_var].values, sigma=smooth_sigma)
        vo_smoothed = ndimage.gaussian_filter(polar_data[vo_var].values, sigma=smooth_sigma)
        
        # Ищем минимумы давления
        slp_minima = np.zeros_like(slp_smoothed, dtype=bool)
        
        for t in range(slp_smoothed.shape[0]):
            # Для каждого временного шага
            slp_t = slp_smoothed[t]
            
            # Находим локальные минимумы в скользящем окне 3x3
            slp_min_filter = ndimage.minimum_filter(slp_t, size=3)
            local_minima = (slp_t == slp_min_filter)
            
            # Вычисляем градиент давления
            slp_grad_y, slp_grad_x = np.gradient(slp_t)
            slp_grad_magnitude = np.sqrt(slp_grad_y**2 + slp_grad_x**2)
            
            # Отбираем минимумы с достаточным градиентом
            valid_minima = local_minima & (slp_grad_magnitude > min_slp_gradient)
            
            slp_minima[t] = valid_minima
        
        # Ищем максимумы завихренности
        vo_maxima = np.zeros_like(vo_smoothed, dtype=bool)
        
        for t in range(vo_smoothed.shape[0]):
            # Для каждого временного шага
            vo_t = vo_smoothed[t]
            
            # Находим локальные максимумы в скользящем окне 3x3
            vo_max_filter = ndimage.maximum_filter(vo_t, size=3)
            local_maxima = (vo_t == vo_max_filter) & (vo_t > min_vorticity)
            
            vo_maxima[t] = local_maxima
        
        # Собираем информацию о центрах циклонов
        cyclone_centers = []
        
        times = polar_data.time.values
        lats = polar_data.latitude.values
        lons = polar_data.longitude.values
        
        for t in range(len(times)):
            for i in range(len(lats)):
                for j in range(len(lons)):
                    if slp_minima[t, i, j]:
                        # Нашли минимум давления
                        cyclone_info = {
                            'time': times[t],
                            'latitude': lats[i],
                            'longitude': lons[j],
                            'pressure': float(polar_data[slp_var][t, i, j].values),
                            'type': 'pressure_minimum'
                        }
                        
                        # Проверяем, совпадает ли с максимумом завихренности
                        if vo_maxima[t, i, j]:
                            cyclone_info['vorticity'] = float(polar_data[vo_var][t, i, j].values)
                            cyclone_info['type'] = 'pressure_vorticity_match'
                        
                        cyclone_centers.append(cyclone_info)
                    elif vo_maxima[t, i, j] and not slp_minima[t, i, j]:
                        # Нашли только максимум завихренности
                        cyclone_info = {
                            'time': times[t],
                            'latitude': lats[i],
                            'longitude': lons[j],
                            'vorticity': float(polar_data[vo_var][t, i, j].values),
                            'type': 'vorticity_maximum'
                        }
                        
                        # Находим ближайшее значение давления
                        cyclone_info['pressure'] = float(polar_data[slp_var][t, i, j].values)
                        
                        cyclone_centers.append(cyclone_info)
        
        return pd.DataFrame(cyclone_centers)