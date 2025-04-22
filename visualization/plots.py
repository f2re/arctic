# Функции визуализации
"""
Функции для визуализации результатов обнаружения циклонов.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from detection.algorithms import detect_cyclones_improved

def visualize_detection_methods_effect(cyclones, detection_methods, output_dir, file_prefix=''):
    """
    Визуализирует эффект применения различных методов обнаружения циклонов.
    Создает набор графиков, показывающих распределения параметров обнаруженных циклонов
    в зависимости от применяемых критериев фильтрации.
    
    Параметры:
    ----------
    cyclones : list
        Список обнаруженных циклонов
    detection_methods : list
        Список использованных методов обнаружения
    output_dir : str
        Директория для сохранения результатов
    file_prefix : str
        Префикс для имен файлов
    """
    if not cyclones:
        print("Нет данных для визуализации эффекта методов обнаружения")
        return
    
    # Создаем DataFrame для облегчения работы с данными
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    df = pd.DataFrame(cyclones)
    
    # Определяем число графиков на основе доступных данных
    plot_columns = []
    if 'pressure' in df.columns:
        plot_columns.append(('pressure', 'Давление (гПа)', 'lightblue'))
    if 'depth' in df.columns:
        plot_columns.append(('depth', 'Глубина (гПа)', 'lightgreen'))
    if 'gradient' in df.columns and 'gradient' in detection_methods:
        plot_columns.append(('gradient', 'Градиент давления (гПа/градус)', 'salmon'))
    if 'max_wind' in df.columns and 'wind_speed' in detection_methods:
        plot_columns.append(('max_wind', 'Скорость ветра (м/с)', 'gold'))
    if 'max_vorticity' in df.columns and 'vorticity' in detection_methods:
        plot_columns.append(('max_vorticity', 'Завихренность (с⁻¹)', 'mediumpurple'))
    if 'radius' in df.columns:
        plot_columns.append(('radius', 'Радиус (км)', 'lightcoral'))
    
    # Вычисляем количество строк и столбцов для сетки графиков
    n_plots = len(plot_columns)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Создаем график для каждого параметра
    for i, (col, label, color) in enumerate(plot_columns):
        if i < len(axes):
            ax = axes[i]
            
            # Особая обработка для завихренности (масштабирование для улучшения читаемости)
            if col == 'max_vorticity':
                values = df[col] * 1e6  # Преобразуем в 10⁻⁶ с⁻¹
                xlabel = 'Завихренность (10⁻⁶ с⁻¹)'
            else:
                values = df[col]
                xlabel = label
            
            # Создаем гистограмму
            ax.hist(values, bins=20, color=color, edgecolor='black', alpha=0.7)
            
            # Добавляем вертикальную линию для порогового значения, если это применимо
            if col == 'pressure' and 'pressure_minima' in detection_methods:
                threshold = next((c['min_pressure_threshold'] for c in df['cyclone_params'].unique() 
                                 if hasattr(c, 'get') and c.get('min_pressure_threshold')), None)
                if threshold:
                    ax.axvline(x=threshold, color='red', linestyle='--', 
                              label=f'Порог: {threshold} гПа')
                    ax.legend()
            
            elif col == 'depth' and 'pressure_minima' in detection_methods:
                threshold = next((c['min_depth'] for c in df['cyclone_params'].unique() 
                                 if hasattr(c, 'get') and c.get('min_depth')), None)
                if threshold:
                    ax.axvline(x=threshold, color='red', linestyle='--', 
                              label=f'Порог: {threshold} гПа')
                    ax.legend()
            
            elif col == 'gradient' and 'gradient' in detection_methods:
                threshold = next((c['pressure_gradient_threshold'] for c in df['cyclone_params'].unique() 
                                 if hasattr(c, 'get') and c.get('pressure_gradient_threshold')), None)
                if threshold:
                    ax.axvline(x=threshold, color='red', linestyle='--', 
                              label=f'Порог: {threshold} гПа/градус')
                    ax.legend()
            
            elif col == 'max_wind' and 'wind_speed' in detection_methods:
                threshold = next((c['min_wind_speed'] for c in df['cyclone_params'].unique() 
                                 if hasattr(c, 'get') and c.get('min_wind_speed')), None)
                if threshold:
                    ax.axvline(x=threshold, color='red', linestyle='--', 
                              label=f'Порог: {threshold} м/с')
                    ax.legend()
            
            elif col == 'max_vorticity' and 'vorticity' in detection_methods:
                threshold = next((c['min_vorticity'] for c in df['cyclone_params'].unique() 
                                 if hasattr(c, 'get') and c.get('min_vorticity')), None)
                if threshold:
                    ax.axvline(x=threshold * 1e6, color='red', linestyle='--', 
                              label=f'Порог: {threshold * 1e6:.1f} · 10⁻⁶ с⁻¹')
                    ax.legend()
            
            ax.set_title(label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Количество циклонов')
            ax.grid(alpha=0.3)
    
    # Скрываем пустые графики
    for i in range(len(plot_columns), len(axes)):
        axes[i].set_visible(False)
    
    # Добавляем общий заголовок с информацией о методах обнаружения
    plt.suptitle(f"Распределение параметров циклонов, обнаруженных с использованием методов:\n{', '.join(detection_methods)}", 
                fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{file_prefix}detection_methods_effect.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем круговую диаграмму распределения причин отклонения потенциальных циклонов
    if 'rejected_centers' in locals() and rejected_centers:
        # Подсчитываем причины отклонения
        reasons = {}
        for center in rejected_centers:
            reason = center.get("reason", "unknown")
            if reason in reasons:
                reasons[reason] += 1
            else:
                reasons[reason] = 1
        
        # Словарь понятных названий причин
        reason_labels = {
            "small_size": "Малый размер области",
            "high_pressure": "Высокое давление",
            "shallow_depth": "Недостаточная глубина",
            "no_closed_contour": "Отсутствие замкнутых изобар",
            "weak_gradient": "Слабый градиент давления",
            "low_latitude": "Широта ниже 70°N",
            "invalid_size": "Размер вне допустимого диапазона",
            "weak_wind": "Слабый ветер",
            "weak_vorticity": "Слабая завихренность",
            "unknown": "Неизвестная причина"
        }
        
        # Создаем круговую диаграмму
        plt.figure(figsize=(12, 10))
        
        # Подготовка данных для диаграммы
        labels = []
        sizes = []
        
        for reason, count in reasons.items():
            label = reason_labels.get(reason, reason)
            labels.append(f"{label} ({count})")
            sizes.append(count)
        
        # Создаем круговую диаграмму
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
              shadow=True, explode=[0.05] * len(sizes))
        plt.axis('equal')  # Чтобы круг был правильной формы
        
        plt.title(f"Причины отклонения потенциальных циклонических систем при использовании методов:\n{', '.join(detection_methods)}")
        
        rejection_file = os.path.join(output_dir, f"{file_prefix}rejection_reasons_distribution.png")
        plt.savefig(rejection_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Диаграмма распределения причин отклонения сохранена: {rejection_file}")
    
    print(f"Визуализация эффекта методов обнаружения сохранена в {output_dir}")


def visualize_cyclones_with_diagnostics(ds, time_idx, time_dim, pressure_var, lat_var, lon_var,
                                      output_dir, cyclone_params=None, save_diagnostic=True, file_prefix='',
                                      detection_methods=None):
    """
    Оптимизированная функция визуализации обнаруженных циклонов с расширенной диагностикой.
    
    Создает многопанельную визуализацию, включающую:
    1. Поле приземного давления с отмеченными центрами циклонов
    2. Лапласиан давления с указанием порогового значения
    3. Градиент давления с указанием порогового значения
    4. Скорость ветра (если доступна и выбрана) с указанием порогового значения
    5. Относительная завихренность (если доступна и выбрана) с указанием порогового значения
    
    Параметры:
    ----------
    ds : xarray.Dataset
        Набор данных с полем приземного давления и другими переменными
    time_idx : int
        Индекс временного шага
    time_dim : str
        Имя измерения времени
    pressure_var : str
        Имя переменной давления
    lat_var : str
        Имя переменной широты
    lon_var : str
        Имя переменной долготы
    output_dir : str
        Директория для сохранения изображений
    cyclone_params : dict, optional
        Словарь с параметрами алгоритма
    save_diagnostic : bool, default=True
        Флаг для сохранения диагностических изображений
    file_prefix : str, default=''
        Префикс для имен файлов
    detection_methods : list, optional
        Список методов обнаружения циклонов для визуализации.
        Если None, визуализируются все доступные методы.
    
    Возвращает:
    -----------
    tuple
        (список обнаруженных центров циклонов, найдены ли циклоны, диагностические данные)
    """
    # Проверяем параметры
    if cyclone_params is None:
        from detection.parameters import get_cyclone_params
        cyclone_params = get_cyclone_params('mesoscale')
    
    # Создаем директорию для сохранения изображений, если она не существует
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Получаем текущее время из датасета
    if time_dim in ds.dims:
        current_time = ds[time_dim].values[time_idx]
        # Преобразуем время в строку, удобную для имени файла
        if isinstance(current_time, np.datetime64):
            time_str = np.datetime_as_string(current_time, unit='h').replace(':', '-')
            time_display = str(current_time)[:19]  # Более читабельный формат для отображения
        else:
            # Если время не datetime64, используем просто индекс
            time_str = f"step_{time_idx}"
            time_display = f"Шаг {time_idx}"
    else:
        # Если нет временного измерения, используем индекс
        time_str = f"step_{time_idx}"
        time_display = f"Шаг {time_idx}"

    # Обнаруживаем циклоны с улучшенным алгоритмом
    try:
        pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
            ds, time_idx, time_dim, pressure_var, lat_var, lon_var, cyclone_params)

        # Сохраняем изображение если найдены циклоны или включена диагностика
        if cyclones_found or save_diagnostic:
            # Получаем сетку широты и долготы
            lat_values = ds[lat_var].values
            lon_values = ds[lon_var].values

            # Преобразуем в гПа для отображения
            pressure_hpa = pressure / 100.0 if np.max(pressure) > 10000 else pressure

            # Определяем количество панелей на основе доступных данных и выбранных методов
            has_wind_data = diagnostic_data.get("has_wind_data", False)
            has_vorticity_data = diagnostic_data.get("has_vorticity_data", False)
            
            num_panels = 3  # Базовые панели: давление, лапласиан, градиент
            if has_wind_data and (detection_methods is None or 'wind_speed' in detection_methods):
                num_panels += 1
            if has_vorticity_data and (detection_methods is None or 'vorticity' in detection_methods):
                num_panels += 1
            
            # Создаем многопанельную визуализацию
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from matplotlib.gridspec import GridSpec
            
            # Определяем размер фигуры в зависимости от числа панелей
            figwidth = min(20, 5 * num_panels)  # Ограничиваем ширину
            fig = plt.figure(figsize=(figwidth, 12))
            
            # Создаем сетку с одинаковыми пропорциями для всех панелей
            gs = GridSpec(1, num_panels, width_ratios=[1] * num_panels)

            # Оптимизация: создаем сетку координат однократно
            lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

            # 1. Панель давления
            ax1 = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo(central_longitude=0))
            ax1.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
            ax1.add_feature(cfeature.COASTLINE)
            ax1.add_feature(cfeature.BORDERS, linestyle=':')
            ax1.gridlines(draw_labels=True)

            # Отображаем поле давления с контурами
            levels = np.arange(950, 1050, 2)
            cs1 = ax1.contourf(lon_grid, lat_grid, pressure_hpa, levels=levels,
                             transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.7)

            # Добавляем изолинии давления
            contour_levels = np.arange(950, 1050, 5)
            contours1 = ax1.contour(lon_grid, lat_grid, pressure_hpa, levels=contour_levels,
                                  colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
            plt.clabel(contours1, inline=True, fontsize=8, fmt='%1.0f')

            # Добавляем информацию о циклонах и подписываем
            if cyclones_found:
                for cyclone_data in cyclone_centers:
                    lat, lon, pressure, depth, gradient = cyclone_data[:5]
                    radius_km = cyclone_data[5] if len(cyclone_data) > 5 else None

                    ax1.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())

                    # Информация для подписи
                    label_text = f'{pressure:.0f} гПа\n{depth:.1f} гПа'
                    if radius_km is not None:
                        label_text += f'\n{radius_km:.0f} км'

                    ax1.text(lon, lat, label_text,
                           transform=ccrs.PlateCarree(), fontsize=8,
                           ha='left', va='bottom', color='red',
                           bbox=dict(facecolor='white', alpha=0.7))

            # Добавляем цветовую шкалу и заголовок
            cb1 = plt.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05, label='Давление (гПа)')
            ax1.set_title(f"Приземное давление (гПа)\n{time_display}")

            # 2. Панель лапласиана
            ax2 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo(central_longitude=0))
            ax2.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
            ax2.add_feature(cfeature.COASTLINE)
            ax2.add_feature(cfeature.BORDERS, linestyle=':')
            ax2.gridlines(draw_labels=True)

            # Определяем уровни для лапласиана
            laplacian_min = np.min(laplacian)
            laplacian_max = np.max(laplacian)
            laplacian_threshold = cyclone_params['laplacian_threshold']

            # Создаем уровни вокруг порогового значения
            levels_lap = np.linspace(min(-0.5, laplacian_min), max(0.5, laplacian_max), 20)

            # Отображаем лапласиан
            cs2 = ax2.contourf(lon_grid, lat_grid, laplacian, levels=levels_lap,
                             transform=ccrs.PlateCarree(), cmap='RdBu_r')

            # Добавляем контур порогового значения
            threshold_contour = ax2.contour(lon_grid, lat_grid, laplacian,
                                          levels=[laplacian_threshold],
                                          colors='k', linewidths=1.5,
                                          transform=ccrs.PlateCarree())

            # Отмечаем центры циклонов
            if cyclones_found:
                for cyclone_data in cyclone_centers:
                    lat, lon = cyclone_data[:2]
                    ax2.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())

            # Добавляем цветовую шкалу и заголовок
            cb2 = plt.colorbar(cs2, ax=ax2, orientation='horizontal', pad=0.05, label='Лапласиан давления (гПа/м²)')
            ax2.set_title(f"Лапласиан давления (гПа/м²)\nПорог: {laplacian_threshold}")

            # 3. Панель градиента
            ax3 = fig.add_subplot(gs[2], projection=ccrs.NorthPolarStereo(central_longitude=0))
            ax3.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
            ax3.add_feature(cfeature.COASTLINE)
            ax3.add_feature(cfeature.BORDERS, linestyle=':')
            ax3.gridlines(draw_labels=True)

            # Отображаем градиент давления
            gradient_magnitude = diagnostic_data["gradient_magnitude"]
            gradient_levels = np.linspace(0, min(5, np.max(gradient_magnitude)), 20)

            cs3 = ax3.contourf(lon_grid, lat_grid, gradient_magnitude, levels=gradient_levels,
                             transform=ccrs.PlateCarree(), cmap='viridis')

            # Добавляем контур порогового значения градиента
            gradient_threshold = cyclone_params['pressure_gradient_threshold']
            gradient_contour = ax3.contour(lon_grid, lat_grid, gradient_magnitude,
                                         levels=[gradient_threshold],
                                         colors='r', linewidths=1.5,
                                         transform=ccrs.PlateCarree())

            # Отмечаем центры циклонов
            if cyclones_found:
                for cyclone_data in cyclone_centers:
                    lat, lon = cyclone_data[:2]
                    ax3.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())

            # Добавляем цветовую шкалу и заголовок
            cb3 = plt.colorbar(cs3, ax=ax3, orientation='horizontal', pad=0.05, label='Градиент давления (гПа/градус)')
            ax3.set_title(f"Градиент давления (гПа/градус)\nПорог: {gradient_threshold}")

            # 4. Панель скорости ветра (если доступна и метод выбран)
            panel_idx = 3
            if has_wind_data and (detection_methods is None or 'wind_speed' in detection_methods):
                ax4 = fig.add_subplot(gs[panel_idx], projection=ccrs.NorthPolarStereo(central_longitude=0))
                ax4.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
                ax4.add_feature(cfeature.COASTLINE)
                ax4.add_feature(cfeature.BORDERS, linestyle=':')
                ax4.gridlines(draw_labels=True)
                
                # Получаем поле скорости ветра из диагностических данных
                wind_field = diagnostic_data.get("wind_field")
                
                if wind_field is not None:
                    # Определяем уровни для скорости ветра
                    wind_levels = np.linspace(0, min(30, np.max(wind_field)), 20)
                    
                    # Отображаем скорость ветра
                    cs4 = ax4.contourf(lon_grid, lat_grid, wind_field, levels=wind_levels,
                                      transform=ccrs.PlateCarree(), cmap='YlOrRd')
                    
                    # Добавляем контур порогового значения скорости ветра
                    wind_threshold = cyclone_params.get('min_wind_speed', 15.0)
                    wind_contour = ax4.contour(lon_grid, lat_grid, wind_field,
                                            levels=[wind_threshold],
                                            colors='b', linewidths=1.5,
                                            transform=ccrs.PlateCarree())
                    
                    # Отмечаем центры циклонов
                    if cyclones_found:
                        for cyclone_data in cyclone_centers:
                            lat, lon = cyclone_data[:2]
                            ax4.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
                            
                            # Если доступна информация о скорости ветра для этого циклона
                            if len(cyclone_data) > 6:
                                max_wind = cyclone_data[6]
                                ax4.text(lon, lat, f'{max_wind:.1f} м/с',
                                       transform=ccrs.PlateCarree(), fontsize=8,
                                       ha='right', va='bottom', color='blue',
                                       bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Добавляем цветовую шкалу и заголовок
                    cb4 = plt.colorbar(cs4, ax=ax4, orientation='horizontal', pad=0.05, label='Скорость ветра (м/с)')
                    ax4.set_title(f"Скорость ветра на 10м (м/с)\nПорог: {wind_threshold}")
                
                panel_idx += 1
            
            # 5. Панель завихренности (если доступна и метод выбран)
            if has_vorticity_data and (detection_methods is None or 'vorticity' in detection_methods):
                ax5 = fig.add_subplot(gs[panel_idx], projection=ccrs.NorthPolarStereo(central_longitude=0))
                ax5.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
                ax5.add_feature(cfeature.COASTLINE)
                ax5.add_feature(cfeature.BORDERS, linestyle=':')
                ax5.gridlines(draw_labels=True)
                
                # Получаем поле завихренности из диагностических данных
                vorticity_field = diagnostic_data.get("vorticity_field")
                
                if vorticity_field is not None:
                    # Определяем уровни для завихренности
                    vorticity_max = np.max(vorticity_field)
                    vorticity_min = np.min(vorticity_field)
                    # Создаем симметричную цветовую шкалу для завихренности
                    vort_abs_max = max(abs(vorticity_min), abs(vorticity_max))
                    vorticity_levels = np.linspace(-vort_abs_max, vort_abs_max, 20)
                    
                    # Отображаем завихренность
                    cs5 = ax5.contourf(lon_grid, lat_grid, vorticity_field, levels=vorticity_levels,
                                     transform=ccrs.PlateCarree(), cmap='RdBu_r')
                    
                    # Добавляем контур порогового значения завихренности
                    vorticity_threshold = cyclone_params.get('min_vorticity', 0.5e-5)
                    vorticity_contour = ax5.contour(lon_grid, lat_grid, vorticity_field,
                                                 levels=[vorticity_threshold],
                                                 colors='g', linewidths=1.5,
                                                 transform=ccrs.PlateCarree())
                    
                    # Отмечаем центры циклонов
                    if cyclones_found:
                        for cyclone_data in cyclone_centers:
                            lat, lon = cyclone_data[:2]
                            ax5.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree())
                            
                            # Если доступна информация о завихренности для этого циклона
                            if len(cyclone_data) > 7:
                                max_vorticity = cyclone_data[7]
                                ax5.text(lon, lat, f'{max_vorticity:.1e} с⁻¹',
                                       transform=ccrs.PlateCarree(), fontsize=8,
                                       ha='right', va='top', color='green',
                                       bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Добавляем цветовую шкалу и заголовок
                    cb5 = plt.colorbar(cs5, ax=ax5, orientation='horizontal', pad=0.05, 
                                     label='Относительная завихренность (с⁻¹)')
                    ax5.set_title(f"Относительная завихренность\nПорог: {vorticity_threshold:.1e} с⁻¹")

            # Общий заголовок с указанием используемых методов обнаружения
            title_str = f"Анализ циклонов в Арктике - {time_display}\nОбнаружено циклонов: {len(cyclone_centers)}"
            if detection_methods:
                methods_str = ", ".join(detection_methods)
                title_str += f"\nИспользуемые методы: {methods_str}"
            plt.suptitle(title_str, fontsize=16, y=0.98)

            grid_info = diagnostic_data["grid_info"]

            # Добавляем дополнительную информацию о сетке и параметрах
            plt.figtext(0.01, 0, f"Сетка: {grid_info['grid_resolution']} "
                      f"({grid_info['lat_step_km']:.1f}×{grid_info['lon_step_km']:.1f} км)",
                      fontsize=8)

            # Создаем путь для сохранения изображения
            output_file = os.path.join(output_dir, f'{file_prefix}arctic_cyclones_{time_str}.png')

            # Устанавливаем плотную компоновку и сохраняем
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Изображение сохранено: {output_file}")
            
            # 6. Создаем отдельную диаграмму с причинами отклонения циклонов
            if save_diagnostic and "rejected_centers" in diagnostic_data and diagnostic_data["rejected_centers"]:
                rejected_centers = diagnostic_data["rejected_centers"]
                
                # Подсчитываем причины отклонения
                reasons = {}
                for center in rejected_centers:
                    reason = center["reason"]
                    if reason in reasons:
                        reasons[reason] += 1
                    else:
                        reasons[reason] = 1
                
                # Создаем словарь понятных названий причин
                reason_labels = {
                    "small_size": "Малый размер области",
                    "high_pressure": "Высокое давление",
                    "shallow_depth": "Недостаточная глубина",
                    "no_closed_contour": "Отсутствие замкнутых изобар",
                    "weak_gradient": "Слабый градиент давления",
                    "low_latitude": "Широта ниже 70°N",
                    "invalid_size": "Размер вне допустимого диапазона",
                    "weak_wind": "Слабый ветер",
                    "weak_vorticity": "Слабая завихренность"
                }
                
                # Создаем круговую диаграмму причин отклонения
                fig2 = plt.figure(figsize=(10, 8))
                
                # Подготовка данных для диаграммы
                labels = []
                sizes = []
                
                for reason, count in reasons.items():
                    label = reason_labels.get(reason, reason)
                    labels.append(f"{label} ({count})")
                    sizes.append(count)
                
                # Создаем круговую диаграмму
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                      shadow=True, explode=[0.05] * len(sizes))
                plt.axis('equal')  # Чтобы круг был правильной формы
                
                plt.title(f"Причины отклонения циклонических систем\n{time_display}")
                
                # Сохраняем диаграмму
                rejection_file = os.path.join(output_dir, f'{file_prefix}rejection_reasons_{time_str}.png')
                plt.savefig(rejection_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Диаграмма причин отклонения сохранена: {rejection_file}")
                
        else:
            print(f"Изображение не сохранено: циклоны не обнаружены для {time_display}")

        return cyclone_centers, cyclones_found, diagnostic_data

    except Exception as e:
        print(f"Ошибка при визуализации для временного шага {time_idx}: {e}")
        import traceback
        traceback.print_exc()
        return [], False, None

def create_cyclone_statistics(cyclones, output_dir, file_prefix=''):
    """
    Создает статистические визуализации для обнаруженных циклонов.
    
    Параметры:
    ----------
    cyclones : list
        Список циклонов с их атрибутами
    output_dir : str
        Директория для сохранения результатов
    file_prefix : str
        Префикс для имен файлов
    """
    # Реализация без изменений из оригинального файла
    if not cyclones:
        print("Нет данных для создания статистики")
        return
        
    # Создаем DataFrame из списка циклонов
    df = pd.DataFrame(cyclones)
    
    # 1. Гистограмма давления в центре циклонов
    plt.figure(figsize=(10, 6))
    plt.hist(df['pressure'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Распределение давления в центре циклонов')
    plt.xlabel('Давление (гПа)')
    plt.ylabel('Количество циклонов')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}pressure_histogram.png'), dpi=300)
    plt.close()
    
    # 2. Гистограмма размеров циклонов
    plt.figure(figsize=(10, 6))
    plt.hist(df['radius'], bins=20, color='lightgreen', edgecolor='black')
    plt.title('Распределение размеров циклонов')
    plt.xlabel('Радиус (км)')
    plt.ylabel('Количество циклонов')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}radius_histogram.png'), dpi=300)
    plt.close()
    
    # 3. Диаграмма рассеяния: давление vs. глубина
    plt.figure(figsize=(10, 6))
    plt.scatter(df['pressure'], df['depth'], alpha=0.7)
    plt.title('Зависимость глубины циклона от давления в центре')
    plt.xlabel('Давление (гПа)')
    plt.ylabel('Глубина (гПа)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}pressure_depth_scatter.png'), dpi=300)
    plt.close()
    
    print(f"Статистические графики сохранены в директорию: {output_dir}")

def visualize_thermal_distribution(cyclones, start_date, end_date, output_dir, file_prefix=''):
    """
    Создает визуализацию пространственного распределения циклонов по типам термической структуры.
    
    Параметры:
    ----------
    cyclones : list
        Список циклонов с их атрибутами, включая термическую структуру
    start_date, end_date : str
        Начальная и конечная даты периода анализа
    output_dir : str
        Директория для сохранения результатов
    file_prefix : str
        Префикс для имен файлов
    """
    if not cyclones:
        print("Нет данных для создания визуализации")
        return
    
    # Разделяем циклоны по типам ядер
    cold_core = [c for c in cyclones if c['core_type'] == 'cold']
    warm_core = [c for c in cyclones if c['core_type'] == 'warm']
    mixed_core = [c for c in cyclones if c['core_type'] == 'mixed']
    
    # Создаем карту распределения циклонов
    plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines()
    
    # Наносим точки для разных типов циклонов
    if cold_core:
        cold_lats = [c['latitude'] for c in cold_core]
        cold_lons = [c['longitude'] for c in cold_core]
        plt.scatter(cold_lons, cold_lats, c='blue',
                   transform=ccrs.PlateCarree(), alpha=0.7, s=30, label='Холодноядерные')
    
    if warm_core:
        warm_lats = [c['latitude'] for c in warm_core]
        warm_lons = [c['longitude'] for c in warm_core]
        plt.scatter(warm_lons, warm_lats, c='red',
                   transform=ccrs.PlateCarree(), alpha=0.7, s=30, label='Теплоядерные')
    
    if mixed_core:
        mixed_lats = [c['latitude'] for c in mixed_core]
        mixed_lons = [c['longitude'] for c in mixed_core]
        plt.scatter(mixed_lons, mixed_lats, c='purple',
                   transform=ccrs.PlateCarree(), alpha=0.7, s=30, label='Смешанные')
    
    plt.legend(loc='lower left')
    plt.title(f'Пространственное распределение арктических циклонов по термической структуре\n{start_date} - {end_date}')
    
    map_file = os.path.join(output_dir, f"{file_prefix}cyclones_thermal_distribution_{start_date}_{end_date}.png")
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем круговую диаграмму распределения типов циклонов
    plt.figure(figsize=(10, 8))
    labels = ['Холодноядерные', 'Теплоядерные', 'Смешанные']
    sizes = [len(cold_core), len(warm_core), len(mixed_core)]
    colors = ['blue', 'red', 'purple']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Чтобы круг был правильной формы
    plt.title(f'Распределение циклонов по термической структуре\n{start_date} - {end_date}')
    
    pie_file = os.path.join(output_dir, f"{file_prefix}cyclones_thermal_pie_{start_date}_{end_date}.png")
    plt.savefig(pie_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализации сохранены в {output_dir}")