# Диагностические визуализации
"""
Функции для создания диагностических визуализаций.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from .config import DEFAULT_CRITERIA_DIR

def visualize_detection_criteria(detection_methods=None):
    """
    Создает наглядные изображения критериев обнаружения арктических циклонов
    на основе стандартов детекции с возможностью выбора методов.
    
    Параметры:
    ----------
    detection_methods : list, optional
        Список методов обнаружения циклонов для визуализации.
        Если None, визуализируются все доступные методы.
    """
    # Установка методов по умолчанию, если не указаны
    if detection_methods is None:
        detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed']
    
    # Создаем директорию для сохранения визуализаций
    output_dir = DEFAULT_CRITERIA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем визуализации только для выбранных методов
    methods_to_visualize = []
    if 'laplacian' in detection_methods:
        methods_to_visualize.append(('laplacian', 'Лапласиан давления'))
    if 'pressure_minima' in detection_methods:
        methods_to_visualize.append(('pressure_minima', 'Минимумы давления'))
    if 'closed_contour' in detection_methods:
        methods_to_visualize.append(('closed_contour', 'Замкнутые изобары'))
    if 'gradient' in detection_methods:
        methods_to_visualize.append(('gradient', 'Градиент давления'))
    if 'vorticity' in detection_methods:
        methods_to_visualize.append(('vorticity', 'Относительная завихренность'))
    if 'wind_speed' in detection_methods:
        methods_to_visualize.append(('wind_speed', 'Скорость ветра'))
    
    # 1. Визуализация выбранных критериев на одном графике
    n_methods = len(methods_to_visualize)
    rows = (n_methods + 1) // 2
    cols = min(2, n_methods)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Создаем схематическое представление для каждого выбранного метода
    for i, (method, title) in enumerate(methods_to_visualize):
        if i < len(axes):
            ax = axes[i]
            
            # Схематическое представление в зависимости от метода
            if method == 'laplacian':
                # Визуализация лапласиана
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                P = 1000 - 15 * np.exp(-R**2/0.3)  # Давление с минимумом в центре
                
                # Лапласиан давления
                dP2_dx2 = (np.gradient(np.gradient(P, axis=1), axis=1))
                dP2_dy2 = (np.gradient(np.gradient(P, axis=0), axis=0))
                laplacian = dP2_dx2 + dP2_dy2
                
                levels_p = np.linspace(985, 1000, 11)
                levels_l = np.linspace(-1, 1, 21)
                
                cs = ax.contour(X, Y, P, levels=levels_p, colors='k', linewidths=0.5)
                cf = ax.contourf(X, Y, laplacian, levels=levels_l, cmap='RdBu_r', alpha=0.7)
                ax.contour(X, Y, laplacian, levels=[-0.15], colors='r', linewidths=1.5)
                plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                plt.colorbar(cf, ax=ax, label='Лапласиан давления')
            
            elif method == 'pressure_minima':
                # Визуализация минимумов давления
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                P = 1000 - 15 * np.exp(-R**2/0.3)  # Давление с минимумом в центре
                
                levels_p = np.linspace(985, 1000, 11)
                cs = ax.contour(X, Y, P, levels=levels_p, colors='k', linewidths=0.5)
                cf = ax.contourf(X, Y, P, levels=levels_p, cmap='viridis', alpha=0.7)
                plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                
                # Отмечаем центр
                ax.plot(0, 0, 'ro', markersize=10)
                ax.text(0, 0, '985', ha='center', va='bottom', color='white', fontsize=10,
                      bbox=dict(facecolor='red', alpha=0.5))
                
                plt.colorbar(cf, ax=ax, label='Давление (гПа)')
            
            elif method == 'closed_contour':
                # Визуализация замкнутых изобар
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                P = 1000 - 15 * np.exp(-R**2/0.3)  # Давление с минимумом в центре
                
                levels_p = np.linspace(985, 1000, 11)
                cs = ax.contour(X, Y, P, levels=levels_p, colors='k', linewidths=0.5)
                plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                
                # Отмечаем центр и радиус проверки
                circle = plt.Circle((0, 0), 0.5, fill=False, color='red', linestyle='--')
                ax.add_patch(circle)
                ax.plot(0, 0, 'ro', markersize=8)
                ax.text(0, 0, '985', ha='right', va='bottom', color='red', fontsize=10)
                ax.text(0.5, 0.5, 'Радиус проверки', ha='right', va='bottom', color='red', fontsize=10)
            
            elif method == 'gradient':
                # Визуализация градиента давления
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                P = 1000 - 15 * np.exp(-R**2/0.3)  # Давление с минимумом в центре
                
                # Градиент давления
                dP_dx = np.gradient(P, axis=1)
                dP_dy = np.gradient(P, axis=0)
                gradient = np.sqrt(dP_dx**2 + dP_dy**2)
                
                levels_p = np.linspace(985, 1000, 11)
                cs = ax.contour(X, Y, P, levels=levels_p, colors='k', linewidths=0.5)
                cf = ax.contourf(X, Y, gradient, levels=20, cmap='YlOrRd')
                ax.contour(X, Y, gradient, levels=[0.5], colors='r', linewidths=1.5)
                plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                
                plt.colorbar(cf, ax=ax, label='Градиент давления (гПа/градус)')
            
            elif method == 'vorticity':
                # Визуализация завихренности
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                
                # Модель завихренности
                vorticity = -np.exp(-R**2/0.3) * 1e-5
                
                levels_v = np.linspace(-2e-5, 2e-5, 21)
                cf = ax.contourf(X, Y, vorticity, levels=levels_v, cmap='RdBu_r')
                ax.contour(X, Y, vorticity, levels=[-1e-5], colors='r', linewidths=1.5)
                
                plt.colorbar(cf, ax=ax, label='Относительная завихренность (с⁻¹)')
            
            elif method == 'wind_speed':
                # Визуализация скорости ветра
                x = np.linspace(-1, 1, 100)
                y = np.linspace(-1, 1, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2)
                
                # Модель поля ветра вокруг циклона
                wind_speed = 15 * (1 - np.exp(-R**2/0.2))
                
                levels_w = np.linspace(0, 20, 21)
                cf = ax.contourf(X, Y, wind_speed, levels=levels_w, cmap='YlGnBu')
                ax.contour(X, Y, wind_speed, levels=[15], colors='r', linewidths=1.5)
                
                # Отмечаем центр
                ax.plot(0, 0, 'ro', markersize=8)
                
                plt.colorbar(cf, ax=ax, label='Скорость ветра (м/с)')
            
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.axis('off')
    
    # Скрываем пустые графики
    for i in range(len(methods_to_visualize), len(axes)):
        axes[i].set_visible(False)
    
    # Общий заголовок
    plt.suptitle("Критерии обнаружения арктических циклонов", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'cyclone_detection_criteria.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 2. Визуализация комплексного подхода к детекции
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Okubo-Weiss метод
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Метод Okubo-Weiss')
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    W = -(1 - R**2) * np.exp(-R**2/0.5)
    
    levels = np.linspace(-1, 1, 21)
    cf1 = ax1.contourf(X, Y, W, levels=levels, cmap='RdBu_r')
    ax1.contour(X, Y, W, levels=[-0.2], colors='k', linewidths=1.5)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Метод Okubo-Weiss\nW < -0.2σw')
    
    # Метод вортичности
    ax2 = fig.add_subplot(gs[0, 1])
    vorticity = -np.exp(-R**2/0.3)
    levels_v = np.linspace(-1, 1, 21)
    cf2 = ax2.contourf(X, Y, vorticity, levels=levels_v, cmap='RdBu_r')
    ax2.contour(X, Y, vorticity, levels=[-0.3], colors='k', linewidths=1.5)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_aspect('equal')
    ax2.set_title('Метод завихренности\nω > пороговое значение')
    
    # Метод градиента SST
    ax3 = fig.add_subplot(gs[0, 2])
    SST = 15 - 5 * np.exp(-((X-0.3)**2 + (Y-0.3)**2)/0.2)
    dSST_dx = np.gradient(SST, axis=1)
    dSST_dy = np.gradient(SST, axis=0)
    sst_gradient = np.sqrt(dSST_dx**2 + dSST_dy**2)
    levels_sst = np.linspace(0, 2, 21)
    cf3 = ax3.contourf(X, Y, sst_gradient, levels=levels_sst, cmap='YlOrRd')
    ax3.contour(X, Y, SST, levels=np.linspace(10, 15, 6), colors='k', linewidths=0.5)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal')
    ax3.set_title('Градиент SST\nДополнительный критерий')
    
    # Комбинированный подход
    ax4 = fig.add_subplot(gs[1, :])
    
    # Создаем маски для разных методов
    mask_ow = W < -0.2
    mask_vort = vorticity < -0.3
    mask_sst = sst_gradient > 1.0
    
    # Комбинация методов
    combined_mask = np.zeros_like(W)
    combined_mask[mask_ow & mask_vort] = 1  # Уровень доверия: два метода
    combined_mask[mask_ow & mask_vort & mask_sst] = 2  # Уровень доверия: три метода
    
    levels_comb = [-0.5, 0.5, 1.5, 2.5]
    cmap = plt.cm.get_cmap('viridis', 3)
    cf4 = ax4.contourf(X, Y, combined_mask, levels=levels_comb, cmap=cmap)
    
    # Добавляем контуры давления
    P = 1000 - 15 * np.exp(-R**2/0.3)
    levels_p = np.linspace(985, 1000, 11)
    cs = ax4.contour(X, Y, P, levels=levels_p, colors='k', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    ax4.set_aspect('equal')
    ax4.set_title('Комбинированный подход с уровнями доверия')
    
    # Создаем легенду
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor='k', label='Нет циклона'),
        Patch(facecolor=cmap(1), edgecolor='k', label='Средний уровень доверия\n(два метода согласны)'),
        Patch(facecolor=cmap(2), edgecolor='k', label='Высокий уровень доверия\n(все методы согласны)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cyclone_detection_criteria_2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Визуализация пороговых значений для разных типов циклонов
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Параметры для разных типов циклонов
    cyclone_types = ['Синоптический\nмасштаб', 'Мезомасштабные\nциклоны', 'Полярные\nмезоциклоны']
    
    # Пороги для разных параметров
    laplacian_thresholds = [-0.15, -0.12, -0.10]
    pressure_thresholds = [1015.0, 1018.0, 1020.0]
    depth_thresholds = [2.5, 1.0, 0.8]
    gradient_thresholds = [0.7, 0.5, 0.4]
    radius_ranges = [(200, 1500), (100, 600), (50, 400)]
    vorticity_thresholds = [0.5e-5, 1.0e-5, 1.5e-5]
    
    # Позиции для групп столбцов
    x = np.arange(len(cyclone_types))
    width = 0.15
    
    # Создаем столбцы для каждого параметра
    rects1 = ax.bar(x - 2*width, [-l*100 for l in laplacian_thresholds], width, label='Лапласиан давления\n(×10⁻²)')
    rects2 = ax.bar(x - width, [p-1000 for p in pressure_thresholds], width, label='Порог давления\n(>1000 гПа)')
    rects3 = ax.bar(x, depth_thresholds, width, label='Мин. глубина (гПа)')
    rects4 = ax.bar(x + width, gradient_thresholds, width, label='Мин. градиент\n(гПа/градус)')
    rects5 = ax.bar(x + 2*width, [v*1e6 for v in vorticity_thresholds], width, label='Мин. завихренность\n(×10⁻⁶ с⁻¹)')
    
    # Настройка осей и меток
    ax.set_ylabel('Значение порога')
    ax.set_title('Пороговые значения для разных типов циклонов')
    ax.set_xticks(x)
    ax.set_xticklabels(cyclone_types)
    ax.legend()
    
    # Добавляем подписи со значениями
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    
    # Добавляем таблицу с диапазонами радиусов
    table_data = []
    for i, (min_r, max_r) in enumerate(radius_ranges):
        table_data.append([f'{min_r}-{max_r} км'])
    
    table = plt.table(cellText=table_data,
                     rowLabels=cyclone_types,
                     colLabels=['Диапазон радиусов'],
                     loc='bottom',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Корректируем положение нижней границы графика
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cyclone_detection_thresholds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализации критериев обнаружения циклонов сохранены в {output_dir}")