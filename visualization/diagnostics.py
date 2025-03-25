# Диагностические визуализации
"""
Функции для создания диагностических визуализаций.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def visualize_detection_criteria():
    """
    Создает наглядные изображения критериев обнаружения арктических циклонов
    на основе стандартов детекции.
    """
    # Создаем директорию для сохранения визуализаций
    output_dir = os.path.join('/content/drive/MyDrive/arctic', 'criteria_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Визуализация критериев Okubo-Weiss
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Схематический рисунок для вихря с положительной завихренностью
    ax = axes[0, 0]
    ax.set_aspect('equal')
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='blue')
    ax.add_patch(circle)
    for theta in np.linspace(0, 2*np.pi, 12):
        r = 0.4
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        dx = 0.1 * np.sin(theta)
        dy = -0.1 * np.cos(theta)
        ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.03, fc='blue', ec='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Циклонический вихрь\n(положительная завихренность)')
    ax.axis('off')
    
    # Схематический график для параметра Okubo-Weiss
    ax = axes[0, 1]
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    W = -(1 - R**2) * np.exp(-R**2/0.5)
    
    levels = np.linspace(-1, 1, 21)
    cf = ax.contourf(X, Y, W, levels=levels, cmap='RdBu_r')
    ax.contour(X, Y, W, levels=[0], colors='k', linewidths=1.5)
    ax.set_aspect('equal')
    ax.set_title('Параметр Okubo-Weiss (W)\nW < 0 в областях с доминирующей завихренностью')
    plt.colorbar(cf, ax=ax)
    
    # Схематическое представление критерия на основе лапласиана давления
    ax = axes[1, 0]
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
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    ax.set_aspect('equal')
    ax.set_title('Лапласиан давления\nОтрицательные значения указывают на циклон')
    plt.colorbar(cf, ax=ax)
    
    # Схематическое представление замкнутых изобар
    ax = axes[1, 1]
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
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    
    # Отмечаем центр и радиус проверки
    circle = plt.Circle((0, 0), 0.5, fill=False, color='red', linestyle='--')
    ax.add_patch(circle)
    ax.plot(0, 0, 'ro', markersize=8)
    ax.text(0, 0, '985', ha='right', va='bottom', color='red', fontsize=10)
    
    ax.set_aspect('equal')
    ax.set_title('Замкнутые изобары\nНеобходимый критерий для циклона')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cyclone_detection_criteria_1.png'), dpi=300, bbox_inches='tight')
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