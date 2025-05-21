"""
Module for visualizing cyclone detection criteria fields.

Provides functions to visualize and save the fields used by different detection criteria
such as pressure laplacian, vorticity, etc.
"""

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.path as mpath
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Union, Any, List
import matplotlib.gridspec as gridspec

from .mappers import create_arctic_map, save_figure

# Dictionary to map criterion names to their respective plot functions
CRITERION_PLOT_FUNCTIONS = {}

# Initialize logger
logger = logging.getLogger(__name__)

def format_timestep(time_step):
    # Проверяем тип данных
    if isinstance(time_step, np.datetime64):
        # Метод 1: Преобразование через строку (простой и не требует доп. библиотек)
        dt_str = str(time_step)
        if 'T' in dt_str:
            date_part = dt_str.split('T')[0]  # Получаем '2010-01-01'
            time_part = dt_str.split('T')[1]  # Получаем '00:00:00.000000000'
            hour_part = time_part.split(':')[0]  # Получаем '00'
            return f"{date_part}_{hour_part}"
        
        # Метод 2: Через datetime (если первый метод не сработал)
        try:
            dt_obj = time_step.astype('datetime64[s]').item()  # Преобразуем в Python datetime
            return dt_obj.strftime('%Y-%m-%d_%H')
        except (AttributeError, ValueError):
            pass
        
    # Обработка других типов временных меток
    elif hasattr(time_step, 'strftime'):  # Для объектов datetime
        return time_step.strftime('%Y-%m-%d_%H')
    
    elif isinstance(time_step, str):  # Для строковых представлений
        if 'T' in time_step:  # ISO формат
            date_part = time_step.split('T')[0]
            time_part = time_step.split('T')[1]
            hour_part = time_part[:2] if ':' in time_part else time_part
            return f"{date_part}_{hour_part}"
    
    # Fallback for other cases
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return timestamp

# Register plot functions in the dictionary
def register_plot_functions():
    global CRITERION_PLOT_FUNCTIONS
    CRITERION_PLOT_FUNCTIONS = {
        'pressure_laplacian': plot_laplacian_field,
        'vorticity': plot_vorticity_field,
        'wind_threshold': plot_wind_field,
        'closed_contour': plot_closed_contour_field,
        'pressure': plot_pressure_field
    }

def plot_combined_criteria(criteria_data: Dict[str, Dict], 
                          time_step: Any,
                          output_dir: Union[str, Path]) -> Path:
    """
    Creates a combined visualization of multiple criteria fields on a single plot.
    
    Arguments:
        criteria_data: Dictionary mapping criterion names to their data dictionaries.
                      Each data dictionary should contain all necessary parameters for the
                      respective plot function.
        time_step: Time step for the data (used in filename and title)
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved figure
    """
    # Debug log the criteria data keys and their contents
    logger.info(f"Creating combined visualization with criteria: {list(criteria_data.keys())}")
    for criterion, data in criteria_data.items():
        logger.debug(f"Criterion {criterion} data keys: {list(data.keys())}")
        for key, value in data.items():
            if key not in ['time_step', 'output_dir', 'threshold']:
                if hasattr(value, 'shape'):
                    logger.debug(f"  {key} shape: {value.shape}")
                elif isinstance(value, (list, np.ndarray)):
                    logger.debug(f"  {key} length: {len(value)}")
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Format the timestamp for the filename
    timestamp = format_timestep(time_step)
    
    # Count the number of criteria to plot
    num_criteria = len(criteria_data)
    
    if num_criteria == 0:
        logger.warning("No criteria data provided for combined plot")
        return Path(output_dir)
    
    # Create figure with subplots based on number of criteria
    if num_criteria <= 2:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(1, num_criteria, figure=fig)
    elif num_criteria <= 4:
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 2, figure=fig)
    else:
        # For 5 or more criteria, use a 3x2 grid
        fig = plt.figure(figsize=(18, 24))
        gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Add a main title for the entire figure
    if hasattr(time_step, 'strftime'):
        time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
    else:
        time_str = str(time_step)
    
    fig.suptitle(f'Combined Criteria Fields - {time_str}', fontsize=16, fontweight='bold')
    
    # Plot each criterion in its subplot
    for i, (criterion_name, data) in enumerate(criteria_data.items()):
        # Create subplot with Arctic projection
        ax = plt.subplot(gs[i], projection=ccrs.NorthPolarStereo())
        
        # Get the plot function for this criterion
        if criterion_name in CRITERION_PLOT_FUNCTIONS:
            plot_func = CRITERION_PLOT_FUNCTIONS[criterion_name]
            
            # Add the axes to the data dictionary
            data['ax'] = ax
            data['show_title'] = True  # Show titles in the subplots
            
            # Call the plot function with the data
            try:
                # Make a copy of the data to avoid modifying the original
                plot_data = data.copy()
                
                # Check dimensions and ensure they match for all criteria types
                if criterion_name == 'pressure_laplacian' and 'laplacian' in plot_data and 'lats' in plot_data:
                    lats = plot_data.get('lats', [])
                    laplacian = plot_data.get('laplacian', [])
                    
                    # Log dimensions for debugging
                    logger.debug(f"Plotting {criterion_name} - lats shape: {np.shape(lats)}, data shape: {np.shape(laplacian)}")
                    
                    # Ensure dimensions match
                    if hasattr(laplacian, 'shape') and len(lats) != laplacian.shape[0]:
                        logger.warning(f"Dimension mismatch in {criterion_name}: lats={len(lats)}, data rows={laplacian.shape[0]}")
                        # Use only the valid part of the data that matches latitude dimension
                        min_dim = min(len(lats), laplacian.shape[0])
                        plot_data['lats'] = lats[:min_dim]
                        plot_data['laplacian'] = laplacian[:min_dim, :]
                        logger.info(f"Adjusted dimensions to match: {min_dim} rows")
                
                elif criterion_name == 'vorticity' and 'vorticity' in plot_data and 'lats' in plot_data:
                    lats = plot_data.get('lats', [])
                    vorticity = plot_data.get('vorticity', [])
                    
                    if hasattr(vorticity, 'shape') and len(lats) != vorticity.shape[0]:
                        logger.warning(f"Dimension mismatch in {criterion_name}: lats={len(lats)}, data rows={vorticity.shape[0]}")
                        min_dim = min(len(lats), vorticity.shape[0])
                        plot_data['lats'] = lats[:min_dim]
                        plot_data['vorticity'] = vorticity[:min_dim, :]
                
                elif criterion_name == 'closed_contour' and 'pressure' in plot_data and 'lats' in plot_data:
                    lats = plot_data.get('lats', [])
                    pressure = plot_data.get('pressure', [])
                    contour_mask = plot_data.get('contour_mask', [])
                    
                    if hasattr(pressure, 'shape') and len(lats) != pressure.shape[0]:
                        logger.warning(f"Dimension mismatch in {criterion_name}: lats={len(lats)}, data rows={pressure.shape[0]}")
                        min_dim = min(len(lats), pressure.shape[0])
                        plot_data['lats'] = lats[:min_dim]
                        plot_data['pressure'] = pressure[:min_dim, :]
                        if hasattr(contour_mask, 'shape'):
                            plot_data['contour_mask'] = contour_mask[:min_dim, :]
                
                # Always ensure ax is included
                plot_data['ax'] = ax
                plot_data['show_title'] = True
                
                # Call the plot function with the prepared data
                logger.info(f"Calling plot function for {criterion_name} with data keys: {list(plot_data.keys())}")
                plot_func(**plot_data)
                logger.info(f"Successfully plotted {criterion_name}")
            except Exception as e:
                logger.error(f"Error plotting {criterion_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())  # Print full traceback for debugging
                ax.text(0.5, 0.5, f"Error plotting {criterion_name}\n{str(e)}", 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            logger.warning(f"No plot function found for criterion: {criterion_name}")
            ax.text(0.5, 0.5, f"Unknown criterion: {criterion_name}", 
                   transform=ax.transAxes, ha='center', va='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the suptitle
    
    # Save the combined figure
    output_file = output_path / f"combined_criteria_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved combined criteria visualization to {output_file}")
    return output_file

def plot_criterion_field(field_data: np.ndarray, 
                        lats: np.ndarray, 
                        lons: np.ndarray,
                        title: str,
                        criterion_name: str,
                        time_step: Any,
                        output_dir: Union[str, Path],
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cmap: str = 'viridis',
                        min_latitude: float = 60.0,
                        figsize: Tuple[float, float] = (10, 8)) -> Path:
    """
    Visualizes a criterion field (e.g., laplacian, vorticity) on an Arctic map.
    
    Arguments:
        field_data: 2D array containing the field values
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        title: Title for the plot
        criterion_name: Name of the criterion (used in filename)
        time_step: Time step for the data (used in filename)
        output_dir: Directory to save the output image
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        cmap: Colormap to use
        min_latitude: Minimum latitude for the map
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to the saved figure
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection
    fig, ax = create_arctic_map(min_latitude=min_latitude, figsize=figsize)
    
    timestamp = format_timestep(time_step)
    
    # Define the transform for data coordinates
    transform = ccrs.PlateCarree()
    
    # Handle NaN values in the field data
    masked_data = np.ma.masked_invalid(field_data)
    
    # Log data shapes and ranges for debugging
    logger.debug(f"Plotting {criterion_name} field with shapes - lats: {lats.shape}, lons: {lons.shape}, data: {masked_data.shape}")
    logger.debug(f"Data range: min={np.nanmin(masked_data.compressed())}, max={np.nanmax(masked_data.compressed())}")
    logger.debug(f"Lat range: {np.nanmin(lats)}, {np.nanmax(lats)}, Lon range: {np.nanmin(lons)}, {np.nanmax(lons)}")
    
    # Ensure coordinates are 2D for proper plotting
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    # Mask data outside the Arctic region (below min_latitude)
    arctic_mask = lats_mesh >= min_latitude
    if not np.all(arctic_mask):
        logger.debug(f"Masking data below {min_latitude}°N")
        masked_data = np.ma.masked_where(~arctic_mask, masked_data)
    
    # Create the contourf plot with appropriate levels
    try:
        # Determine appropriate contour levels
        if vmin is None or vmax is None:
            # Calculate reasonable min/max values, ignoring outliers
            valid_data = masked_data.compressed()
            if len(valid_data) > 0:
                data_min, data_max = np.percentile(valid_data.compressed() if hasattr(valid_data, 'compressed') else valid_data, [2, 98])
                # Ensure we have some range even with constant data
                if data_min == data_max:
                    data_min = data_min - 0.1 * abs(data_min) if data_min != 0 else -0.1
                    data_max = data_max + 0.1 * abs(data_max) if data_max != 0 else 0.1
            else:
                # Fallback if no valid data
                data_min, data_max = -1, 1
                logger.warning("No valid data points for contour plot")
            
            levels = np.linspace(data_min, data_max, 15)
            logger.debug(f"Using auto-calculated levels from {data_min} to {data_max}")
        else:
            levels = np.linspace(vmin, vmax, 15)
            logger.debug(f"Using specified levels from {vmin} to {vmax}")
        
        # Create the contour plot
        contour = ax.contourf(lons_mesh, lats_mesh, masked_data, 
                            transform=transform,
                            cmap=cmap, 
                            levels=levels,
                            extend='both',  # Extend colormap for out-of-range values
                            zorder=10)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                        pad=0.05, shrink=0.8)
        cbar.set_label(f'{criterion_name.replace("_", " ").title()} Values')
        
        # Add a circular boundary for polar stereographic projection
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Add title with timestamp
        if hasattr(time_step, 'strftime'):
            time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
        else:
            time_str = str(time_step)
        
        plt.title(f'{title}\n{time_str}')
        
        # Save the figure
        output_file = output_path / f"{criterion_name}_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {criterion_name} field visualization to {output_file}")
    except Exception as e:
        logger.error(f"Error creating contour plot: {str(e)}")
        # Save the figure even if contour fails, to see what's happening
        output_file = output_path / f"{criterion_name}_{timestamp}_error.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error visualization to {output_file}")
    
    plt.close(fig)
    return output_file

def plot_laplacian_field(laplacian: np.ndarray, 
                        lats: np.ndarray, 
                        lons: np.ndarray,
                        threshold: float,
                        time_step: Any,
                        output_dir: Union[str, Path],
                        ax: Optional[plt.Axes] = None,
                        show_title: bool = True) -> Path:
    """
    Visualizes a laplacian field with threshold highlighted and isolines.
    """
    # [Первая часть функции без изменений]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection
    fig, ax = create_arctic_map(min_latitude=65.0, figsize=(10, 8))
    
    timestamp = format_timestep(time_step)
    
    # Define the transform for data coordinates
    transform = ccrs.PlateCarree()
    
    # Ensure coordinates are 2D for proper plotting
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    # Mask data outside the Arctic region
    arctic_mask = lats_mesh >= 65.0
    if not np.all(arctic_mask):
        logger.debug(f"Masking laplacian data below 65°N")
        masked_laplacian = np.ma.array(laplacian, mask=~arctic_mask)
    else:
        masked_laplacian = np.ma.masked_invalid(laplacian)
    
    try:
        # Get valid data for level calculation
        flat_laplacian = masked_laplacian.flatten()
        flat_mask = np.ma.getmaskarray(flat_laplacian) | np.isnan(flat_laplacian)
        valid_data = flat_laplacian[~flat_mask]
        
        # Log the actual data range
        if len(valid_data) > 0:
            actual_min = np.min(valid_data)
            actual_max = np.max(valid_data)
            logger.info(f"Laplacian actual range: min={actual_min:.6f}, max={actual_max:.6f} Pa/km²")
            
            # Создаем симметричный диапазон вокруг нуля для правильной визуализации
            abs_max = max(abs(actual_min), abs(actual_max))
            data_min = -abs_max
            data_max = abs_max
            
            # Находим местоположения минимумов и максимумов
            min_idx = np.unravel_index(np.argmin(masked_laplacian), masked_laplacian.shape)
            max_idx = np.unravel_index(np.argmax(masked_laplacian), masked_laplacian.shape)
            min_lat, min_lon = lats_mesh[min_idx], lons_mesh[min_idx]
            max_lat, max_lon = lats_mesh[max_idx], lons_mesh[max_idx]
        else:
            # Fallback если нет валидных данных
            data_min, data_max = -0.02, 0.02
            logger.warning("No valid laplacian data points for contour plot")
        
        # Создаем уровни для заливки и изолиний
        fill_levels = np.linspace(data_min, data_max, 30)
        line_levels = np.linspace(data_min, data_max, 20)
        
        # Добавляем явно нулевой уровень, если его нет в наборе уровней
        if data_min < 0 < data_max and 0 not in fill_levels:
            fill_levels = np.sort(np.append(fill_levels, 0))
            line_levels = np.sort(np.append(line_levels, 0))
        
        # Используем TwoSlopeNorm для гарантии, что 0 будет соответствовать белому цвету
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)
        
        # Заполненный контур с центрированием на нуле
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_laplacian, 
                             transform=transform,
                             cmap='RdBu_r',  # Синий для отрицательных, красный для положительных
                             levels=fill_levels,
                             norm=norm,  # Нормализация для фиксации нуля на белом
                             extend='both',
                             alpha=0.8,
                             zorder=5)
        
        # Изолинии лапласиана
        # contour = ax.contour(lons_mesh, lats_mesh, masked_laplacian,
        #                    levels=line_levels,
        #                    colors='black',
        #                    linewidths=0.7,
        #                    alpha=0.6,
        #                    transform=transform,
        #                    zorder=6)
        
        # Специальная линия для нуля - обводит границу между положительными и отрицательными значениями
        zero_contour = ax.contour(lons_mesh, lats_mesh, masked_laplacian,
                               levels=[0],
                               colors='grey',
                               linewidths=0.5,
                               alpha=0.8,
                               transform=transform,
                               zorder=7)
        
        # Подписи к изолиниям (не ко всем, чтобы избежать загромождения)
        # contour_labels = plt.clabel(contour, contour.levels[::3], inline=True, fontsize=8, fmt='%.4f')
        plt.clabel(zero_contour, [0], inline=True, fontsize=9, fmt='%.1f')
        
        # Контур порогового значения
        threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_laplacian,
                                    levels=[threshold],
                                    colors='red',
                                    linewidths=0.5,
                                    transform=transform,
                                    zorder=8)
        
        # Подпись порогового значения
        plt.clabel(threshold_contour, [threshold], inline=True, fontsize=9, fmt='%.2f',
                 colors='red')
        
        # Добавляем географические объекты поверх графика
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray', zorder=9)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='lightblue', edgecolor='black', zorder=8, alpha=0.5)
        
        # Легенда цветов
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', 
                          pad=0.05, shrink=0.8)
        cbar.set_label('Pressure Laplacian (Pa/km²)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        if len(valid_data) > 0:
            ax.plot(min_lon, min_lat, marker='v', color='blue', markersize=8, markeredgecolor='white', transform=transform, zorder=20)
            ax.plot(max_lon, max_lat, marker='^', color='red', markersize=8, markeredgecolor='white', transform=transform, zorder=20)
            ax.text(min_lon, min_lat, f"Min\n{actual_min:.4f}", color='blue', fontsize=8, fontweight='bold', ha='right', va='bottom', transform=transform, zorder=21, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            ax.text(max_lon, max_lat, f"Max\n{actual_max:.4f}", color='red', fontsize=8, fontweight='bold', ha='left', va='top', transform=transform, zorder=21, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            plt.figtext(0.99, 0.01, f"Min: {actual_min:.4f} at ({min_lat:.2f}N, {min_lon:.2f}E)\nMax: {actual_max:.4f} at ({max_lat:.2f}N, {max_lon:.2f}E)", fontsize=8, ha='right', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Круговая граница для полярной стереографической проекции
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Добавляем координатную сетку
        gl = ax.gridlines(crs=transform, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Заголовок с датой
        if hasattr(time_step, 'strftime'):
            time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
        else:
            time_str = str(time_step)
        
        plt.title(f'Pressure Laplacian Field (Threshold: {threshold} Pa/km²)\n{time_str}', 
                fontsize=12, fontweight='bold')
        
        # Пояснение о положительных значениях
        plt.figtext(0.02, 0.02, 'Positive values (red) indicate cyclonic circulation', fontsize=8, ha='left')
        
        # Сохраняем изображение
        output_file = output_path / f"laplacian_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved laplacian field visualization to {output_file}")
    except Exception as e:
        logger.error(f"Error creating laplacian field plot: {str(e)}")
        output_file = output_path / f"laplacian_{timestamp}_error.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error visualization to {output_file}")
    
    plt.close(fig)
    return output_file



def plot_vorticity_field(vorticity: np.ndarray, 
                        lats: np.ndarray, 
                        lons: np.ndarray,
                        threshold: float,
                        time_step: Any,
                        output_dir: Union[str, Path],
                        ax: Optional[plt.Axes] = None,
                        show_title: bool = True) -> Path:
    """
    Visualizes a vorticity field with threshold highlighted and isolines.
    
    Arguments:
        vorticity: 2D array containing vorticity values
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        threshold: Threshold value used for detection
        time_step: Time step for the data
        output_dir: Directory to save the output image
        ax: Optional matplotlib axes for embedding in a larger figure
        show_title: Whether to show the title (useful when embedding)
        
    Returns:
        Path to the saved figure
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection if ax is not provided
    if ax is None:
        fig, ax = create_arctic_map(min_latitude=65.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Define the transform for data coordinates
    transform = ccrs.PlateCarree()
    
    # Ensure coordinates are 2D for proper plotting
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    # Mask data outside the Arctic region
    arctic_mask = lats_mesh >= 65.0
    if not np.all(arctic_mask):
        logger.debug(f"Masking vorticity data below 65°N")
        masked_vorticity = np.ma.array(vorticity, mask=~arctic_mask)
    else:
        masked_vorticity = np.ma.masked_invalid(vorticity)
    
    try:
        # Get valid data for level calculation
        flat_vorticity = masked_vorticity.flatten()
        flat_mask = np.ma.getmaskarray(flat_vorticity) | np.isnan(flat_vorticity)
        valid_data = flat_vorticity[~flat_mask]
        
        # Log the actual data range
        if len(valid_data) > 0:
            actual_min = np.min(valid_data)
            actual_max = np.max(valid_data)
            logger.info(f"Vorticity actual range: min={actual_min:.6f}, max={actual_max:.6f} 1/s")
            
            # Create a symmetric range around zero for proper visualization
            abs_max = max(abs(actual_min), abs(actual_max))
            data_min = -abs_max
            data_max = abs_max
            
            # Find locations of minimums and maximums
            min_idx = np.unravel_index(np.argmin(masked_vorticity), masked_vorticity.shape)
            max_idx = np.unravel_index(np.argmax(masked_vorticity), masked_vorticity.shape)
            min_lat, min_lon = lats_mesh[min_idx], lons_mesh[min_idx]
            max_lat, max_lon = lats_mesh[max_idx], lons_mesh[max_idx]
        else:
            # Fallback if no valid data
            data_min, data_max = -1e-4, 1e-4
            logger.warning("No valid vorticity data points for contour plot")
        
        # Create levels for filled contours and isolines
        fill_levels = np.linspace(data_min, data_max, 30)
        line_levels = np.linspace(data_min, data_max, 20)
        
        # Add zero level explicitly if not in the levels
        if data_min < 0 < data_max and 0 not in fill_levels:
            fill_levels = np.sort(np.append(fill_levels, 0))
            line_levels = np.sort(np.append(line_levels, 0))
        
        # Use TwoSlopeNorm to ensure 0 corresponds to white color
        norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)
        
        # Filled contour with zero-centered coloring
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_vorticity, 
                             transform=transform,
                             cmap='RdBu_r',  # Blue for negative, red for positive
                             levels=fill_levels,
                             norm=norm,  # Normalize to fix zero at white
                             extend='both',
                             alpha=0.8,
                             zorder=5)
        
        # Vorticity isolines
        contour = ax.contour(lons_mesh, lats_mesh, masked_vorticity,
                           levels=line_levels[::2],  # Use fewer levels for clarity
                           colors='black',
                           linewidths=0.5,
                           alpha=0.6,
                           transform=transform,
                           zorder=6)
        
        # Special line for zero - outlines boundary between positive and negative values
        zero_contour = ax.contour(lons_mesh, lats_mesh, masked_vorticity,
                                levels=[0],
                                colors='grey',
                                linewidths=0.8,
                                alpha=0.8,
                                transform=transform,
                                zorder=7)
        
        # Contour labels (not for all to avoid clutter)
        plt.clabel(contour, contour.levels[::4], inline=True, fontsize=8, fmt='%.1e')
        plt.clabel(zero_contour, [0], inline=True, fontsize=9, fmt='%.1f')
        
        # Threshold contour - highlight the detection threshold
        threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_vorticity,
                                     levels=[threshold],
                                     colors='red',
                                     linewidths=1.5,
                                     transform=transform,
                                     zorder=8)
        
        # Threshold label
        plt.clabel(threshold_contour, [threshold], inline=True, fontsize=9, fmt='%.2e',
                  colors='red')
        
        # Add geographic features on top of the plot
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray', zorder=9)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='lightblue', edgecolor='black', zorder=8, alpha=0.5)
        
        # Color legend
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', 
                          pad=0.05, shrink=0.8)
        cbar.set_label('Relative Vorticity (1/s)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        if len(valid_data) > 0:
            ax.plot(min_lon, min_lat, marker='v', color='blue', markersize=8, markeredgecolor='white', transform=transform, zorder=20)
            ax.plot(max_lon, max_lat, marker='^', color='red', markersize=8, markeredgecolor='white', transform=transform, zorder=20)
            ax.text(min_lon, min_lat, f"Min\n{actual_min:.2e}", color='blue', fontsize=8, fontweight='bold', ha='right', va='bottom', transform=transform, zorder=21, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            ax.text(max_lon, max_lat, f"Max\n{actual_max:.2e}", color='red', fontsize=8, fontweight='bold', ha='left', va='top', transform=transform, zorder=21, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            plt.figtext(0.99, 0.01, f"Min: {actual_min:.2e} at ({min_lat:.2f}N, {min_lon:.2f}E)\nMax: {actual_max:.2e} at ({max_lat:.2f}N, {max_lon:.2f}E)", fontsize=8, ha='right', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Circular boundary for polar stereographic projection
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Add coordinate grid
        gl = ax.gridlines(crs=transform, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Title with date
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            plt.title(f'Vorticity Field (Threshold: {threshold:.2e} 1/s)\n{time_str}', 
                    fontsize=12, fontweight='bold')
        
        # Explanation about positive values
        plt.figtext(0.02, 0.02, 'Positive values (red) indicate cyclonic circulation in NH', fontsize=8, ha='left')
        
        # Save the image if this is a standalone plot (not embedded)
        if ax is None:
            output_file = output_path / f"vorticity_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved vorticity field visualization to {output_file}")
            return output_file
        else:
            # If embedded, just return the path without saving
            return output_path / f"vorticity_{timestamp}.png"
            
    except Exception as e:
        logger.error(f"Error creating vorticity field plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        output_file = output_path / f"vorticity_{timestamp}_error.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error visualization to {output_file}")
        plt.close(fig)
        return output_file

def plot_pressure_field(pressure: np.ndarray, 
                      lats: np.ndarray, 
                      lons: np.ndarray,
                      time_step: Any,
                      output_dir: Union[str, Path],
                      ax: Optional[plt.Axes] = None,
                      show_title: bool = True) -> Path:
    """
    Visualizes a pressure field with isobars at 5 hPa intervals for meteorological analysis.
    
    Arguments:
        pressure: 2D array containing pressure values (in Pa)
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        time_step: Time step for the data
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved figure
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection if ax is not provided
    if ax is None:
        fig, ax = create_arctic_map(min_latitude=60.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Convert Pa to hPa for better readability
    pressure_hpa = pressure 
    
    # Define the transform for data coordinates
    transform = ccrs.PlateCarree()
    
    # Ensure coordinates are 2D for proper plotting
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    # Mask data outside the Arctic region
    arctic_mask = lats_mesh >= 60.0
    if not np.all(arctic_mask):
        logger.debug(f"Masking pressure data below 60°N")
        # Create a masked array properly to avoid 2D boolean indexing error
        masked_pressure = np.ma.array(pressure_hpa, mask=~arctic_mask)
    else:
        masked_pressure = np.ma.masked_invalid(pressure_hpa)
    
    try:
        # Get valid data for level calculation (avoiding NaN and masked values)
        # Convert to 1D array first to avoid 2D boolean indexing error
        flat_pressure = masked_pressure.flatten()
        flat_mask = np.ma.getmaskarray(flat_pressure) | np.isnan(flat_pressure)
        valid_data = flat_pressure[~flat_mask]
        
        if len(valid_data) > 0:
            # Standard sea level pressure range is typically 980-1050 hPa
            # Use actual data range but constrain to meteorologically meaningful values
            data_min = max(960, np.floor(np.percentile(valid_data.compressed() if hasattr(valid_data, 'compressed') else valid_data, 2) / 5) * 5)
            data_max = min(1050, np.ceil(np.percentile(valid_data.compressed() if hasattr(valid_data, 'compressed') else valid_data, 98) / 5) * 5)
            
            # If range is too small, use standard range
            if data_max - data_min < 20:
                logger.warning("Pressure range too small, using standard range")
                data_min = 960
                data_max = 1050
        else:
            # Standard sea level pressure range if no valid data
            data_min, data_max = 960, 1050
            logger.warning("No valid pressure data points for contour plot")
        
        logger.debug(f"Using pressure levels from {data_min} to {data_max} hPa")
        
        # Create filled contour levels for coloring
        fill_levels = np.linspace(data_min, data_max, 15)
        
        # Create line contour levels at 5 hPa intervals
        line_levels = np.arange(data_min, data_max + 5, 5)  # +5 to include the max value
        
        # Check if the pressure data appears uniform (which would make contours invisible)
        pressure_std = np.std(valid_data)
        pressure_range = np.max(valid_data) - np.min(valid_data)
        logger.debug(f"Pressure data standard deviation: {pressure_std} hPa, range: {pressure_range} hPa")
        logger.debug(f"Pressure data min: {np.min(valid_data)} hPa, max: {np.max(valid_data)} hPa")
        
        # Log the actual data values for debugging
        logger.info(f"Original pressure field range: {np.min(valid_data):.2f} to {np.max(valid_data):.2f} hPa")
        
        # Don't modify the original data - let's see what the actual ERA5 field looks like
        # Instead, adjust the contour levels to better represent the actual data range
        if pressure_range < 20:  # If the range is small but not zero
            # Use a more appropriate range for the data
            mean_pressure = np.mean(valid_data)
            half_range = max(pressure_range / 2, 10)  # At least 10 hPa range
            data_min = max(980, np.floor((mean_pressure - half_range) / 5) * 5)
            data_max = min(1050, np.ceil((mean_pressure + half_range) / 5) * 5)
            logger.info(f"Adjusting contour levels to better show small pressure variations: {data_min}-{data_max} hPa")
            
            # Recalculate the levels with the new range
            fill_levels = np.linspace(data_min, data_max, 15)
            line_levels = np.arange(data_min, data_max + 5, 5)
        
        # Create the filled contour plot with lower zorder so coastlines appear on top
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_pressure, 
                             transform=transform,
                             cmap='viridis_r',  # Reversed colormap so low pressure (cyclones) is highlighted
                             levels=fill_levels,
                             extend='both',
                             alpha=0.8,  # Slightly transparent to better see coastlines
                             zorder=5)  # Lower zorder so coastlines appear on top
        
        # Add contour lines at 5 hPa intervals with higher contrast
        contour = ax.contour(lons_mesh, lats_mesh, masked_pressure,
                           levels=line_levels,
                           colors='black',
                           linewidths=1.0,  # Thicker lines for better visibility
                           transform=transform,
                           alpha=0.6,
                           zorder=6)  # Higher than filled contours but lower than coastlines
        
        # Add contour labels (every other line to avoid crowding)
        contour_labels = plt.clabel(contour, contour.levels[::2], inline=True, fontsize=6, fmt='%d')
        
        # Add white background to labels for better visibility
        # for label in contour_labels:
            # Make text appear bold by using path effects instead of fontweight
            # import matplotlib.patheffects as path_effects
            # label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            # label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        
        # Re-add coastlines and borders with higher zorder to ensure they're on top
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray', zorder=9)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='lightblue', edgecolor='black', zorder=8, alpha=0.5)
        
        # Add colorbar with proper hPa formatting
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', 
                          pad=0.05, shrink=0.8)
        cbar.set_label('Sea Level Pressure (hPa)', fontsize=10, fontweight='bold')
        
        # Format colorbar ticks to ensure they show proper hPa values
        cbar.ax.tick_params(labelsize=9)
        
        # Add a circular boundary for polar stereographic projection
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Add grid lines for geographic reference
        gl = ax.gridlines(crs=transform, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title with timestamp
        if hasattr(time_step, 'strftime'):
            time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
        else:
            time_str = str(time_step)
        
        plt.title(f'Mean Sea Level Pressure (hPa)\n{time_str}', fontsize=12, fontweight='bold')
        
        # Add a text annotation explaining the map
        plt.figtext(0.02, 0.02, 'Isobars at 5 hPa intervals', fontsize=8, ha='left')
        
        # Save the figure
        output_file = output_path / f"pressure_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved pressure field visualization to {output_file}")
    except Exception as e:
        logger.error(f"Error creating pressure field plot: {str(e)}")
        # Save the figure even if contour fails, to see what's happening
        output_file = output_path / f"pressure_{timestamp}_error.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error visualization to {output_file}")
    
    plt.close(fig)
    return output_file

def plot_wind_field(u_wind: np.ndarray, 
                   v_wind: np.ndarray,
                   lats: np.ndarray, 
                   lons: np.ndarray,
                   threshold: float,
                   time_step: Any,
                   output_dir: Union[str, Path],
                   ax: Optional[plt.Axes] = None,
                   show_title: bool = True) -> Path:
    """
    Visualizes wind field with vectors and speed.
    
    Arguments:
        u_wind: 2D array containing U-component of wind
        v_wind: 2D array containing V-component of wind
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        threshold: Wind speed threshold used for detection
        time_step: Time step for the data
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved figure
    """
    # Calculate wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection if ax is not provided
    if ax is None:
        fig, ax = create_arctic_map(min_latitude=60.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Define the transform for data coordinates
    transform = ccrs.PlateCarree()
    
    # Log data shapes and ranges for debugging
    logger.debug(f"Plotting wind field with shapes - lats: {lats.shape}, lons: {lons.shape}, u: {u_wind.shape}, v: {v_wind.shape}")
    logger.debug(f"Wind speed range: min={np.nanmin(wind_speed.compressed()) if hasattr(wind_speed, 'compressed') else np.nanmin(wind_speed)}, max={np.nanmax(wind_speed.compressed()) if hasattr(wind_speed, 'compressed') else np.nanmax(wind_speed)}")
    
    # Ensure coordinates are 2D for proper plotting
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    try:
        # Правильная работа с маской для полярных регионов
        # Создаем маску напрямую в numpy.ma без 2D булевого индексирования
        arctic_mask = lats_mesh < 60.0  # Маска для данных вне арктического региона
        
        # Создаем маскированные массивы с явным указанием маски
        masked_speed = np.ma.array(wind_speed, mask=arctic_mask)
        masked_u = np.ma.array(u_wind, mask=arctic_mask)
        masked_v = np.ma.array(v_wind, mask=arctic_mask)
        
        # Извлекаем действительные значения для статистики
        valid_speed = masked_speed.compressed()  # Немаскированные значения как 1D массив
        
        if len(valid_speed) > 0:
            data_min, data_max = np.percentile(valid_speed.compressed() if hasattr(valid_speed, 'compressed') else valid_speed, [2, 98])
            # Обеспечиваем корректный диапазон даже с почти постоянными данными
            if np.isclose(data_min, data_max):
                data_min = max(0, data_min - 0.1 * abs(data_min))
                data_max = data_max + 0.1 * abs(data_max) if data_max != 0 else 0.1
            
            logger.debug(f"Valid wind speed range: {data_min:.2f} to {data_max:.2f} m/s")
        else:
            # Запасной вариант, если нет действительных данных
            data_min, data_max = 0, threshold * 2 if threshold > 0 else 10
            logger.warning("No valid wind speed data points for contour plot")
        
        # Создаем уровни с отметкой порогового значения
        levels = np.linspace(data_min, data_max, 5)
        if threshold > data_min and threshold < data_max and threshold not in levels:
            levels = np.sort(np.append(levels, threshold))
        
        # Создаем заполненный контур для скорости ветра
        contour = ax.contourf(lons_mesh, lats_mesh, masked_speed, 
                             transform=transform,
                             cmap='YlOrRd', 
                             levels=levels,
                             extend='both',
                             zorder=5)
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label('Wind Speed (m/s)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Выделение областей, где скорость ветра превышает пороговое значение
        threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_speed,
                                    levels=[threshold],
                                    colors='red',
                                    linewidths=1.5,
                                    transform=transform,
                                    zorder=8)
        
        # Добавляем векторы ветра (с прореживанием для наглядности)
        stride = 10  # Регулируем в зависимости от разрешения сетки
        
        # Прореживаем массивы для стрелок
        if lats.ndim == 1 and lons.ndim == 1:
            # Для 1D координат создаем разреженную сетку
            lons_s = lons[::stride]
            lats_s = lats[::stride]
            
            # Прореживаем 2D массивы ветра
            u_s = u_wind[::stride, ::stride]
            v_s = v_wind[::stride, ::stride]
            
            # Создаем сетку для прореженных точек
            lon_mesh_s, lat_mesh_s = np.meshgrid(lons_s, lats_s)
        else:
            # Если координаты уже 2D, просто прореживаем все массивы
            lon_mesh_s = lons_mesh[::stride, ::stride]
            lat_mesh_s = lats_mesh[::stride, ::stride]
            u_s = u_wind[::stride, ::stride]
            v_s = v_wind[::stride, ::stride]
        
        # Добавляем маску для стрелок ветра, чтобы показать только точки в Арктике
        quiver_mask = lat_mesh_s >= 60.0
        
        # Маскируем точки вне Арктики с помощью np.where
        q_lons = np.where(quiver_mask, lon_mesh_s, np.nan)
        q_lats = np.where(quiver_mask, lat_mesh_s, np.nan)
        q_u = np.where(quiver_mask, u_s, np.nan)
        q_v = np.where(quiver_mask, v_s, np.nan)
        
        # Рисуем стрелки ветра
        ax.quiver(q_lons, q_lats, q_u, q_v,
                transform=transform, 
                scale=500, 
                width=0.002, 
                headwidth=4, 
                color='black', 
                alpha=0.5,
                zorder=6)
        
        # Добавляем географические объекты
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, edgecolor='gray', zorder=9)
        ax.add_feature(cfeature.LAKES.with_scale('50m'), facecolor='lightblue', edgecolor='black', zorder=8, alpha=0.5)
        
        # Добавляем круговую границу для полярной стереографической проекции
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Добавляем координатную сетку
        gl = ax.gridlines(crs=transform, draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Добавляем заголовок с временной меткой если show_title True
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            ax.set_title(f'Wind Field (Threshold: {threshold} m/s)\n{time_str}', 
                     fontsize=12, fontweight='bold')
        
        # Добавляем пояснение к карте только если это отдельный график
        if ax is None:
            plt.figtext(0.02, 0.02, f'Red contour: Wind speed ≥ {threshold} m/s', 
                        fontsize=8, ha='left', color='darkred')
        else:
            # Add a text annotation to the axes instead
            ax.text(0.02, 0.02, f'Wind ≥ {threshold} m/s', 
                   fontsize=8, ha='left', color='darkred', transform=ax.transAxes)
        
        # Сохраняем изображение только если ax is None (standalone plot)
        if ax is None:
            output_file = output_path / f"wind_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved wind field visualization to {output_file}")
            plt.close(fig)
        else:
            output_file = output_path / f"wind_{timestamp}.png"
    except Exception as e:
        logger.error(f"Error creating wind field plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())  # Добавляем полный стек ошибки для диагностики
        if ax is None:
            output_file = output_path / f"wind_{timestamp}_error.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error visualization to {output_file}")
            plt.close(fig)
        else:
            output_file = output_path / f"wind_{timestamp}_error.png"
    
    return output_file


def plot_closed_contour_field(pressure: np.ndarray, 
                            contour_mask: np.ndarray,
                            lats: np.ndarray, 
                            lons: np.ndarray,
                            time_step: Any,
                            output_dir: Union[str, Path],
                            ax: Optional[plt.Axes] = None,
                            show_title: bool = True) -> Path:
    """
    Visualizes closed pressure contours.
    
    Arguments:
        pressure: 2D array containing pressure values (in Pa)
        contour_mask: 2D boolean array where True indicates closed contours
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        time_step: Time step for the data
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved figure
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with Arctic projection if ax is not provided
    if ax is None:
        fig, ax = create_arctic_map(min_latitude=60.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Format the timestamp for the filename
    if isinstance(time_step, str):
        timestamp = time_step.replace(':', '').replace('-', '').replace(' ', '_')
    elif hasattr(time_step, 'strftime'):
        timestamp = time_step.strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    transform = ccrs.PlateCarree()
    
    # Convert Pa to hPa for better readability
    pressure_hpa = pressure / 100.0
    
    # Plot pressure contours
    contour_levels = np.linspace(np.nanmin(pressure_hpa.compressed()) if hasattr(pressure_hpa, 'compressed') else np.nanmin(pressure_hpa), np.nanmax(pressure_hpa.compressed()) if hasattr(pressure_hpa, 'compressed') else np.nanmax(pressure_hpa), 15)
    cs = ax.contour(lons, lats, pressure_hpa, 
                   levels=contour_levels,
                   transform=transform,
                   colors='black',
                   linewidths=0.5,
                   alpha=0.7)
    
    # Add contour labels
    plt.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%1.0f')
    
    # Highlight closed contours
    if np.any(contour_mask):
        ax.contourf(lons, lats, contour_mask.astype(float), 
                   levels=[0.5, 1.5],
                   transform=transform,
                   colors=['red'],
                   alpha=0.3)
    
    # Add title with timestamp if show_title is True
    if show_title:
        if hasattr(time_step, 'strftime'):
            time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
        else:
            time_str = str(time_step)
        
        ax.set_title(f'Closed Pressure Contours\n{time_str}')
    
    # Save the figure if ax is None (standalone plot)
    if ax is None:
        output_file = output_path / f"closed_contour_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved closed contour visualization to {output_file}")
        return output_file
    else:
        # If ax is provided, we're in a combined plot, so just return the path
        return Path(output_dir) / f"closed_contour_{timestamp}.png"

# Call the registration function after all plot functions are defined
register_plot_functions()
