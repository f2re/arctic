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
    """Formats a timestep into a string suitable for filenames."""
    # Check data type
    if isinstance(time_step, np.datetime64):
        # Method 1: Convert via string
        dt_str = str(time_step)
        if 'T' in dt_str:
            date_part = dt_str.split('T')[0]  # Get '2010-01-01'
            time_part = dt_str.split('T')[1]  # Get '00:00:00.000000000'
            hour_part = time_part.split(':')[0]  # Get '00'
            return f"{date_part}_{hour_part}"
        
        # Method 2: Via datetime (if first method didn't work)
        try:
            dt_obj = time_step.astype('datetime64[s]').item()  # Convert to Python datetime
            return dt_obj.strftime('%Y-%m-%d_%H')
        except (AttributeError, ValueError):
            pass
    
    # Handle other timestamp types
    elif hasattr(time_step, 'strftime'):  # For datetime objects
        return time_step.strftime('%Y-%m-%d_%H')
    elif isinstance(time_step, str):  # For string representations
        if 'T' in time_step:  # ISO format
            date_part = time_step.split('T')[0]
            time_part = time_step.split('T')[1]
            hour_part = time_part[:2] if ':' in time_part else time_part
            return f"{date_part}_{hour_part}"
    
    # Fallback for other cases
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return timestamp


def create_polar_stereographic_map(min_latitude=65.0, figsize=(10, 8), central_longitude=0.0):
    """Creates a standard map in polar stereographic projection."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=central_longitude))
    ax.set_extent([-180, 180, min_latitude, 90], ccrs.PlateCarree())
    
    # Add standard geographic features
    add_standard_features(ax)
    # Add coordinate grid
    add_grid_lines(ax)
    # Add circular boundary
    add_circular_boundary(ax)
    
    return fig, ax


def add_standard_features(ax, resolution='50m'):
    """Adds standard geographic features to the map."""
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.8, edgecolor='black', zorder=10)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.5, edgecolor='gray', zorder=9)
    ax.add_feature(cfeature.LAKES.with_scale(resolution), facecolor='lightblue', edgecolor='black', zorder=8, alpha=0.5)


def add_grid_lines(ax):
    """Adds coordinate grid to the map."""
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': 'gray'}
    gl.ylabel_style = {'size': 8, 'color': 'gray'}


def add_circular_boundary(ax):
    """Adds a circular boundary for polar stereographic projection."""
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def prepare_plot_data(field_data, lats, lons, min_latitude=65.0):
    """Prepares data for plotting."""
    # Convert coordinates to a grid if they are 1D
    if lats.ndim == 1 or lons.ndim == 1:
        logger.info("Converting 1D coordinate arrays to 2D meshgrid for plotting")
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    else:
        lons_mesh, lats_mesh = lons, lats
    
    # Create mask for the Arctic region
    arctic_mask = lats_mesh >= min_latitude
    
    # Mask data outside the Arctic
    if not np.all(arctic_mask):
        logger.debug(f"Masking data below {min_latitude}°N")
        masked_data = np.ma.masked_where(~arctic_mask, field_data)
    else:
        masked_data = np.ma.masked_invalid(field_data)
    
    return masked_data, lons_mesh, lats_mesh, arctic_mask


def create_symmetric_norm(data, default_vmax=1.0):
    """Creates a symmetric normalization for data with positive and negative values."""
    valid_data = data.compressed() if hasattr(data, 'compressed') else data[~np.isnan(data)]
    
    if len(valid_data) > 0:
        abs_max = max(abs(np.min(valid_data)), abs(np.max(valid_data)))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -default_vmax, default_vmax
    
    # Create normalization with zero at center
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    return norm, vmin, vmax


def add_colorbar(fig, ax, contour, label, pad=0.05, shrink=0.8, fontsize=10, orientation='horizontal'):
    """Adds a standard colorbar."""
    cbar = fig.colorbar(contour, ax=ax, orientation=orientation, pad=pad, shrink=shrink)
    cbar.set_label(label, fontsize=fontsize, fontweight='bold')
    cbar.ax.tick_params(labelsize=fontsize-1)
    return cbar


def add_min_max_annotations(ax, data, lons, lats, transform=ccrs.PlateCarree(), 
                           min_color='blue', max_color='red', fontsize=8, zorder=20):
    """Adds annotations with minimum and maximum values."""
    valid_data = np.ma.masked_invalid(data)
    min_idx = np.unravel_index(np.ma.argmin(valid_data), valid_data.shape)
    max_idx = np.unravel_index(np.ma.argmax(valid_data), valid_data.shape)
    
    min_val = valid_data[min_idx]
    max_val = valid_data[max_idx]
    min_lat, min_lon = lats[min_idx], lons[min_idx]
    max_lat, max_lon = lats[max_idx], lons[max_idx]
    
    # Display markers for minimum and maximum
    ax.plot(min_lon, min_lat, marker='v', color=min_color, markersize=8, 
            markeredgecolor='white', transform=transform, zorder=zorder)
    ax.plot(max_lon, max_lat, marker='^', color=max_color, markersize=8, 
            markeredgecolor='white', transform=transform, zorder=zorder)
    
    # Add labels to markers
    ax.text(min_lon, min_lat, f"Min\n{min_val:.4g}", color=min_color, fontsize=fontsize, 
           fontweight='bold', ha='right', va='bottom', transform=transform, zorder=zorder+1,
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax.text(max_lon, max_lat, f"Max\n{max_val:.4g}", color=max_color, fontsize=fontsize, 
           fontweight='bold', ha='left', va='top', transform=transform, zorder=zorder+1, 
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Information in bottom corner
    plt.figtext(0.99, 0.01, 
               f"Min: {min_val:.4g} at ({min_lat:.2f}°N, {min_lon:.2f}°E)\n"
               f"Max: {max_val:.4g} at ({max_lat:.2f}°N, {max_lon:.2f}°E)", 
               fontsize=fontsize, ha='right', va='bottom', color='black',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


def plot_bidirectional_field(field_data, lats, lons, time_step, output_dir, 
                            field_name, field_unit, threshold=None, 
                            min_latitude=65.0, figsize=(10, 8),
                            ax=None, show_title=True):
    """Visualizes a field with positive and negative values."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    standalone = ax is None
    if standalone:
        fig, ax = create_polar_stereographic_map(min_latitude=min_latitude, figsize=figsize)
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    masked_data, lons_mesh, lats_mesh, arctic_mask = prepare_plot_data(
        field_data, lats, lons, min_latitude)
    
    try:
        # Create symmetric normalization for zero center
        norm, vmin, vmax = create_symmetric_norm(masked_data)
        
        # Levels for filled contours and isolines
        fill_levels = np.linspace(vmin, vmax, 30)
        
        # Add zero level explicitly
        if vmin < 0 < vmax and 0 not in fill_levels:
            fill_levels = np.sort(np.append(fill_levels, 0))
        
        # Filled contour with zero-centered coloring
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_data,
                             transform=ccrs.PlateCarree(),
                             cmap='RdBu_r',  # Red for positive, blue for negative
                             levels=fill_levels,
                             norm=norm,  # Normalize to fix zero at white
                             extend='both',
                             alpha=0.8,
                             zorder=5)
        
        # Special line for zero - outlines boundary between positive and negative values
        zero_contour = ax.contour(lons_mesh, lats_mesh, masked_data,
                                levels=[0],
                                colors='grey',
                                linewidths=0.5,
                                alpha=0.8,
                                transform=ccrs.PlateCarree(),
                                zorder=7)
        
        plt.clabel(zero_contour, [0], inline=True, fontsize=9, fmt='%.1f')
        
        # Threshold contour
        if threshold is not None:
            threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_data,
                                         levels=[threshold],
                                         colors='red',
                                         linewidths=1.5,
                                         transform=ccrs.PlateCarree(),
                                         zorder=8)
            
            plt.clabel(threshold_contour, [threshold], inline=True, fontsize=9, 
                      fmt='%.2g', colors='red')
        
        # Add colorbar
        cbar = add_colorbar(fig, ax, contourf, f'{field_name} ({field_unit})')
        
        # Minimum and maximum annotations
        if standalone:
            add_min_max_annotations(ax, masked_data, lons_mesh, lats_mesh)
        
        # Title with timestamp
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            threshold_info = f" (Threshold: {threshold} {field_unit})" if threshold else ""
            ax.set_title(f'{field_name}{threshold_info}\n{time_str}',
                       fontsize=12, fontweight='bold')
        
        # Explanation about positive values
        logger.info(f"Field name: {field_name}")
        if field_name.lower() in ['relative vorticity','vorticity', 'pressure laplacian']:
            plt.figtext(0.02, 0.02, 'Positive values (red) indicate cyclonic circulation', 
                      fontsize=8, ha='left')
        
        # Save the image for standalone plot
        if standalone:
            output_file = output_path / f"{field_name.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved {field_name} visualization to {output_file}")
            plt.close(fig)
            return output_file
        
        return output_path / f"{field_name.lower().replace(' ', '_')}_{timestamp}.png"
    
    except Exception as e:
        logger.error(f"Error creating {field_name} plot: {str(e)}")
        if standalone:
            output_file = output_path / f"{field_name.lower().replace(' ', '_')}_{timestamp}_error.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return output_file
        
        return output_path / f"{field_name.lower().replace(' ', '_')}_{timestamp}_error.png"


def plot_wind_field(u_wind, v_wind, lats, lons, threshold, time_step, output_dir,
                   ax=None, show_title=True):
    """
    Visualizes wind field with vectors and speed.
    
    Args:
        u_wind: 2D array containing U-component of wind
        v_wind: 2D array containing V-component of wind
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        threshold: Wind speed threshold used for detection
        time_step: Time step for the data
        output_dir: Directory to save the output image
        ax: Optional matplotlib axes for embedding in a larger figure
        show_title: Whether to show the title (useful when embedding)
        
    Returns:
        Path to the saved figure
    """
    # Calculate wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    standalone = ax is None
    if standalone:
        fig, ax = create_polar_stereographic_map(min_latitude=65.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Prepare data for plotting
    masked_speed, lons_mesh, lats_mesh, arctic_mask = prepare_plot_data(
        wind_speed, lats, lons, min_latitude=65.0)
    
    # Prepare u and v components
    masked_u, _, _, _ = prepare_plot_data(u_wind, lats, lons, min_latitude=65.0)
    masked_v, _, _, _ = prepare_plot_data(v_wind, lats, lons, min_latitude=65.0)
    
    try:
        # Get valid data for visualization range
        valid_speed = masked_speed.compressed() if hasattr(masked_speed, 'compressed') else masked_speed[~np.isnan(masked_speed)]
        
        if len(valid_speed) > 0:
            # Use 2nd and 98th percentiles to exclude outliers
            vmin, vmax = np.percentile(valid_speed, [2, 98])
            vmin = max(0, vmin)  # Wind speed can't be negative
            vmax = max(vmax, threshold * 1.5)  # Ensure threshold is visible
        else:
            vmin, vmax = 0, max(20, threshold * 1.5)
            logger.warning("No valid wind speed data points for contour plot")
        
        # Create contour levels with threshold included
        levels = np.linspace(vmin, vmax, 20)
        if threshold not in levels and vmin <= threshold <= vmax:
            levels = np.sort(np.append(levels, threshold))
        
        # Filled contour for wind speed
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_speed,
                              transform=ccrs.PlateCarree(),
                              cmap='YlOrRd',  # Yellow-Orange-Red colormap
                              levels=levels,
                              extend='max',
                              alpha=0.8,
                              zorder=5)
        
        # Highlight threshold with a contour line
        threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_speed,
                                     levels=[threshold],
                                     colors='red',
                                     linewidths=1.5,
                                     transform=ccrs.PlateCarree(),
                                     zorder=8)
        
        plt.clabel(threshold_contour, [threshold], inline=True, fontsize=9, 
                  fmt='%.2g', colors='red')
        
        # Add wind vectors with thinning for clarity
        stride = 8  # Adjust based on grid resolution
        
        # Check if coordinates are 1D or 2D
        if lats.ndim == 1 and lons.ndim == 1:
            lons_thin = lons[::stride]
            lats_thin = lats[::stride]
            u_thin = masked_u[::stride, ::stride]
            v_thin = masked_v[::stride, ::stride]
            lon_mesh_thin, lat_mesh_thin = np.meshgrid(lons_thin, lats_thin)
        else:
            lon_mesh_thin = lons_mesh[::stride, ::stride]
            lat_mesh_thin = lats_mesh[::stride, ::stride]
            u_thin = masked_u[::stride, ::stride]
            v_thin = masked_v[::stride, ::stride]
        
        # Add quiver for wind vectors
        quiver = ax.quiver(lon_mesh_thin, lat_mesh_thin, u_thin, v_thin,
                          transform=ccrs.PlateCarree(),
                          scale=500,
                          width=0.002,
                          headwidth=4,
                          color='black',
                          alpha=0.7,
                          zorder=10)
        
        # Add key for wind vectors scale
        ax.quiverkey(quiver, 0.9, 0.95, 10, '10 m/s', labelpos='E',
                    coordinates='axes', color='black', fontproperties={'size': 9})
        
        # Add colorbar
        cbar = add_colorbar(fig, ax, contourf, 'Wind Speed (m/s)')
        
        # Add minimum and maximum annotations
        if standalone:
            add_min_max_annotations(ax, masked_speed, lons_mesh, lats_mesh)
        
        # Title with timestamp
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            ax.set_title(f'Wind Field (Threshold: {threshold} m/s)\n{time_str}',
                       fontsize=12, fontweight='bold')
        
        # Explanation about threshold
        plt.figtext(0.02, 0.02, f'Red contour: Wind speed ≥ {threshold} m/s', 
                   fontsize=8, ha='left', color='darkred')
        
        # Save the image for standalone plot
        if standalone:
            output_file = output_path / f"wind_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved wind field visualization to {output_file}")
            plt.close(fig)
            return output_file
        
        return output_path / f"wind_{timestamp}.png"
    
    except Exception as e:
        logger.error(f"Error creating wind field plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        if standalone:
            output_file = output_path / f"wind_{timestamp}_error.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return output_file
        
        return output_path / f"wind_{timestamp}_error.png"


def plot_pressure_field(pressure, lats, lons, time_step, output_dir,
                       threshold=None, ax=None, show_title=True):
    """
    Visualizes a pressure field with isobars at 5 hPa intervals.
    
    Args:
        pressure: 2D array containing pressure values (in Pa)
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        time_step: Time step for the data
        output_dir: Directory to save the output image
        threshold: Optional threshold value to highlight
        ax: Optional matplotlib axes for embedding in a larger figure
        show_title: Whether to show the title (useful when embedding)
        
    Returns:
        Path to the saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    standalone = ax is None
    if standalone:
        fig, ax = create_polar_stereographic_map(min_latitude=65.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Convert Pa to hPa for better readability if necessary
    if np.mean(pressure) > 10000:  # Assuming data is in Pa
        pressure_hpa = pressure / 100.0
    else:  # Already in hPa
        pressure_hpa = pressure
    
    # Prepare data for plotting
    masked_pressure, lons_mesh, lats_mesh, arctic_mask = prepare_plot_data(
        pressure_hpa, lats, lons, min_latitude=65.0)
    
    try:
        # Get valid data for visualization range
        valid_pressure = masked_pressure.compressed() if hasattr(masked_pressure, 'compressed') else masked_pressure[~np.isnan(masked_pressure)]
        
        if len(valid_pressure) > 0:
            # Use typical sea level pressure range but adjust based on data
            data_min = max(960, np.percentile(valid_pressure, 1))
            data_max = min(1040, np.percentile(valid_pressure, 99))
            
            # Round to nearest 5 for cleaner intervals
            data_min = np.floor(data_min / 5) * 5
            data_max = np.ceil(data_max / 5) * 5
        else:
            data_min, data_max = 960, 1040
            logger.warning("No valid pressure data points for contour plot")
        
        # Create contour levels
        fill_levels = np.linspace(data_min, data_max, 20)
        line_levels = np.arange(data_min, data_max + 5, 5)  # Isobars at 5 hPa intervals
        
        # Filled contour for pressure
        contourf = ax.contourf(lons_mesh, lats_mesh, masked_pressure,
                              transform=ccrs.PlateCarree(),
                              cmap='viridis_r',  # Reversed so low pressure (cyclones) is highlighted
                              levels=fill_levels,
                              extend='both',
                              alpha=0.7,
                              zorder=5)
        
        # Add isobars
        contour = ax.contour(lons_mesh, lats_mesh, masked_pressure,
                            levels=line_levels,
                            colors='black',
                            linewidths=0.7,
                            alpha=0.7,
                            transform=ccrs.PlateCarree(),
                            zorder=7)
        
        # Add labels to isobars (every other line to avoid crowding)
        plt.clabel(contour, line_levels[::2], inline=True, fontsize=8, fmt='%d')
        
        # Add threshold contour if provided
        if threshold is not None:
            threshold_contour = ax.contour(lons_mesh, lats_mesh, masked_pressure,
                                         levels=[threshold],
                                         colors='red',
                                         linewidths=1.5,
                                         transform=ccrs.PlateCarree(),
                                         zorder=8)
            
            plt.clabel(threshold_contour, [threshold], inline=True, fontsize=9, 
                      fmt='%.2g', colors='red')
        
        # Add colorbar
        cbar = add_colorbar(fig, ax, contourf, 'Sea Level Pressure (hPa)')
        
        # Add minimum and maximum annotations
        if standalone:
            add_min_max_annotations(ax, masked_pressure, lons_mesh, lats_mesh, 
                                  min_color='red', max_color='blue')  # Reversed colors for pressure
        
        # Title with timestamp
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            ax.set_title(f'Mean Sea Level Pressure\n{time_str}',
                       fontsize=12, fontweight='bold')
        
        # Explanation about isobars
        plt.figtext(0.02, 0.02, 'Isobars at 5 hPa intervals', 
                   fontsize=8, ha='left')
        
        # Save the image for standalone plot
        if standalone:
            output_file = output_path / f"pressure_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pressure field visualization to {output_file}")
            plt.close(fig)
            return output_file
        
        return output_path / f"pressure_{timestamp}.png"
    
    except Exception as e:
        logger.error(f"Error creating pressure field plot: {str(e)}")
        
        if standalone:
            output_file = output_path / f"pressure_{timestamp}_error.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return output_file
        
        return output_path / f"pressure_{timestamp}_error.png"


def plot_closed_contour_field(pressure, contour_mask, lats, lons, time_step, output_dir,
                             ax=None, show_title=True):
    """
    Visualizes closed pressure contours.
    
    Args:
        pressure: 2D array containing pressure values (in Pa)
        contour_mask: 2D boolean array where True indicates closed contours
        lats: 2D array of latitudes
        lons: 2D array of longitudes
        time_step: Time step for the data
        output_dir: Directory to save the output image
        ax: Optional matplotlib axes for embedding in a larger figure
        show_title: Whether to show the title (useful when embedding)
        
    Returns:
        Path to the saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    standalone = ax is None
    if standalone:
        fig, ax = create_polar_stereographic_map(min_latitude=65.0, figsize=(10, 8))
    else:
        fig = ax.figure
    
    timestamp = format_timestep(time_step)
    
    # Convert Pa to hPa for better readability if necessary
    if np.mean(pressure) > 10000:  # Assuming data is in Pa
        pressure_hpa = pressure / 100.0
    else:  # Already in hPa
        pressure_hpa = pressure
    
    # Prepare data for plotting
    masked_pressure, lons_mesh, lats_mesh, arctic_mask = prepare_plot_data(
        pressure_hpa, lats, lons, min_latitude=65.0)
    
    # Also mask the contour mask
    if contour_mask.shape == pressure.shape:
        masked_contours, _, _, _ = prepare_plot_data(
            contour_mask, lats, lons, min_latitude=65.0)
    else:
        masked_contours = contour_mask
        logger.warning("Contour mask shape does not match pressure field shape, using as is")
    
    try:
        # Get valid data for visualization range
        valid_pressure = masked_pressure.compressed() if hasattr(masked_pressure, 'compressed') else masked_pressure[~np.isnan(masked_pressure)]
        
        if len(valid_pressure) > 0:
            # Use typical sea level pressure range but adjust based on data
            data_min = max(960, np.percentile(valid_pressure, 1))
            data_max = min(1040, np.percentile(valid_pressure, 99))
            
            # Round to nearest 5 for cleaner intervals
            data_min = np.floor(data_min / 5) * 5
            data_max = np.ceil(data_max / 5) * 5
        else:
            data_min, data_max = 960, 1040
            logger.warning("No valid pressure data points for contour plot")
        
        # Create contour levels
        line_levels = np.arange(data_min, data_max + 5, 5)  # Isobars at 5 hPa intervals
        
        # Add pressure contours
        contour = ax.contour(lons_mesh, lats_mesh, masked_pressure,
                            levels=line_levels,
                            colors='black',
                            linewidths=0.7,
                            alpha=0.7,
                            transform=ccrs.PlateCarree(),
                            zorder=7)
        
        # Add labels to isobars (every other line to avoid crowding)
        plt.clabel(contour, line_levels[::2], inline=True, fontsize=8, fmt='%d')
        
        # Highlight closed contours
        if np.any(masked_contours):
            closed_contour = ax.contourf(lons_mesh, lats_mesh, masked_contours.astype(float),
                                       levels=[0.5, 1.5],  # For boolean mask
                                       colors=['red'],
                                       alpha=0.3,
                                       transform=ccrs.PlateCarree(),
                                       zorder=6)
            
            # Add a legend or explanation
            plt.figtext(0.02, 0.02, 'Red areas: Closed pressure contours', 
                       fontsize=8, ha='left', color='darkred')
        
        # Title with timestamp
        if show_title:
            if hasattr(time_step, 'strftime'):
                time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
            else:
                time_str = str(time_step)
            
            ax.set_title(f'Closed Pressure Contours\n{time_str}',
                       fontsize=12, fontweight='bold')
        
        # Save the image for standalone plot
        if standalone:
            output_file = output_path / f"closed_contour_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved closed contour visualization to {output_file}")
            plt.close(fig)
            return output_file
        
        return output_path / f"closed_contour_{timestamp}.png"
    
    except Exception as e:
        logger.error(f"Error creating closed contour plot: {str(e)}")
        
        if standalone:
            output_file = output_path / f"closed_contour_{timestamp}_error.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return output_file
        
        return output_path / f"closed_contour_{timestamp}_error.png"


def plot_laplacian_field(laplacian, lats, lons, threshold, time_step, output_dir,
                        ax=None, show_title=True):
    """
    Visualizes a laplacian field with threshold highlighted.
    
    Args:
        laplacian: 2D array containing laplacian values
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
    return plot_bidirectional_field(
        field_data=laplacian,
        lats=lats,
        lons=lons,
        time_step=time_step,
        output_dir=output_dir,
        field_name='Pressure Laplacian',
        field_unit='Pa/km²',
        threshold=threshold,
        min_latitude=65.0,
        ax=ax,
        show_title=show_title
    )


def plot_vorticity_field(vorticity, lats, lons, threshold, time_step, output_dir,
                        ax=None, show_title=True):
    """
    Visualizes a vorticity field with threshold highlighted.
    
    Args:
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
    return plot_bidirectional_field(
        field_data=vorticity,
        lats=lats,
        lons=lons,
        time_step=time_step,
        output_dir=output_dir,
        field_name='Relative Vorticity',
        field_unit='1/s',
        threshold=threshold,
        min_latitude=65.0,
        ax=ax,
        show_title=show_title
    )


def register_plot_functions():
    """Registers all visualization functions for different criteria."""
    global CRITERION_PLOT_FUNCTIONS
    
    CRITERION_PLOT_FUNCTIONS = {
        'pressure_laplacian': plot_laplacian_field,
        'vorticity': plot_vorticity_field,
        'wind_threshold': plot_wind_field,
        'closed_contour': plot_closed_contour_field,
        'pressure': plot_pressure_field,
        'pressure_minimum': plot_pressure_field,
    }


def plot_combined_criteria(criteria_data, time_step, output_dir):
    """
    Creates a combined visualization of multiple criteria fields.
    
    Args:
        criteria_data: Dictionary mapping criterion names to their data dictionaries
        time_step: Time step for the data (used in filename and title)
        output_dir: Directory to save the output image
        
    Returns:
        Path to the saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = format_timestep(time_step)
    num_criteria = len(criteria_data)
    
    if num_criteria == 0:
        logger.warning("No criteria data provided for combined plot")
        return Path(output_dir)
    
    # Determine layout for subplots
    if num_criteria <= 2:
        nrows, ncols = 1, num_criteria
    elif num_criteria <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 3, 2
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows), 
                           subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0)})
    
    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Register plot functions if not already done
    register_plot_functions()
    current_plot_idx = 0
    
    # Create subplot for each criterion
    for criterion_name, data in criteria_data.items():
        if criterion_name in CRITERION_PLOT_FUNCTIONS:
            if current_plot_idx < nrows * ncols:
                ax = axes.flat[current_plot_idx]
                plot_func = CRITERION_PLOT_FUNCTIONS[criterion_name]
                
                try:
                    logger.info(f"Adding {criterion_name} to combined plot")
                    plot_data = data.copy()
                    plot_data['time_step'] = time_step
                    plot_data['output_dir'] = output_path
                    
                    # Call the plotting function for the specific subplot
                    plot_path = plot_func(**plot_data, ax=ax, show_title=False)
                    
                    # Add a title to the subplot
                    ax.set_title(criterion_name.replace('_', ' ').title(), fontsize=10)
                    current_plot_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error for {criterion_name}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Mark the subplot as failed
                    ax.text(0.5, 0.5, f'{criterion_name}\nFailed to plot',
                           ha='center', va='center', transform=ax.transAxes, color='red')
                    current_plot_idx += 1
            else:
                logger.warning(f"Not enough subplots for criterion {criterion_name}")
        else:
            logger.warning(f"Plot function for criterion '{criterion_name}' not found")
    
    # Hide unused subplots
    for i in range(current_plot_idx, nrows * ncols):
        fig.delaxes(axes.flat[i])
    
    # Add a main title
    if hasattr(time_step, 'strftime'):
        time_str = time_step.strftime('%Y-%m-%d %H:%M UTC')
    else:
        time_str = str(time_step)
    
    fig.suptitle(f'Combined Criteria Visualization - {time_str}', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the combined figure
    combined_plot_path = output_path / f"combined_criteria_{timestamp}.png"
    save_figure(fig, combined_plot_path)
    plt.close(fig)
    
    return combined_plot_path


# Register plot functions when module is loaded
register_plot_functions()
