"""
Main entry point with example workflow for the ArcticCyclone package.

This script demonstrates a complete workflow from data acquisition to
visualization of arctic mesocyclones.

Note: To use the ERA5 data acquisition, you need to have a CDS API key.
See: https://cds.climate.copernicus.eu/api-how-to
"""

import os
import argparse
import datetime
import logging
from pathlib import Path

from core.config import ConfigManager
from core.logging_setup import setup_logging
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager
from detection.tracker import CycloneDetector, CycloneTracker
from export.formats.csv_exporter import CycloneCSVExporter
from visualization.tracks import plot_cyclone_tracks
# from visualization.heatmaps import plot_cyclone_density
from visualization.parameters import plot_cyclone_parameters


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Arctic Mesocyclone Detection and Analysis')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for output files')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()


def run_workflow(start_date, end_date, config_path='config.yaml', 
                output_dir='output', log_level='INFO'):
    """
    Run the complete Arctic Mesocyclone detection workflow.
    
    Args:
        start_date: Start date for analysis (datetime object or string YYYY-MM-DD)
        end_date: End date for analysis (datetime object or string YYYY-MM-DD)
        config_path: Path to configuration file
        output_dir: Directory for output files
        log_level: Logging level
        
    Returns:
        Dict containing results summary
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Setup logging
    setup_logging(log_level=log_level, 
                 log_file=os.path.join(output_dir, 'arctic_cyclone.log'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Arctic Mesocyclone detection workflow")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = ConfigManager(Path(config_path))
    logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize credential manager
    project_dir = Path(__file__).parent.absolute()
    # Путь к файлу с учетными данными в директории проекта
    credentials_file = project_dir /  ".credentials.json"
    credentials = CredentialManager(credentials_file=credentials_file)
    
    # Check if ERA5 credentials are available
    if not credentials.get("ERA5"):
        logger.warning("ERA5 credentials not found. Using environment variables if available.")
    
    # Initialize data source manager

    data_manager = DataSourceManager(
        config_path=Path(config_path),
        credentials=credentials 
    )
    
    # Define region and parameters for data acquisition
    region = {
        'north': 90.0,  # Northern boundary (North Pole)
        'south': 70.0,  # Southern boundary (Arctic Circle)
        'east': 180.0,  # Eastern boundary
        'west': -180.0  # Western boundary (full longitude range)
    }
    
    # Create timeframe dictionary for ERA5 API
    # Формируем только запрашиваемый период вместо всего года
    start_month = start_date.month
    end_month = end_date.month
    start_day = start_date.day
    end_day = end_date.day

    timeframe = {
        'years': [str(year) for year in range(start_date.year, end_date.year + 1)],
        'months': [str(month) for month in range(start_month, end_month + 1)],
        'days': [str(day) for day in range(start_day, end_day + 1)],
        'hours': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    }

    
    # Required meteorological parameters and pressure levels
    parameters = {
        'variables': ['z', 'u', 'v', 't', 'vo'],  # Geopotential, wind components, temperature, vorticity
        'levels': [1000, 925, 850, 700, 500]      # Standard pressure levels in hPa
    }
    
    # Download data from ERA5
    logger.info(f"Downloading ERA5 data for period {start_date} to {end_date}")
    try:
        dataset = data_manager.get_data(
            source="ERA5",
            parameters=parameters,
            region=region,
            timeframe=timeframe,
            use_cache=True
        )
        
        # Проверка и преобразование координат времени
        if 'valid_time' in dataset.dims and 'time' not in dataset.dims:
            logger.info("Обнаружена координата 'valid_time' вместо 'time', выполняется переименование")
            dataset = dataset.rename({'valid_time': 'time'})

        # Фильтрация набора данных по запрошенному временному диапазону
        dataset = dataset.sel(time=slice(start_date, end_date))       

        logger.info("Data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        return {"status": "error", "message": str(e)}
    
    # Initialize cyclone detector
    detector = CycloneDetector(min_latitude=70.0)
    
    # Set detection criteria - using pressure minimum and vorticity for identification
    detector.set_criteria(["pressure_minimum", "vorticity"])
    
    # Detect cyclones for each time step
    logger.info("Detecting cyclones...")
    all_cyclones = {}
    
    for time_step in dataset.time.values:
        try:
            cyclones = detector.detect(dataset, time_step)
            all_cyclones[time_step] = cyclones
            logger.info(f"Detected {len(cyclones)} cyclones at {time_step}")
        except Exception as e:
            logger.error(f"Error detecting cyclones at time {time_step}: {str(e)}")
    
    # Initialize cyclone tracker
    tracker = CycloneTracker()
    
    # Track cyclones through time
    logger.info("Tracking cyclones...")
    cyclone_tracks = tracker.track(all_cyclones)
    
    # Filter tracks by minimum duration and number of observations
    filtered_tracks = tracker.filter_tracks(cyclone_tracks, min_duration=12.0, min_points=3)
    logger.info(f"Found {len(filtered_tracks)} cyclone tracks with duration >= 12 hours")
    
    # Export tracks to CSV
    exporter = CycloneCSVExporter()
    csv_file = output_dir / "cyclone_tracks.csv"
    exporter.export_cyclone_tracks(filtered_tracks, csv_file)
    logger.info(f"Exported cyclone tracks to {csv_file}")
    
    # Analyze cyclone characteristics
    logger.info("Analyzing cyclone characteristics...")
    track_summaries = []
    for i, track in enumerate(filtered_tracks[:5]):  # Analyze first 5 tracks
        lifecycle = track[0].calculate_lifecycle_metrics()
        track_summary = {
            "track_id": getattr(track[0], 'track_id', f"track_{i}"),
            "duration_hours": lifecycle['lifespan_hours'],
            "deepening_rate": lifecycle['deepening_rate'],
            "displacement_km": lifecycle['displacement'],
            "mean_speed_kmh": lifecycle['mean_speed'],
            "min_pressure": min([c.central_pressure for c in track])
        }
        track_summaries.append(track_summary)
        logger.info(f"Track {i+1}: Duration={lifecycle['lifespan_hours']:.1f}h, "
                   f"Deepening={lifecycle['deepening_rate']:.2f} hPa/h, "
                   f"Distance={lifecycle['displacement']:.1f}km")
    
    # Visualize results
    logger.info("Generating visualizations...")
    
    # Plot cyclone tracks
    tracks_file = output_dir / "cyclone_tracks.png"
    plot_cyclone_tracks(filtered_tracks, region, tracks_file)
    logger.info(f"Saved cyclone tracks plot to {tracks_file}")
    
    # Plot cyclone density heatmap
    # heatmap_file = output_dir / "cyclone_density.png"
    # plot_cyclone_density([c for track in filtered_tracks for c in track], region, heatmap_file)
    # logger.info(f"Saved cyclone density heatmap to {heatmap_file}")
    
    # Plot parameters for the most intense cyclone (lowest pressure)
    if filtered_tracks:
        most_intense_track = min(filtered_tracks, 
                                key=lambda track: min([c.central_pressure for c in track]))
        params_file = output_dir / "cyclone_parameters.png"
        plot_cyclone_parameters(most_intense_track, params_file)
        logger.info(f"Saved cyclone parameters plot to {params_file}")
    
    logger.info("Workflow completed successfully")
    
    return {
        "status": "success",
        "tracks_count": len(filtered_tracks),
        "cyclones_count": sum(len(track) for track in filtered_tracks),
        "track_summaries": track_summaries,
        "output_files": {
            "csv": str(csv_file),
            "tracks_plot": str(tracks_file),
            "density_plot": str(heatmap_file),
            "parameters_plot": str(params_file) if filtered_tracks else None
        }
    }


def main():
    """Main entry point when run as script."""
    args = parse_arguments()
    result = run_workflow(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config,
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    if result["status"] == "success":
        print(f"Successfully detected {result['tracks_count']} cyclone tracks")
        print(f"Results saved to {args.output_dir}")
    else:
        print(f"Workflow failed: {result['message']}")


if __name__ == "__main__":
    main()