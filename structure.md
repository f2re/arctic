/Users/meteo/Documents/WWW/arctic/         # Root project directory
├── analysis/                   # Cyclone analysis capabilities
│   ├── climatology.py          # Long-term pattern and climatology analysis of cyclones
│   ├── comparisons.py          # Comparison methods between different cyclones or datasets
│   ├── __init__.py             # Package initializer with imports for analysis modules
│   └── statistics.py           # Statistical analysis and metrics for cyclone data
├── core/                       # Core system components
│   ├── config.py               # Configuration management for system settings
│   ├── exceptions.py           # Custom exception hierarchy for error handling
│   ├── __init__.py             # Core package initializer and imports
│   └── logging_setup.py        # Specialized logging system for scientific workflows
├── data/                       # Data management subsystem
│   ├── __init__.py             # Data package initializer
│   ├── acquisition.py          # Main data acquisition and source management
│   ├── base.py                 # Base classes for data handling
│   ├── catalog.py              # Dataset inventory and metadata management
│   ├── credentials.py          # Secure credential management for data sources
│   ├── adapters/               # Data source adapters for different reanalysis products
│   │   ├── __init__.py         # Adapters package initializer
│   │   └── era5.py             # ERA5 reanalysis data adapter implementation
│   ├── cache/                  # Directory for cached data storage
│   └── processors/             # Data processing and transformation modules
│       ├── __init__.py         # Processors package initializer
│       └── era5_processor.py   # Specialized ERA5 data processor
├── detection/                  # Cyclone detection and tracking subsystem
│   ├── __init__.py             # Detection package initializer
│   ├── tracker.py              # Cyclone tracking and lifecycle monitoring
│   ├── validators.py           # Validation of detected cyclone candidates
│   ├── algorithms/             # Detection algorithm implementations
│   │   ├── __init__.py         # Algorithms package initializer
│   │   ├── algorithm_factory.py # Factory for creating detection algorithm instances
│   │   ├── arctic_mesocyclone.py # Specialized arctic mesocyclone detection algorithm
│   │   ├── base_algorithm.py   # Base class for detection algorithms
│   │   ├── multi_parameter.py  # Multi-parameter combined detection algorithm
│   │   ├── pressure_minima.py  # Pressure minima based detection algorithm
│   │   └── serreze.py          # Implementation of Serreze algorithm for cyclone detection
│   └── criteria/               # Detection criteria implementations
│       ├── __init__.py         # Criteria package initializer
│       ├── closed_contour.py   # Closed contour detection criterion
│       ├── gradient.py         # Pressure gradient criterion for detection
│       ├── laplacian.py        # Laplacian of pressure criterion
│       ├── pressure.py         # Pressure-based detection criterion
│       ├── vorticity.py        # Vorticity-based detection criterion
│       └── wind.py             # Wind-based detection criterion
├── export/                     # Data export and publishing subsystem
│   ├── __init__.py             # Export package initializer
│   ├── publishers.py           # Data publishing to files, web services, and other destinations
│   └── formats/                # Export format implementations
│       ├── __init__.py         # Export formats package initializer
│       ├── csv_exporter.py     # CSV format exporter for cyclone data
│       ├── geojson_exporter.py # GeoJSON format exporter for geospatial cyclone data
│       └── netcdf_exporter.py  # NetCDF format exporter for scientific data
├── models/                     # Data models and representations
│   ├── __init__.py             # Models package initializer
│   ├── classifications.py      # Cyclone classification schemes and enumerations
│   ├── cyclone.py              # Core Cyclone class representation
│   └── parameters.py           # Cyclone meteorological parameters
├── visualization/              # Visualization capabilities
│   ├── __init__.py             # Visualization package initializer
│   ├── criteria.py             # Visualization of detection criteria
│   ├── heatmaps.py             # Spatial distribution visualization tools
│   ├── mappers.py              # Base mapping and spatial visualization tools
│   ├── parameters.py           # Visualization of cyclone parameters
│   └── tracks.py               # Cyclone track visualization
├── config.yaml                 # Configuration file for the application
├── README.md                   # Project documentation and overview
├── requirements.txt            # Python dependencies list
└── structure.md                # This file - describes project structure