#!/bin/bash

# Script to initialize Arctic Mesocyclone Framework project structure

# Create core module
mkdir -p core
touch core/__init__.py
touch core/config.py
touch core/exceptions.py
touch core/logging_setup.py

# Create data management subsystem
mkdir -p data/processors data/adapters
touch data/__init__.py
touch data/acquisition.py
touch data/credentials.py
touch data/catalog.py
touch data/processors/__init__.py
touch data/adapters/__init__.py

# Create detection algorithms
mkdir -p detection/criteria detection/algorithms
touch detection/__init__.py
touch detection/tracker.py
touch detection/validators.py
touch detection/criteria/__init__.py
touch detection/algorithms/__init__.py

# Create meteorological data representation
mkdir -p models
touch models/__init__.py
touch models/cyclone.py
touch models/parameters.py
touch models/classifications.py

# Create analysis capabilities
mkdir -p analysis
touch analysis/__init__.py
touch analysis/statistics.py
touch analysis/comparisons.py
touch analysis/climatology.py

# Create visualization subsystem
mkdir -p visualization
touch visualization/__init__.py
touch visualization/mappers.py
touch visualization/tracks.py
touch visualization/parameters.py
touch visualization/heatmaps.py

# Create export facilities
mkdir -p export/formats
touch export/__init__.py
touch export/formats/__init__.py
touch export/publishers.py
touch export/formats/csv_exporter.py

# Create output directories
mkdir -p output/figures output/data

# Create main workflow script
touch main.py

# Create setup files
touch setup.py
touch README.md
touch requirements.txt

echo "Arctic Mesocyclone Framework project structure initialized successfully!"