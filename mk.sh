#!/bin/bash

# Create main project directory
mkdir -p arctic_cyclones

# Create main files
touch arctic_cyclones/config.py
touch arctic_cyclones/main.py

# Create data module
mkdir -p arctic_cyclones/data
touch arctic_cyclones/data/__init__.py
touch arctic_cyclones/data/download.py
touch arctic_cyclones/data/preprocessing.py

# Create detection module
mkdir -p arctic_cyclones/detection
touch arctic_cyclones/detection/__init__.py
touch arctic_cyclones/detection/parameters.py
touch arctic_cyclones/detection/algorithms.py
touch arctic_cyclones/detection/thermal.py

# Create analysis module
mkdir -p arctic_cyclones/analysis
touch arctic_cyclones/analysis/__init__.py
touch arctic_cyclones/analysis/metrics.py
touch arctic_cyclones/analysis/tracking.py

# Create visualization module
mkdir -p arctic_cyclones/visualization
touch arctic_cyclones/visualization/__init__.py
touch arctic_cyclones/visualization/plots.py
touch arctic_cyclones/visualization/diagnostics.py

# Add file descriptions as comments
echo "# Конфигурационные параметры" > arctic_cyclones/config.py
echo "# Главный скрипт запуска" > arctic_cyclones/main.py
echo "# Функции загрузки данных ERA5" > arctic_cyclones/data/download.py
echo "# Предобработка данных" > arctic_cyclones/data/preprocessing.py
echo "# Параметры для разных типов циклонов" > arctic_cyclones/detection/parameters.py
echo "# Алгоритмы обнаружения" > arctic_cyclones/detection/algorithms.py
echo "# Классификация термической структуры" > arctic_cyclones/detection/thermal.py
echo "# Расчет метрик циклонов" > arctic_cyclones/analysis/metrics.py
echo "# Отслеживание циклонов во времени" > arctic_cyclones/analysis/tracking.py
echo "# Функции визуализации" > arctic_cyclones/visualization/plots.py
echo "# Диагностические визуализации" > arctic_cyclones/visualization/diagnostics.py

echo "Arctic cyclones project structure created successfully!"

# Print the created structure
echo "Created structure:"
find arctic_cyclones -type f | sort
