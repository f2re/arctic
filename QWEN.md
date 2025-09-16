# **ArcticCyclone Developer and Contribution Guide**

## 1. Project Philosophy and Scientific Rigor

**ArcticCyclone** is a scientific software framework designed for the reproducible and verifiable study of Arctic mesocyclones. The core philosophy is to blend meteorological science with robust software engineering practices. Every component, from data acquisition to analysis, must be implemented with transparency, modularity, and extensibility in mind.

All algorithms and methods should be grounded in established atmospheric physics. The implementation must facilitate sensitivity analyses, validation against independent datasets, and comparison with other published methods.

## 2. Refined Project Structure

The proposed structure is logical but can be improved by formalizing directories for tests, notebooks for experimentation, and documentation. This revised structure clearly separates source code, tests, documentation, and user-facing examples.

```
arctic-cyclone/
├── .github/                    # GitHub specific files (e.g., issue templates, workflows for CI/CD)
│   └── WORKFLOWS/
│       └── python-ci.yml       # Continuous Integration workflow
├── data/                       # Default directory for downloaded and processed data (added to .gitignore)
├── docs/                       # Project documentation
│   ├── index.md                # Main documentation page
│   ├── installation.md
│   ├── user_guide.md
│   ├── api_reference.md        # Auto-generated API documentation
│   └── scientific_methods.md   # Detailed description of scientific algorithms
├── notebooks/                  # Jupyter notebooks for exploration, tutorials, and result presentation
│   ├── 01_data_acquisition.ipynb
│   ├── 02_cyclone_detection_tutorial.ipynb
│   └── 03_visualization_examples.ipynb
├── src/
│   └── arctic/                 # Main source code package
│       ├── __init__.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── climatology.py
│       │   └── statistics.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py       # Handles typed configuration using Pydantic
│       │   ├── exceptions.py
│       │   └── logging_setup.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── acquisition.py
│       │   ├── catalog.py
│       │   └── processors.py   # Merged processing logic here
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── tracker.py
│       │   ├── algorithms.py   # Combined detection algorithms
│       │   └── criteria.py     # Refined criteria definitions
│       ├── io/                 # Renamed from 'export' for clarity (Input/Output)
│       │   ├── __init__.py
│       │   ├── reader.py       # For reading cyclone data
│       │   └── writer.py       # For writing cyclone data (CSV, NetCDF, etc.)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cyclone.py
│       │   └── track.py        # Dedicated model for a cyclone track
│       └── viz/                # Shortened from 'visualization'
│           ├── __init__.py
│           ├── maps.py         # For Cartopy-based map plots (tracks, heatmaps)
│           └── plots.py        # For matplotlib plots (parameter evolution)
├── tests/                      # Testing suite
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   └── conftest.py             # Pytest fixtures
├── .gitignore
├── config.yaml                 # Example configuration file
├── LICENSE
├── pyproject.toml              # Modern Python project metadata, dependencies, and tool config
└── README.md
```

## 3. Analysis of Detection Methods

Each detection method must be understood by its physical basis and its implementation trade-offs.

1.  **Pressure Minimum**:
    *   **Physical Basis**: Cyclones are fundamentally low-pressure systems. This is the most direct and common detection method.
    *   **Implementation**: Use a 2D filter (e.g., `scipy.ndimage.minimum_filter`) to find pixels that are local minima within a defined neighborhood (e.g., 5x5 grid cells, corresponding to ~250-500 km).
    *   **Strengths**: Simple, computationally efficient, and effective for well-defined, mature cyclones.
    *   **Weaknesses**: Prone to identifying minor, insignificant troughs or creating multiple detections within a single large system. Requires post-processing and validation.

2.  **Relative Vorticity Maximum**:
    *   **Physical Basis**: Cyclonic motion is characterized by positive relative vorticity in the Northern Hemisphere. This identifies the dynamic center of rotation, which may not perfectly align with the pressure minimum, especially in developing or occluding systems. The 850 hPa level is typically used as it is above most surface friction effects but still within the lower troposphere.
    *   **Implementation**: Calculate relative vorticity from `u` and `v` wind components (`vort = dv/dx - du/dy`). Use a maximum filter (`scipy.ndimage.maximum_filter`) to locate rotational centers.
    *   **Strengths**: Excellent at identifying dynamically active systems and can capture cyclones earlier in their lifecycle than pressure minima alone. Less sensitive to broad, weak pressure fields.
    *   **Weaknesses**: Can be "noisy," identifying small-scale shear zones or eddies that are not true mesocyclones. Requires careful thresholding.

3.  **Pressure Gradient**:
    *   **Physical Basis**: A strong pressure gradient implies strong winds (the geostrophic wind approximation). Intense cyclones are associated with tightly packed isobars.
    *   **Implementation**: Calculate the magnitude of the pressure gradient vector (`sqrt((dP/dx)^2 + (dP/dy)^2)`). Identify regions where this value exceeds a specified threshold.
    *   **Strengths**: Good for selecting only intense, dynamically significant systems.
    *   **Weaknesses**: Not a primary detection method on its own, as it identifies regions of strong winds, not necessarily the cyclone center. Best used as a secondary validation criterion.

4.  **Laplacian of Pressure**:
    *   **Physical Basis**: The Laplacian (∇²) of the pressure field highlights areas of maximum curvature. A large positive value in ∇²P is a robust indicator of a pressure minimum (the bottom of a "bowl").
    *   **Implementation**: Compute the second-order partial derivatives (`d²P/dx² + d²P/dy²`). Search for local maxima in this field.
    *   **Strengths**: Mathematically more robust for identifying the center of a low-pressure feature than a simple minimum filter. It is less sensitive to small-scale noise in the pressure field.
    *   **Weaknesses**: More computationally intensive than a simple minimum search.

## 4. Python Standards and Best Practices

To ensure code quality, maintainability, and collaboration, the project must adhere to strict standards.

### 4.1. Project and Dependency Management
-   **Use `pyproject.toml`**: This file should be the single source of truth for project metadata, dependencies, and tool configurations. It replaces `requirements.txt`, `setup.py`, and other legacy files. Use a modern build backend like `poetry` or `flit`.
-   **Virtual Environments**: All development must occur within a dedicated virtual environment (e.g., using `venv` or `conda`) to isolate dependencies.

### 4.2. Code Formatting and Linting
-   **Formatter**: Use **Black** for automatic, non-negotiable code formatting. This eliminates all arguments about style.
-   **Linter**: Use **Ruff** or **Flake8** with plugins (`flake8-bugbear`, `flake8-annotations`) to catch logical errors, style issues, and missing type hints.
-   **Import Sorting**: Use **isort** to automatically sort and group imports.
-   **Pre-commit Hooks**: Use the `pre-commit` framework to run Black, isort, and the linter automatically before each commit, ensuring no poorly formatted code enters the repository.

### 4.3. Typing
-   **Static Typing**: Use Python's `typing` module extensively. All function signatures (arguments and return values) and class variables must have type hints.
-   **Type Checker**: Use **Mypy** in its strict mode as part of the CI pipeline to statically check for type errors. This catches a huge class of bugs before runtime.
-   **Data Structures**: Use `Pydantic` for data models (like `Cyclone` and configuration files). This provides runtime data validation and serialization with clear, typed definitions.

### 4.4. Configuration
-   The `core/config.py` module should use Pydantic to load and validate the `config.yaml`. This ensures that the configuration is type-safe and all required fields are present at startup.

## 5. Testing Strategy

A multi-layered testing strategy is essential for scientific software.

### 5.1. Framework
-   Use **`pytest`** as the testing framework for its powerful features, fixtures, and plugin ecosystem.

### 5.2. Test Layers
1.  **Unit Tests (`tests/unit`)**:
    *   **Purpose**: Test individual functions and classes in isolation.
    *   **Characteristics**: Fast, no external dependencies (like network or large files). Use "mocking" to replace dependencies.
    *   **Example**: A test for a vorticity calculation function that provides a small, known `numpy` array of `u`/`v` winds and asserts that the output vorticity matches a pre-calculated value.

2.  **Integration Tests (`tests/integration`)**:
    *   **Purpose**: Test how different parts of the system work together.
    *   **Characteristics**: Slower, may require small, sample data files.
    *   **Example**: A test that runs the full workflow: `acquisition` (from a local sample GRIB/NetCDF file) -> `detection` -> `tracking`. The test would assert that a known cyclone in the sample data is detected and tracked correctly.

3.  **Scientific Validation**:
    *   **Purpose**: Ensure the scientific validity of the results. This is the most crucial layer for a scientific tool.
    *   **Methodology**:
        *   **Case Studies**: Run the framework on well-documented historical Arctic cyclone events (e.g., famous polar lows). Compare the detected track, intensity, and structure against published literature.
        *   **Intercomparison**: Compare the project's output climatology (e.g., cyclone track density) against established datasets like those from other research groups or models (e.g., Murray and Simmonds' tracking scheme).
        *   **Sensitivity Analysis**: Create notebooks (`notebooks/`) that systematically vary detection parameters (e.g., vorticity threshold, pressure minimum neighborhood size) and analyze the impact on the number and characteristics of detected cyclones.

### 5.3. Continuous Integration (CI)
-   Use **GitHub Actions** (`.github/workflows/python-ci.yml`) to automatically run the full test suite (linting, type checking, unit tests, integration tests) on every push and pull request. This ensures that the main branch is always stable and working.

## 6. Documentation Principles

Documentation is not an afterthought; it is a critical component of the software.

### 6.1. Audience and Types
-   **For Users**: `README.md`, installation guide, tutorials (`notebooks/`). Focus on "how to use it."
-   **For Contributors**: This developer guide, API reference, scientific methods documentation. Focus on "how to build and extend it."

### 6.2. Docstrings (In-Code Documentation)
-   **Style**: Use a standard format like **Google Style** or **NumPy Style** docstrings. This allows for automatic generation of a beautiful API reference using tools like **Sphinx** with the `autodoc` and `napoleon` extensions.
-   **Content**: Every public module, class, and function must have a docstring that explains:
    *   A one-line summary of its purpose.
    *   A more detailed explanation of what it does and why.
    *   Descriptions of all arguments (`Args:`).
    *   Description of the return value (`Returns:`).
    *   Any exceptions it might raise (`Raises:`).

### 6.3. Russian Language Comments

For internal clarity within a Russian-speaking team, comments should follow a strict protocol.

-   **Principle**: Use English for all code elements (variable names, function names, docstrings). Use Russian for inline comments that explain *why* a complex or non-obvious piece of logic exists. Do not explain *what* the code is doing—the code itself should be clear enough.
-   **Принцип**: Весь код (имена переменных, функций, классы) и докстринги должны быть на английском языке для соответствия международным стандартам. Внутристрочные комментарии на русском языке следует использовать исключительно для объяснения **причины** существования сложного, нетривиального или неочевидного участка кода, а не для описания **того, что** код делает. Код должен быть самодокументируемым.

**Good Example (Хороший пример):**
```python
# Рассчитываем лапласиан давления. Используем порядок аппроксимации 4,
# так как стандартный метод второго порядка дает слишком много шума на данных ERA5.
# This comment explains the "why" — why a 4th order approximation was chosen over the default.
pressure_laplacian = calculate_laplacian(pressure_field, order=4)
```
**Bad Example (Плохой пример):**
```python
# Цикл по всем временным шагам
# This comment is useless, it just describes what the code clearly shows.
for t in time_steps:
    # Найти циклоны
    find_cyclones(t)
```

## Code running
when run code you must activate env:
```bash
source venv/bin/activate
```