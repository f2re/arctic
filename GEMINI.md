# **ArcticCyclone Developer and Contribution Guide (GEMINI Edition)**

## 1. Project Philosophy and Scientific Rigor

**ArcticCyclone** is a scientific software framework designed for the reproducible and verifiable study of Arctic mesocyclones. The core philosophy is to blend meteorological science with robust software engineering practices. Every component, from data acquisition to analysis, must be implemented with transparency, modularity, and extensibility in mind.

All algorithms and methods should be grounded in established atmospheric physics. The implementation must facilitate sensitivity analyses, validation against independent datasets, and comparison with other published methods.

## 2. Project Structure

The project follows a flat structure with source code in the root directory. This is a key difference from the `src` layout proposed in `QWEN.md`.

```
arctic-cyclone/
├── analysis/
│   ├── climatology.py
│   └── statistics.py
├── core/
│   ├── config.py       # Handles configuration from config.yaml
│   ├── exceptions.py
│   └── logging_setup.py
├── data/
│   ├── acquisition.py
│   ├── catalog.py
│   └── processors/
├── detection/
│   ├── tracker.py
│   ├── algorithms/
│   │   ├── algorithm_factory.py
│   │   └── ... (other algorithms)
│   └── criteria/
├── export/
│   └── formats/
├── models/
│   ├── cyclone.py
│   └── ...
├── visualization/
│   ├── mappers.py
│   └── tracks.py
├── .gitignore
├── config.yaml                 # Main configuration file
├── main.py                     # Main entry point for the workflow
├── requirements.txt            # Project dependencies
└── README.md
```

## 3. Analysis of Detection Methods

Each detection method must be understood by its physical basis and its implementation trade-offs. The following algorithms are available through the `AlgorithmFactory`:

1.  **`pressure_minima`**:
    *   **Physical Basis**: Cyclones are fundamentally low-pressure systems. This is the most direct and common detection method.
    *   **Implementation**: Uses a 2D filter to find local pressure minima.
    *   **Strengths**: Simple, computationally efficient.
    *   **Weaknesses**: Prone to identifying minor, insignificant troughs.

2.  **`multi_parameter`**:
    *   **Physical Basis**: Combines multiple parameters (e.g., pressure, vorticity, wind speed) to identify cyclones.
    *   **Implementation**: A weighted combination of different criteria.
    *   **Strengths**: More robust than single-parameter methods.
    *   **Weaknesses**: Requires careful tuning of weights.

3.  **`arctic_mesocyclone`**:
    *   **Physical Basis**: Specifically tuned for the characteristics of Arctic mesocyclones.
    *   **Implementation**: Likely a combination of criteria with thresholds adapted for the Arctic region.
    *   **Strengths**: Better performance for the target cyclone type.
    *   **Weaknesses**: May not be suitable for other types of cyclones.

4.  **`serreze`**:
    *   **Physical Basis**: Based on the Serreze et al. (1997) algorithm, which is a well-known method for cyclone detection.
    *   **Implementation**: Follows the logic of the published algorithm.
    *   **Strengths**: Reproducible and comparable with other studies.
    *   **Weaknesses**: May be outdated compared to more modern methods.

## 4. Python Standards and Best Practices

### 4.1. Project and Dependency Management
-   **Use `requirements.txt`**: This file lists all project dependencies.
-   **Virtual Environments**: All development must occur within a dedicated virtual environment (e.g., using `venv` or `conda`) to isolate dependencies.

### 4.2. Code Formatting and Linting
-   The project does not currently enforce a strict formatting or linting standard. It is recommended to use **Black** for formatting and **Ruff** or **Flake8** for linting to improve code quality and consistency.

### 4.3. Typing
-   The project uses Python's `typing` module, but not consistently. It is recommended to add type hints to all function signatures and class variables to improve code clarity and catch errors early.

### 4.4. Configuration
-   The `core/config.py` module uses the `ConfigManager` class to load and validate the `config.yaml`. This ensures that the configuration is type-safe and all required fields are present at startup.

## 5. Testing Strategy

The project does not currently have a dedicated test suite. It is highly recommended to add a `tests/` directory with unit and integration tests using the **`pytest`** framework. This will improve the reliability and maintainability of the code.

## 6. Documentation Principles

### 6.1. Docstrings (In-Code Documentation)
-   The project uses docstrings, but the style is not consistent. It is recommended to use a standard format like **Google Style** or **NumPy Style** docstrings to allow for automatic generation of API documentation.

### 6.2. Russian Language Comments
-   The codebase contains comments and docstrings in both English and Russian. For internal clarity within a Russian-speaking team, comments should follow a strict protocol.
-   **Principle**: Use English for all code elements (variable names, function names, docstrings). Use Russian for inline comments that explain *why* a complex or non-obvious piece of logic exists. Do not explain *what* the code is doing—the code itself should be clear enough.

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

## 7. Code Running

When running Python files, you must always work within a virtual environment to ensure you're using the correct dependencies and to avoid conflicts with system packages.

### 7.1. Setting Up the Virtual Environment

If a virtual environment doesn't exist yet, create one:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the environment (for bash/zsh users)
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 7.2. Running the Main Workflow

The `main.py` script is the main entry point for the workflow. It can be run from the command line with the following arguments:

```bash
python main.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --output-dir output
```

### 7.3. Debugging and Development Options

For development and debugging, you can enable additional options:

```bash
# Enable debug tracking mode (saves intermediate CSV files)
python main.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --debug-tracking

# Enable debug plotting mode (saves additional diagnostic plots)
python main.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --debug-plot

# Enable detailed logging
python main.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --log-level DEBUG
```

to run python script use existed venv/bin/activate environment