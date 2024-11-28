# Parvus - Quantum-Inspired Data Compression

A sophisticated data compression and similarity search system that uses quantum-inspired techniques for efficient data storage and retrieval. The name "Parvus" comes from the Latin word for "small" or "reduced", reflecting the system's primary purpose of data reduction while maintaining semantic meaning.

## Features

- **Quantum-Inspired Compression**: Utilizes advanced dimensionality reduction techniques
- **Semantic Search**: Performs similarity searches on compressed data
- **GPU Acceleration**: Supports GPU-accelerated processing for improved performance
- **Interactive GUI**: Streamlit-based UI for easy data manipulation
- **Flexible Input**: Supports both JSON and CSV file formats

## Installation

### From PyPI
```bash
# Basic installation
pip install parvus

# With GPU support
pip install parvus[gpu]

# For development
pip install parvus[dev]
```

### From Source
1. Clone the repository:
```bash
git clone https://github.com/yourusername/parvus.git
cd parvus
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended):
1. Install CUDA toolkit (if using NVIDIA GPU)
2. Install GPU-enabled packages:
```bash
conda install -c pytorch faiss-gpu
```

## Usage

### Command Line Interface

```bash
# Launch the GUI
python -m parvus --gui

# Start the API server
python -m parvus --server

# Compress a file
python -m parvus -i data.json -o compressed_output/

# Load compressed data and query
python -m parvus -l compressed_output/ -q "your search query"

# Show help
python -m parvus --help
```

### Starting the GUI

```bash
python -m parvus --gui
# or
python -m streamlit run gui.py
```

### Using the Python API

```python
from parvus import ParvusCompressor

# Initialize the system
compressor = ParvusCompressor()

# Load and compress data
embeddings = compressor.load_data_from_json('your_data.json')
compressor.compress(embeddings)

# Perform queries
results, distances = compressor.query("Your search query")
```

## System Requirements

- Python 3.8+
- RAM: 16GB recommended
- GPU: NVIDIA GPU with CUDA support (optional)
- Storage: Depends on dataset size

## Project Structure

```
parvus/
├── parvus.py              # Core compression engine
├── gui.py                 # Streamlit-based interface
├── server.py              # Server endpoints and API
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── CONTRIBUTING.md       # Contribution guidelines
├── data/                 # Sample and test data
│   ├── large_chat_history.json
│   ├── sample_data.csv
│   ├── sample_data.npy
│   └── test_data.json
├── models/               # Saved model states
│   ├── compressed_data.pkl
│   └── faiss_index.bin
├── tests/                # Test files and artifacts
│   ├── test_compressed.pkl
│   └── test_index.bin
└── archive/             # Archived/deprecated files
```

## Essential Components

1. **Core Files**:
   - `parvus.py`: Main compression engine implementing quantum-inspired algorithms
   - `gui.py`: Interactive web interface built with Streamlit
   - `server.py`: Server endpoints and API for integration
   - `requirements.txt`: Project dependencies

2. **Data Directory**:
   - Contains sample data files
   - Test datasets
   - JSON and CSV examples

3. **Models Directory**:
   - Saved compression states
   - FAISS indices
   - Serialized model data

4. **Tests Directory**:
   - Test artifacts
   - Compressed test data
   - Test indices

## Architecture

The system consists of three main components:

1. **Core Compression Engine** (`parvus.py`)
   - Handles data compression and decompression
   - Manages similarity search operations
   - Provides GPU acceleration when available

2. **Interactive Interface** (`gui.py`)
   - Web-based user interface
   - File upload and management
   - Query interface
   - Results visualization

3. **Server API** (`server.py`)
   - RESTful API endpoints
   - Data management
   - Remote compression operations

## Performance

Performance metrics on a typical dataset:
- Compression Ratio: ~5x (varies by data)
- Query Time: <100ms (with GPU)
- Memory Usage: Proportional to dataset size

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
