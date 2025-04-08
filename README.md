# Blueprint Analysis

A powerful tool for analyzing architectural and electrical blueprints to identify and count specific symbols with high accuracy.

## Features
- PDF blueprint processing
- Intelligent symbol detection
- Multi-scale analysis
- Boundary symbol handling
- Detailed reporting
- Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dhatricds/Blueprint_Analysis.git
cd Blueprint_Analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Basic Symbol Detection
```bash
python process_pdf.py --blueprint path/to/blueprint.pdf --symbol path/to/symbol.png
```

### Generate Report
```bash
python generate_report.py --results path/to/results.json
```

### Run Tests
```bash
python run_tests.py
```

## Project Structure
```
├── data/               # Input data directory
│   ├── blueprints/    # Blueprint PDFs
│   └── symbols/       # Reference symbols
├── output/            # Output directory
│   ├── tiles/        # Generated tiles
│   ├── results/      # Detection results
│   ├── reports/      # Generated reports
│   └── debug/        # Debug visualizations
├── src/              # Source code
├── tests/            # Test files
└── docs/             # Documentation
```

## Configuration

Key settings in `.env`:
- `API_KEY`: Your API key
- `TILE_SIZE`: Size of tiles (default: 500)
- `OVERLAP`: Tile overlap (default: 100)
- `THRESHOLD`: Detection threshold (default: 0.6)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository.