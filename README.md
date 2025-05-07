markdown
# NQ Trading Assistant

An AI-powered trading analysis tool that provides market insights from chart images or numerical data, with optional lightweight LLM integration for enhanced analysis.

## Features

- **Chart Image Analysis**: Detects trends, support/resistance, and liquidity zones from candlestick charts
- **Numerical Data Processing**: Analyzes OHLC data to identify trends and key levels
- **Liquidity Detection**: Identifies potential liquidity zones in price action
- **Optional LLM Integration**: Get AI-powered market insights (requires `transformers` package)
- **Historical Learning**: Improves recommendations based on past analysis

## Installation

### Minimal Installation (No LLM)
```bash
pip install numpy pandas matplotlib pillow
Full Installation (With LLM Support)
bash
pip install numpy pandas matplotlib pillow transformers torch
Usage
Basic Usage (Without LLM)
bash
# Analyze a chart image
python trading_assistant.py --image path/to/chart.png

# Analyze numerical data (CSV/Excel)
python trading_assistant.py --numerical path/to/data.csv

# Plot numerical data
python trading_assistant.py --numerical path/to/data.csv --plot
Advanced Usage (With LLM)
bash
# Analyze with LLM insights
python trading_assistant.py --image path/to/chart.png --use_llm

# Analyze numerical data with LLM
python trading_assistant.py --numerical path/to/data.csv --use_llm
Training Mode
bash
# Train on historical data
python trading_assistant.py --train

# Train with feedback data
python trading_assistant.py --train --feedback path/to/feedback.csv
Input Formats
Image Analysis
Supports PNG, JPG, BMP formats

Recommended: Clean candlestick charts (600x400px or larger)

Numerical Data
CSV or Excel files

Should contain OHLC columns (Open, High, Low, Close)

Column names auto-detected (case insensitive)

Output
Analysis results are saved in:

./trading_recommendations/ (text reports)

./trading_models/ (learned models)

Performance Notes
Without LLM: Runs instantly on any modern computer

With LLM: First run will download ~300MB model (one-time)

Subsequent LLM runs use ~1GB RAM

Troubleshooting
LLM not working?

bash
# Try with CPU-only (if GPU issues occur)
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
Missing dependencies?

bash
pip install --upgrade -r requirements.txt
License
MIT License - Free for personal and commercial use

