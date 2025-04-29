# Super Simple NQ Trading Assistant - Setup Guide

## Overview

This lightweight trading assistant analyzes NQ market data with minimal dependencies:
- Analyzes candlestick chart images (screenshots from trading platforms)
- Processes numerical data (CSV/Excel files with OHLC data)
- Provides trading recommendations with no external API dependencies
- Learns from past analyses to improve recommendations over time

## Minimal Dependencies Setup

### Install Required Libraries

```bash
pip install numpy pandas matplotlib pillow
```

That's it! No OpenCV, no TensorFlow, no external API keys needed.

## Quick Start

### For Chart Images:
```bash
python trading_assistant.py --image path/to/chart_screenshot.jpg
```

### For Numerical Data (CSV/Excel with OHLC data):
```bash
python trading_assistant.py --numerical path/to/market_data.csv
```

### Plot Your Data (Optional):
```bash
python trading_assistant.py --numerical market_data.csv --plot
```

## Features

- **Zero External Dependencies**: All analysis happens on your computer with minimal libraries
- **Self-Improving**: Records analyses and learns from feedback
- **Instant Results**: Generates recommendations in seconds
- **Multiple Input Types**: Works with both chart images and numerical data
- **Detailed Analysis**: Includes trend identification, support/resistance levels, and trade recommendations

## Output

The assistant creates a timestamped recommendation file in the `trading_recommendations` folder with:
- Market trend analysis (bullish, bearish, neutral)
- Technical indicator readings (when using numerical data)
- Support and resistance levels
- Trading recommendations with entry/exit points

## Learning Capabilities

The assistant automatically improves over time. To accelerate learning:

1. Create a simple feedback CSV with columns: timestamp, source, outcome
2. Run with the training flag:
   ```bash
   python trading_assistant.py --train --feedback my_feedback.csv
   ```

## Example Output

```
NQ Trading Assistant initialized...
Analyzing image: nq_chart.png
Analysis saved to: trading_recommendations/trade_rec_20250429_153045.txt

All analyses completed!
Source: nq_chart.png → Output: trading_recommendations/trade_rec_20250429_153045.txt
```

## Notes

- For best results with image analysis, use clear screenshots with visible candlesticks
- For numerical data, make sure your CSV/Excel has columns for open, high, low, and close prices
- The system automatically adapts to find the right columns in your data