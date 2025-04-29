# Standalone NQ Trading Assistant - Simple Setup Guide

## Overview

This trading assistant analyzes NQ market data locally with no API keys required:
- Analyzes candlestick chart images (screenshots from trading platforms)
- Processes numerical data (CSV/Excel files with OHLC data)
- Provides trading recommendations, including trend analysis, support/resistance levels, and trade ideas
- Learns from past analyses to improve recommendations

## Quick Setup

### 1. Install Required Libraries

```bash
pip install numpy pandas matplotlib pillow opencv-python transformers torch scikit-learn
```

### 2. Run the Assistant

For chart images:
```bash
python trading_assistant.py --image path/to/chart_screenshot.jpg
```

For numerical data (CSV/Excel with OHLC data):
```bash
python trading_assistant.py --numerical path/to/market_data.csv
```

That's it! No API keys or external services needed.

## Features

- **Fully Local Processing**: All analysis happens on your computer
- **Self-Improving**: Records analyses and can learn from feedback
- **Quick Results**: Generates recommendations in seconds
- **Multiple Input Types**: Works with both chart images and numerical data
- **Detailed Analysis**: Includes trend identification, support/resistance levels, and specific entry/exit points

## Output

The assistant creates a timestamped recommendation file in the `trading_recommendations` folder with:
- Market trend analysis (bullish, bearish, neutral)
- Technical indicator readings (RSI, MACD, Bollinger Bands)
- Support and resistance levels
- Trading recommendations with entry/exit points

## Learning Capabilities

The assistant stores all analyses in a history file and can improve over time. To help it learn:

1. Track which recommendations were successful
2. Create a feedback CSV with columns: timestamp, source, outcome
3. Run with the training flag:
   ```bash
   python trading_assistant.py --train --feedback my_feedback.csv
   ```

## Example

```bash
python trading_assistant.py --image nq_chart.png
```

Output will show:
```
NQ Trading Assistant initialized. Starting model loading in background...
Loading local AI models (this may take a minute)...
Models loaded successfully!
Analyzing image: nq_chart.png
Analysis saved to: trading_recommendations/trade_rec_20250429_153045.txt

All analyses completed!
Source: nq_chart.png → Output: trading_recommendations/trade_rec_20250429_153045.txt
```

## Notes

- For best results with image analysis, use clear screenshots of candlestick charts
- For numerical data, make sure your CSV/Excel has columns for open, high, low, and close prices
- This system uses pattern recognition and technical indicators - always use your own judgment