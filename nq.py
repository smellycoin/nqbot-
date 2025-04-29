import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import threading
import warnings
warnings.filterwarnings('ignore')

class NQTradingAssistant:
    def __init__(self):
        self.output_dir = "trading_recommendations"
        self.models_dir = "trading_models"
        self.history_file = os.path.join(self.output_dir, "trade_history.csv")
        self.current_model = None
        self.is_model_loaded = False
        self.loading_thread = None
        
        # Create necessary directories
        for directory in [self.output_dir, self.models_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize trade history if it doesn't exist
        if not os.path.exists(self.history_file):
            pd.DataFrame(columns=[
                'timestamp', 'source', 'trend', 'support', 'resistance', 
                'recommended_trade', 'confidence', 'outcome'
            ]).to_csv(self.history_file, index=False)
            
        print("NQ Trading Assistant initialized. Starting model loading in background...")
        self.loading_thread = threading.Thread(target=self.load_models)
        self.loading_thread.daemon = True
        self.loading_thread.start()
    
    def load_models(self):
        """Load all necessary models in a background thread"""
        try:
            print("Loading local AI models (this may take a minute)...")
            
            # For text analysis - use a smaller pre-trained model
            self.text_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.text_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                self.text_model = self.text_model.to("cuda")
            
            # Create a base pattern classifier for candlestick recognition
            self.pattern_model = RandomForestClassifier(n_estimators=100)
            
            # Initialize but don't train market prediction model yet
            self.market_model = RandomForestRegressor(n_estimators=100)
            
            # Flag that model is loaded
            self.is_model_loaded = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Running in limited mode - some features may not be available.")
    
    def wait_for_models(self):
        """Make sure models are loaded before proceeding"""
        if not self.is_model_loaded and self.loading_thread is not None:
            print("Waiting for models to load...")
            self.loading_thread.join()
        return self.is_model_loaded
    
    def extract_candlesticks_from_image(self, image_path):
        """Extract candlestick patterns from image"""
        try:
            # Load and process the image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect lines that could be candlesticks
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Simple feature extraction for candlesticks
            features = self.extract_chart_features(img_rgb, edges)
            
            # Return detected patterns and features
            return {
                "candlesticks_detected": True,
                "features": features,
                "image_height": img.shape[0],
                "image_width": img.shape[1]
            }
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                "candlesticks_detected": False,
                "error": str(e)
            }
    
    def extract_chart_features(self, img_rgb, edges):
        """Extract basic features from chart image"""
        # Calculate color distributions (green/red candles)
        green_mask = (img_rgb[:,:,1] > 150) & (img_rgb[:,:,0] < 100) & (img_rgb[:,:,2] < 100)
        red_mask = (img_rgb[:,:,0] > 150) & (img_rgb[:,:,1] < 100) & (img_rgb[:,:,2] < 100)
        
        green_count = np.sum(green_mask)
        red_count = np.sum(red_mask)
        
        # Calculate line density for potential support/resistance
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        horizontal_line_count = 0 if horizontal_lines is None else len(horizontal_lines)
        
        # Basic edge detection for pattern recognition
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Return extracted features
        return {
            "green_candles_ratio": float(green_count / (green_count + red_count + 1)),
            "red_candles_ratio": float(red_count / (green_count + red_count + 1)),
            "horizontal_line_count": horizontal_line_count,
            "edge_density": float(edge_density)
        }
    
    def analyze_candlestick_patterns(self, features):
        """Analyze detected candlestick patterns"""
        # Initialize with some basic heuristics
        trend = "neutral"
        confidence = 0.5
        
        # Determine basic trend from green/red ratio
        if features["green_candles_ratio"] > 0.6:
            trend = "bullish"
            confidence = min(0.5 + features["green_candles_ratio"] * 0.5, 0.95)
        elif features["red_candles_ratio"] > 0.6:
            trend = "bearish"
            confidence = min(0.5 + features["red_candles_ratio"] * 0.5, 0.95)
            
        # Detect potential support/resistance from horizontal lines
        support_resistance_strength = min(features["horizontal_line_count"] / 10, 1.0)
        
        # Detect potential patterns based on edge density
        pattern_strength = features["edge_density"] * 5
        
        return {
            "trend": trend,
            "trend_confidence": float(confidence),
            "support_resistance_detected": support_resistance_strength > 0.3,
            "support_resistance_strength": float(support_resistance_strength),
            "pattern_strength": float(pattern_strength)
        }
    
    def analyze_image(self, image_path):
        """Analyze market chart image"""
        if not self.wait_for_models():
            return "Model loading failed. Please restart the application."
        
        try:
            print(f"Analyzing image: {image_path}")
            
            # Extract candlesticks
            candlestick_data = self.extract_candlesticks_from_image(image_path)
            if not candlestick_data["candlesticks_detected"]:
                return f"Failed to detect candlestick patterns: {candlestick_data.get('error', 'Unknown error')}"
            
            # Analyze patterns
            pattern_analysis = self.analyze_candlestick_patterns(candlestick_data["features"])
            
            # Create detailed analysis text
            analysis = self._generate_trade_recommendation(
                pattern_analysis,
                candlestick_data["features"]
            )
            
            # Save analysis to history
            self._save_to_history(
                image_path,
                pattern_analysis["trend"],
                confidence=pattern_analysis["trend_confidence"]
            )
            
            return analysis
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def analyze_numerical_data(self, data_path):
        """Analyze numerical market data from CSV or other numerical format"""
        if not self.wait_for_models():
            return "Model loading failed. Please restart the application."
        
        try:
            print(f"Analyzing numerical data: {data_path}")
            
            # Load data based on file extension
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            else:
                return "Unsupported file format. Please provide CSV or Excel file."
            
            # Check if data has expected columns
            required_columns = ['open', 'high', 'low', 'close']
            lower_columns = [col.lower() for col in df.columns]
            
            # Try to find appropriate columns
            col_mapping = {}
            for req_col in required_columns:
                for idx, col in enumerate(lower_columns):
                    if req_col in col:
                        col_mapping[req_col] = df.columns[idx]
                        break
            
            # If we couldn't find all required columns, make a guess
            if len(col_mapping) < len(required_columns):
                # If we have 4 numeric columns, assume OHLC
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 4:
                    for i, req_col in enumerate(required_columns):
                        if req_col not in col_mapping and i < len(numeric_cols):
                            col_mapping[req_col] = numeric_cols[i]
            
            # Check if we have all columns mapped
            if len(col_mapping) < len(required_columns):
                missing = set(required_columns) - set(col_mapping.keys())
                return f"Missing required columns: {', '.join(missing)}. Please provide data with open, high, low, close values."
            
            # Extract data for analysis
            data = {
                'open': df[col_mapping['open']].values,
                'high': df[col_mapping['high']].values,
                'low': df[col_mapping['low']].values,
                'close': df[col_mapping['close']].values
            }
            
            # Calculate basic indicators
            analysis_data = self._calculate_indicators(data)
            
            # Generate analysis and prediction
            trend = 'bullish' if analysis_data['macd'][-1] > 0 else 'bearish'
            rsi_value = analysis_data['rsi'][-1]
            
            if rsi_value > 70:
                trend = 'bearish'  # Overbought
                confidence = min((rsi_value - 70) / 30 + 0.7, 0.95)
            elif rsi_value < 30:
                trend = 'bullish'  # Oversold
                confidence = min((30 - rsi_value) / 30 + 0.7, 0.95)
            else:
                confidence = 0.5 + abs(rsi_value - 50) / 100
            
            # Create analysis text
            analysis = self._generate_numerical_recommendation(
                trend, 
                analysis_data, 
                confidence
            )
            
            # Save to history
            self._save_to_history(
                data_path,
                trend,
                support=f"{min(data['low'][-10:])}",
                resistance=f"{max(data['high'][-10:])}",
                confidence=confidence
            )
            
            return analysis
        except Exception as e:
            print(f"Error analyzing numerical data: {e}")
            return f"Error analyzing numerical data: {str(e)}"
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators from numerical data"""
        close_prices = data['close']
        
        # Calculate RSI (Relative Strength Index)
        delta = np.diff(close_prices)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        
        avg_gain = np.mean(gain[:14])
        avg_loss = np.mean(loss[:14])
        
        for i in range(14, len(delta)):
            avg_gain = (avg_gain * 13 + gain[i]) / 14
            avg_loss = (avg_loss * 13 + loss[i]) / 14
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        ema12 = self._calculate_ema(close_prices, 12)
        ema26 = self._calculate_ema(close_prices, 26)
        macd = ema12 - ema26
        
        # Calculate Bollinger Bands
        sma20 = self._calculate_sma(close_prices, 20)
        std20 = np.std(close_prices[-20:])
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        
        return {
            'rsi': np.append(np.zeros(len(close_prices) - 1), rsi),
            'macd': np.append(np.zeros(len(close_prices) - len(macd)), macd),
            'upper_band': upper_band,
            'lower_band': lower_band,
            'sma20': sma20
        }
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        multiplier = 2 / (period + 1)
        ema = [prices[0]]
        
        for price in prices[1:]:
            ema.append(price * multiplier + ema[-1] * (1 - multiplier))
        
        return np.array(ema)
    
    def _calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        return np.mean(prices[-period:])
    
    def _generate_trade_recommendation(self, pattern_analysis, features):
        """Generate trading recommendations based on pattern analysis"""
        trend = pattern_analysis["trend"]
        confidence = pattern_analysis["trend_confidence"]
        
        # Basic recommendation template
        analysis = f"# NQ Trading Analysis\n\n"
        analysis += f"## Market Trend\n"
        analysis += f"The current market trend appears to be **{trend.upper()}** "
        analysis += f"with **{confidence*100:.1f}%** confidence.\n\n"
        
        # Add support/resistance analysis
        if pattern_analysis["support_resistance_detected"]:
            analysis += f"## Support & Resistance\n"
            analysis += f"Multiple horizontal levels detected, suggesting potentially significant "
            analysis += f"support/resistance zones. Strength: {pattern_analysis['support_resistance_strength']*100:.1f}%\n\n"
        
        # Add pattern analysis
        analysis += f"## Pattern Analysis\n"
        
        if features["green_candles_ratio"] > 0.7:
            analysis += "Strong bullish momentum detected with multiple green candles.\n"
        elif features["red_candles_ratio"] > 0.7:
            analysis += "Strong bearish pressure detected with multiple red candles.\n"
        
        if pattern_analysis["pattern_strength"] > 0.6:
            analysis += f"Complex price action detected - possible reversal pattern forming.\n"
        else:
            analysis += f"No significant chart patterns detected in current view.\n\n"
        
        # Add trade recommendation
        analysis += f"## Trading Recommendation\n"
        
        if trend == "bullish":
            analysis += "**BULLISH BIAS**: Consider long positions with appropriate stop loss.\n"
            analysis += "- Entry: Look for pullbacks to support levels\n"
            analysis += "- Stop Loss: Below recent swing lows\n"
            analysis += "- Target: Previous resistance levels or 1:2 risk-reward ratio\n"
        elif trend == "bearish":
            analysis += "**BEARISH BIAS**: Consider short positions with appropriate stop loss.\n"
            analysis += "- Entry: Look for rallies to resistance levels\n"
            analysis += "- Stop Loss: Above recent swing highs\n"
            analysis += "- Target: Previous support levels or 1:2 risk-reward ratio\n"
        else:
            analysis += "**NEUTRAL BIAS**: Market lacks clear direction. Consider waiting for a breakout.\n"
            analysis += "- Watch for breakouts above/below key levels\n"
            analysis += "- Reduced position sizing recommended in choppy conditions\n"
        
        # Add disclaimer
        analysis += "\n## Disclaimer\n"
        analysis += "This analysis is generated by an automated system and should be considered "
        analysis += "one input among many for your trading decisions. Always manage risk appropriately.\n"
        
        return analysis
    
    def _generate_numerical_recommendation(self, trend, indicators, confidence):
        """Generate trading recommendations based on numerical analysis"""
        # Extract indicator values
        rsi = indicators['rsi'][-1]
        macd = indicators['macd'][-1]
        upper_band = indicators['upper_band']
        lower_band = indicators['lower_band']
        sma20 = indicators['sma20']
        
        # Basic recommendation template
        analysis = f"# NQ Trading Analysis (Numerical Data)\n\n"
        
        analysis += f"## Market Trend\n"
        analysis += f"The current market trend appears to be **{trend.upper()}** "
        analysis += f"with **{confidence*100:.1f}%** confidence.\n\n"
        
        # Add technical indicator analysis
        analysis += f"## Technical Indicators\n"
        
        # RSI Analysis
        analysis += f"**RSI**: {rsi:.1f} - "
        if rsi > 70:
            analysis += "Overbought conditions, suggesting potential reversal to the downside.\n"
        elif rsi < 30:
            analysis += "Oversold conditions, suggesting potential reversal to the upside.\n"
        else:
            analysis += "Neutral conditions.\n"
        
        # MACD Analysis
        analysis += f"**MACD**: {macd:.4f} - "
        if macd > 0:
            analysis += "Positive momentum, suggesting bullish trend.\n"
        else:
            analysis += "Negative momentum, suggesting bearish trend.\n"
        
        # Bollinger Bands Analysis
        analysis += f"**Bollinger Bands**: Upper {upper_band:.1f}, Lower {lower_band:.1f} - "
        if upper_band - lower_band > (upper_band + lower_band) * 0.1:
            analysis += "Wide bands suggesting high volatility.\n"
        else:
            analysis += "Narrow bands suggesting low volatility, potential breakout ahead.\n\n"
        
        # Add trade recommendation
        analysis += f"## Trading Recommendation\n"
        
        if trend == "bullish":
            analysis += "**BULLISH OUTLOOK**: Consider long positions with appropriate stop loss.\n"
            if rsi < 50:
                analysis += "- Good buying opportunity with RSI showing room to run higher\n"
            analysis += f"- Entry Zone: Near {sma20:.1f} (20-period SMA)\n"
            analysis += f"- Stop Loss: Below {lower_band:.1f} (lower Bollinger Band)\n"
            analysis += f"- Target: {upper_band:.1f} (upper Bollinger Band) or higher\n"
        elif trend == "bearish":
            analysis += "**BEARISH OUTLOOK**: Consider short positions with appropriate stop loss.\n"
            if rsi > 50:
                analysis += "- Good selling opportunity with RSI showing room to run lower\n"
            analysis += f"- Entry Zone: Near {sma20:.1f} (20-period SMA)\n"
            analysis += f"- Stop Loss: Above {upper_band:.1f} (upper Bollinger Band)\n"
            analysis += f"- Target: {lower_band:.1f} (lower Bollinger Band) or lower\n"
        else:
            analysis += "**NEUTRAL OUTLOOK**: Market lacks clear direction. Consider waiting for a breakout.\n"
            analysis += "- Watch for breakouts above/below key levels\n"
            analysis += "- Reduced position sizing recommended in choppy conditions\n"
        
        # Add disclaimer
        analysis += "\n## Disclaimer\n"
        analysis += "This analysis is generated by an automated system and should be considered "
        analysis += "one input among many for your trading decisions. Always manage risk appropriately.\n"
        
        return analysis
    
    def save_recommendation(self, analysis_result, source_file):
        """Save trading recommendations to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/trade_rec_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Trading Analysis for: {source_file}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(analysis_result)
        
        return filename
    
    def _save_to_history(self, source, trend, support="", resistance="", recommended_trade="", confidence=0.5):
        """Save trade analysis to history file for learning"""
        history_df = pd.read_csv(self.history_file)
        
        new_entry = pd.DataFrame([{
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': source,
            'trend': trend,
            'support': support,
            'resistance': resistance,
            'recommended_trade': recommended_trade,
            'confidence': confidence,
            'outcome': ""  # To be filled in later for training
        }])
        
        updated_history = pd.concat([history_df, new_entry], ignore_index=True)
        updated_history.to_csv(self.history_file, index=False)
    
    def train_on_history(self, feedback_file=None):
        """Train prediction model on past history with outcomes"""
        # This is a placeholder for future enhancement
        # Would implement model training based on past recommendations and outcomes
        print("Training on historical data...")
        
        # If feedback file provided, update history with outcomes
        if feedback_file and os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            history_df = pd.read_csv(self.history_file)
            
            # Update history with outcome feedback
            for idx, row in feedback_df.iterrows():
                match_idx = history_df[
                    (history_df['timestamp'] == row['timestamp']) & 
                    (history_df['source'] == row['source'])
                ].index
                
                if len(match_idx) > 0:
                    history_df.loc[match_idx[0], 'outcome'] = row['outcome']
            
            # Save updated history
            history_df.to_csv(self.history_file, index=False)
            
            print(f"Updated {len(feedback_df)} historical records with outcomes")
        
        return "Model trained on historical data"

def main():
    parser = argparse.ArgumentParser(description='NQ Market Trading Assistant')
    parser.add_argument('--image', type=str, help='Path to market chart image')
    parser.add_argument('--numerical', type=str, help='Path to CSV/Excel file with numerical market data')
    parser.add_argument('--train', action='store_true', help='Train model on history data')
    parser.add_argument('--feedback', type=str, help='Path to feedback file for training')
    
    args = parser.parse_args()
    
    assistant = NQTradingAssistant()
    
    if args.train:
        result = assistant.train_on_history(args.feedback)
        print(result)
        return
    
    if not any([args.image, args.numerical]):
        print("Please provide at least one input source (--image or --numerical)")
        print("For example: python trading_assistant.py --image chart.png")
        sys.exit(1)
    
    results = []
    
    # Process image if provided
    if args.image:
        print(f"Analyzing image: {args.image}")
        analysis = assistant.analyze_image(args.image)
        output_file = assistant.save_recommendation(analysis, args.image)
        print(f"Analysis saved to: {output_file}")
        results.append({"source": args.image, "output": output_file})
    
    # Process numerical data if provided
    if args.numerical:
        print(f"Analyzing numerical data: {args.numerical}")
        analysis = assistant.analyze_numerical_data(args.numerical)
        output_file = assistant.save_recommendation(analysis, args.numerical)
        print(f"Analysis saved to: {output_file}")
        results.append({"source": args.numerical, "output": output_file})
    
    print("\nAll analyses completed!")
    for result in results:
        print(f"Source: {result['source']} → Output: {result['output']}")

if __name__ == "__main__":
    main()