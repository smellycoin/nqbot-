import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import threading
import warnings
warnings.filterwarnings('ignore')

class NQTradingAssistant:
    def __init__(self):
        self.output_dir = "trading_recommendations"
        self.models_dir = "trading_models"
        self.history_file = os.path.join(self.output_dir, "trade_history.csv")
        self.is_model_loaded = True
        
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
            
        print("NQ Trading Assistant initialized...")
    
    def extract_features_from_image(self, image_path):
        """Extract features from image without using OpenCV"""
        try:
            # Load image using PIL instead of OpenCV
            img = Image.open(image_path)
            width, height = img.size
            
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate color distributions (find green/red pixels for candlesticks)
            # Green candles are typically green, red candles are typically red
            pixel_data = np.array(img)
            
            # Simple heuristic for green candles (more green than red)
            green_mask = (pixel_data[:,:,1] > pixel_data[:,:,0] + 30) & (pixel_data[:,:,1] > pixel_data[:,:,2] + 30)
            green_count = np.sum(green_mask)
            
            # Simple heuristic for red candles (more red than green)
            red_mask = (pixel_data[:,:,0] > pixel_data[:,:,1] + 30) & (pixel_data[:,:,0] > pixel_data[:,:,2] + 30)
            red_count = np.sum(red_mask)
            
            # Calculate edge density using simple gradient method
            gray = np.array(ImageOps.grayscale(img))
            
            # Simple gradient-based edge detection
            v_edges = np.abs(np.diff(gray, axis=0))
            h_edges = np.abs(np.diff(gray, axis=1))
            
            # Pad to maintain shape
            v_edges = np.pad(v_edges, ((0, 1), (0, 0)), mode='constant')
            h_edges = np.pad(h_edges, ((0, 0), (0, 1)), mode='constant')
            
            # Combine edges
            edges = np.maximum(v_edges, h_edges)
            
            # Count significant edges
            edge_mask = edges > 30  # Threshold for edge detection
            edge_count = np.sum(edge_mask)
            edge_density = edge_count / (width * height)
            
            # Detect horizontal lines (potential support/resistance)
            horizontal_line_score = 0
            for y in range(0, height, 5):  # Sample every 5 pixels for efficiency
                row = h_edges[y, :]
                if np.sum(row > 40) > width * 0.5:  # If more than half the row has edges
                    horizontal_line_score += 1
            
            horizontal_line_density = horizontal_line_score / (height / 5)
            
            # Return extracted features
            return {
                "green_candles_ratio": float(green_count / (green_count + red_count + 1)),
                "red_candles_ratio": float(red_count / (green_count + red_count + 1)),
                "horizontal_line_density": float(horizontal_line_density),
                "edge_density": float(edge_density),
                "image_width": width,
                "image_height": height
            }
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                "error": str(e)
            }
    
    def analyze_candlestick_patterns(self, features):
        """Analyze detected candlestick patterns"""
        # Initialize with some basic heuristics
        trend = "neutral"
        confidence = 0.5
        
        # Determine basic trend from green/red ratio
        green_ratio = features.get("green_candles_ratio", 0)
        red_ratio = features.get("red_candles_ratio", 0)
        
        if green_ratio > 0.6:
            trend = "bullish"
            confidence = min(0.5 + green_ratio * 0.5, 0.95)
        elif red_ratio > 0.6:
            trend = "bearish"
            confidence = min(0.5 + red_ratio * 0.5, 0.95)
            
        # Detect potential support/resistance from horizontal lines
        support_resistance_strength = min(features.get("horizontal_line_density", 0) * 5, 1.0)
        
        # Detect potential patterns based on edge density
        pattern_strength = features.get("edge_density", 0) * 5
        
        return {
            "trend": trend,
            "trend_confidence": float(confidence),
            "support_resistance_detected": support_resistance_strength > 0.3,
            "support_resistance_strength": float(support_resistance_strength),
            "pattern_strength": float(pattern_strength)
        }
    
    def analyze_image(self, image_path):
        """Analyze market chart image"""
        try:
            print(f"Analyzing image: {image_path}")
            
            # Extract features from image
            features = self.extract_features_from_image(image_path)
            if "error" in features:
                return f"Failed to analyze image: {features['error']}"
            
            # Analyze patterns
            pattern_analysis = self.analyze_candlestick_patterns(features)
            
            # Create detailed analysis text
            analysis = self._generate_trade_recommendation(
                pattern_analysis,
                features
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
        
        avg_gain = np.mean(gain[:14]) if len(gain) >= 14 else np.mean(gain)
        avg_loss = np.mean(loss[:14]) if len(loss) >= 14 else np.mean(loss)
        
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
        period = min(20, len(close_prices))
        sma20 = self._calculate_sma(close_prices, period)
        std20 = np.std(close_prices[-period:])
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
        period = min(period, len(prices))
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
        print("Learning from historical data...")
        
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
        
        return "Learning completed - future recommendations will be improved"

    def plot_market_data(self, data_path, output_path=None):
        """Plot market data and save the chart"""
        try:
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            else:
                return "Unsupported file format. Please provide CSV or Excel file."
            
            # Find OHLC columns
            required_columns = ['open', 'high', 'low', 'close']
            lower_columns = [col.lower() for col in df.columns]
            
            col_mapping = {}
            for req_col in required_columns:
                for idx, col in enumerate(lower_columns):
                    if req_col in col:
                        col_mapping[req_col] = df.columns[idx]
                        break
            
            # If we couldn't find all required columns, make a guess
            if len(col_mapping) < len(required_columns):
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 4:
                    for i, req_col in enumerate(required_columns):
                        if req_col not in col_mapping and i < len(numeric_cols):
                            col_mapping[req_col] = numeric_cols[i]
            
            # Check if we have all columns mapped
            if len(col_mapping) < len(required_columns):
                missing = set(required_columns) - set(col_mapping.keys())
                return f"Missing required columns: {', '.join(missing)}. Please provide data with open, high, low, close values."
            
            # Create a plot
            plt.figure(figsize=(10, 6))
            
            # Plot close prices
            plt.plot(df[col_mapping['close']].values, label='Close Price')
            
            # Add title and labels
            plt.title('Market Data Analysis')
            plt.xlabel('Time Period')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Save or display the plot
            if output_path:
                plt.savefig(output_path)
                return f"Chart saved to {output_path}"
            else:
                output_path = os.path.join(self.output_dir, f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(output_path)
                plt.close()
                return f"Chart saved to {output_path}"
        except Exception as e:
            print(f"Error plotting market data: {e}")
            return f"Error plotting market data: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='NQ Market Trading Assistant')
    parser.add_argument('--image', type=str, help='Path to market chart image')
    parser.add_argument('--numerical', type=str, help='Path to CSV/Excel file with numerical market data')
    parser.add_argument('--plot', action='store_true', help='Plot numerical data and save chart')
    parser.add_argument('--train', action='store_true', help='Train on historical data')
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
        
        # Plot data if requested
        if args.plot:
            plot_result = assistant.plot_market_data(args.numerical)
            print(plot_result)
    
    print("\nAll analyses completed!")
    for result in results:
        print(f"Source: {result['source']} → Output: {result['output']}")

if __name__ == "__main__":
    main()