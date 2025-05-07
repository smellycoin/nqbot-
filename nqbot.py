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

# Optional LLM components
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pass

class NQTradingAssistant:
    def __init__(self, use_llm=False):
        self.output_dir = "trading_recommendations"
        self.models_dir = "trading_models"
        self.history_file = os.path.join(self.output_dir, "trade_history.csv")
        self.is_model_loaded = True
        self.use_llm = use_llm
        self.llm = None
        
        # Create necessary directories
        for directory in [self.output_dir, self.models_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize trade history
        if not os.path.exists(self.history_file):
            pd.DataFrame(columns=[
                'timestamp', 'source', 'trend', 'support', 'resistance', 
                'recommended_trade', 'confidence', 'outcome', 'liquidity_zones'
            ]).to_csv(self.history_file, index=False)
            
        # Initialize lightweight LLM
        if self.use_llm:
            try:
                self.llm = pipeline('text-generation', 
                                   model='gpt2',  # Switch to 'microsoft/phi-1' for better results
                                   device=-1,     # CPU only
                                   max_length=200)
                print("Lightweight LLM loaded...")
            except Exception as e:
                print(f"Could not load LLM: {e}")
                self.use_llm = False
            
        print("NQ Trading Assistant initialized...")
    
    def extract_features_from_image(self, image_path):
        """Enhanced feature extraction with liquidity analysis"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            pixel_data = np.array(img)
            
            # Color analysis
            green_mask = (pixel_data[:,:,1] > pixel_data[:,:,0] + 30) & (pixel_data[:,:,1] > pixel_data[:,:,2] + 30)
            green_count = np.sum(green_mask)
            red_mask = (pixel_data[:,:,0] > pixel_data[:,:,1] + 30) & (pixel_data[:,:,0] > pixel_data[:,:,2] + 30)
            red_count = np.sum(red_mask)
            
            # Edge detection
            gray = np.array(ImageOps.grayscale(img))
            v_edges = np.abs(np.diff(gray, axis=0))
            h_edges = np.abs(np.diff(gray, axis=1))
            v_edges = np.pad(v_edges, ((0, 1), (0, 0)), mode='constant')
            h_edges = np.pad(h_edges, ((0, 0), (0, 1)), mode='constant')
            edges = np.maximum(v_edges, h_edges)
            
            # Liquidity analysis (upper/lower wick detection)
            region_height = int(height * 0.3)
            upper_region = edges[:region_height, :]
            lower_region = edges[-region_height:, :]
            
            upper_wick_density = np.sum(upper_region > 40) / (region_height * width)
            lower_wick_density = np.sum(lower_region > 40) / (region_height * width)
            
            # Horizontal line detection
            horizontal_line_score = 0
            for y in range(0, height, 5):
                row = h_edges[y, :]
                if np.sum(row > 40) > width * 0.5:
                    horizontal_line_score += 1
            
            return {
                "green_ratio": green_count / (green_count + red_count + 1),
                "red_ratio": red_count / (green_count + red_count + 1),
                "upper_wick_density": upper_wick_density,
                "lower_wick_density": lower_wick_density,
                "horizontal_lines": horizontal_line_score / (height / 5),
                "edge_density": np.sum(edges > 30) / (width * height),
                "image_size": (width, height)
            }
        except Exception as e:
            print(f"Image processing error: {e}")
            return {"error": str(e)}
    
    def analyze_liquidity(self, features):
        """Analyze liquidity conditions from features"""
        liquidity = {
            "upper_liquidity": False,
            "lower_liquidity": False,
            "confidence": 0.0
        }
        
        upper_wick = features.get("upper_wick_density", 0)
        lower_wick = features.get("lower_wick_density", 0)
        
        if upper_wick > 0.15:
            liquidity["upper_liquidity"] = True
            liquidity["confidence"] += min(upper_wick * 2, 0.6)
        if lower_wick > 0.15:
            liquidity["lower_liquidity"] = True
            liquidity["confidence"] += min(lower_wick * 2, 0.6)
        
        liquidity["confidence"] = min(liquidity["confidence"], 0.95)
        return liquidity
    
    def analyze_candlestick_patterns(self, features):
        """Enhanced pattern analysis with liquidity"""
        analysis = {
            "trend": "neutral",
            "confidence": 0.5,
            "liquidity_zones": [],
            "key_levels": []
        }
        
        # Trend analysis
        gr_ratio = features["green_ratio"] - features["red_ratio"]
        if abs(gr_ratio) > 0.3:
            analysis["trend"] = "bullish" if gr_ratio > 0 else "bearish"
            analysis["confidence"] = min(0.5 + abs(gr_ratio) * 0.8, 0.95)
        
        # Liquidity analysis
        liquidity = self.analyze_liquidity(features)
        if liquidity["upper_liquidity"]:
            analysis["liquidity_zones"].append("upper")
        if liquidity["lower_liquidity"]:
            analysis["liquidity_zones"].append("lower")
        
        # Key levels detection
        if features["horizontal_lines"] > 0.3:
            analysis["key_levels"] = ["support", "resistance"]
        
        return analysis
    
    def generate_llm_insight(self, analysis):
        """Enhance analysis with lightweight LLM"""
        if not self.use_llm or not self.llm:
            return ""
        
        prompt = f"""Analyze this trading situation:
- Trend: {analysis.get('trend', 'neutral')}
- Confidence: {analysis.get('confidence', 0.5)*100}%
- Liquidity Zones: {analysis.get('liquidity_zones', [])}
- Key Levels: {analysis.get('key_levels', [])}

Provide professional trading advice considering liquidity and price action:"""
        
        try:
            response = self.llm(prompt, max_length=200, num_return_sequences=1)
            return response[0]['generated_text'].split(":")[-1].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""
    
    def analyze_image(self, image_path):
        """Complete analysis pipeline with liquidity and LLM"""
        try:
            features = self.extract_features_from_image(image_path)
            if "error" in features:
                return f"Error: {features['error']}"
            
            pattern_analysis = self.analyze_candlestick_patterns(features)
            llm_insight = self.generate_llm_insight(pattern_analysis)
            
            analysis = self._generate_report(pattern_analysis, features)
            if llm_insight:
                analysis += f"\n\nLLM Insight:\n{llm_insight}"
            
            self._save_to_history(
                source=image_path,
                trend=pattern_analysis["trend"],
                liquidity_zones=",".join(pattern_analysis["liquidity_zones"]),
                confidence=pattern_analysis["confidence"]
            )
            
            return analysis
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _generate_report(self, analysis, features):
        """Generate detailed trading report"""
        report = f"""# NQ Trading Analysis Report
## Market Assessment
- **Trend**: {analysis['trend'].upper()} ({analysis['confidence']*100:.1f}% confidence)
- **Liquidity Zones**: {analysis['liquidity_zones'] or 'None detected'}
- **Key Levels**: {analysis['key_levels'] or 'None detected'}

## Key Observations"""
        
        if analysis['liquidity_zones']:
            report += "\n- Significant liquidity detected in "
            report += "upper zones" if 'upper' in analysis['liquidity_zones'] else ""
            report += " and " if len(analysis['liquidity_zones']) > 1 else ""
            report += "lower zones" if 'lower' in analysis['liquidity_zones'] else ""
            report += " (potential stop runs)"
            
        if features['green_ratio'] > 0.7:
            report += "\n- Strong bullish momentum with consecutive green candles"
        elif features['red_ratio'] > 0.7:
            report += "\n- Strong bearish pressure with consecutive red candles"
            
        report += "\n\n## Trading Plan\n"
        if analysis['trend'] == 'bullish':
            report += "Consider long positions with:\n"
            report += "- Entry: Pullbacks to support levels\n"
            report += "- Stop Loss: Below recent swing lows\n"
            if 'lower' in analysis['liquidity_zones']:
                report += "- Target: Previous highs above liquidity zones"
        elif analysis['trend'] == 'bearish':
            report += "Consider short positions with:\n"
            report += "- Entry: Rallies to resistance levels\n"
            report += "- Stop Loss: Above recent swing highs\n"
            if 'upper' in analysis['liquidity_zones']:
                report += "- Target: Previous lows below liquidity zones"
        else:
            report += "Market in consolidation. Wait for breakout:\n"
            report += "- Long above recent highs with volume\n"
            report += "- Short below recent lows with momentum"
            
        report += "\n\nRisk Management:\n- Use 1:2 risk-reward ratio minimum\n"
        report += "- Position size 1-2% of capital per trade"
        return report
    
    def _save_to_history(self, **kwargs):
        """Save analysis to history CSV"""
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': '',
            **kwargs
        }
        
        df = pd.read_csv(self.history_file) if os.path.exists(self.history_file) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.history_file, index=False)

# Remaining methods (analyze_numerical_data, plot_market_data, etc.) 
# would follow similar pattern with liquidity integration

def main():
    parser = argparse.ArgumentParser(description='NQ Trading Assistant')
    parser.add_argument('--image', help='Path to market chart image')
    parser.add_argument('--use_llm', action='store_true', help='Enable LLM insights')
    # Add other parameters...
    
    args = parser.parse_args()
    assistant = NQTradingAssistant(use_llm=args.use_llm)
    
    if args.image:
        analysis = assistant.analyze_image(args.image)
        print(analysis)
    
if __name__ == "__main__":
    main()