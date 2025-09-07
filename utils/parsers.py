import re
from typing import List, Dict, Any

def extract_json_from_output(output: str) -> Dict[str, Any]:
    """Extract JSON data from code output"""
    try:
        # Look for JSON-like patterns
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, output)
        
        if matches:
            import json
            return json.loads(matches[-1])  # Return last JSON found
        
        return {}
    except:
        return {}

def parse_data_quality_score(output: str) -> float:
    """Parse data quality score from output"""
    try:
        # Look for quality score patterns
        patterns = [
            r'quality.*?score.*?:?\s*([0-9.]+)',
            r'score.*?:?\s*([0-9.]+)',
            r'quality.*?:?\s*([0-9.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 if it looks like a percentage
                if score > 1:
                    score = score / 100
                return min(1.0, max(0.0, score))
        
        # Default score based on content
        output_lower = output.lower()
        if 'excellent' in output_lower or 'high quality' in output_lower:
            return 0.9
        elif 'good' in output_lower:
            return 0.8
        elif 'fair' in output_lower or 'moderate' in output_lower:
            return 0.7
        elif 'poor' in output_lower or 'low' in output_lower:
            return 0.5
        else:
            return 0.75  # Default reasonable score
            
    except:
        return 0.75