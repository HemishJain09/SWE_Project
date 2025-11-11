"""
Gemini Content Analyzer Module
Analyzes text or CSV content to determine the most appropriate BERT model for classification.
Uses Google's Gemini API for intelligent content analysis.
"""

import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Optional


class GeminiContentAnalyzer:
    """
    Analyzes content using Gemini API to recommend the best BERT model.
    
    Supported Models:
    - finbert: Financial text classification
    - berttweet: Twitter/social media text
    - patentbert: Patent documents
    - clinicalbert: Clinical/medical text
    - biobert: Biomedical/biological text
    - scibert: Scientific papers and research
    """
    
    # Model configurations with keywords and descriptions
    MODEL_INFO = {
        'finbert': {
            'name': 'FinBERT',
            'description': 'Specialized in financial text and documents',
            'keywords': ['financial', 'stock', 'earnings', 'revenue', 'market', 'investment', 
                        'trading', 'portfolio', 'economy', 'fiscal', 'monetary', 'securities',
                        'bonds', 'equity', 'profit', 'loss', 'dividend', 'assets', 'liabilities']
        },
        'berttweet': {
            'name': 'BERTweet',
            'description': 'Optimized for Twitter and social media content',
            'keywords': ['tweet', 'twitter', 'social media', 'hashtag', 'retweet', 'trending',
                        'viral', 'post', 'comment', 'like', 'share', 'follower', 'emoji',
                        'mention', '@', '#', 'dm', 'thread', 'influencer']
        },
        'patentbert': {
            'name': 'PatentBERT',
            'description': 'Designed for patent documents and legal text',
            'keywords': ['patent', 'invention', 'claim', 'intellectual property', 'trademark',
                        'copyright', 'licensing', 'prior art', 'USPTO', 'filing', 'granted',
                        'infringement', 'embodiment', 'apparatus', 'method', 'legal', 'litigation']
        },
        'clinicalbert': {
            'name': 'ClinicalBERT',
            'description': 'Specialized in medical and clinical records',
            'keywords': ['patient', 'diagnosis', 'treatment', 'clinical', 'medical', 'hospital',
                        'symptom', 'disease', 'prescription', 'doctor', 'nurse', 'therapy',
                        'medication', 'surgery', 'examination', 'health record', 'EHR', 'vital signs']
        },
        'biobert': {
            'name': 'BioBERT',
            'description': 'Expert in biomedical and biological research',
            'keywords': ['protein', 'gene', 'DNA', 'RNA', 'cell', 'molecular', 'biology',
                        'genetics', 'enzyme', 'chromosome', 'mutation', 'genomics', 'sequencing',
                        'organism', 'species', 'metabolism', 'biochemistry', 'biomarker']
        },
        'scibert': {
            'name': 'SciBERT',
            'description': 'Tailored for scientific papers and research',
            'keywords': ['research', 'study', 'experiment', 'hypothesis', 'methodology',
                        'results', 'conclusion', 'analysis', 'scientific', 'publication',
                        'journal', 'peer review', 'abstract', 'findings', 'data', 'statistical']
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Content Analyzer.
        
        Args:
            api_key: Optional Gemini API key. If not provided, loads from environment.
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def _construct_analysis_prompt(self, content: str, input_type: str) -> str:
        """
        Construct the prompt for Gemini to analyze content and select a model.
        
        Args:
            content: The text or CSV content to analyze
            input_type: Type of input ('text' or 'csv')
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert AI system specializing in text classification and domain analysis.
Your task is to analyze the provided content and determine which specialized BERT model would be most appropriate for classification tasks.

**Available Models:**
1. **FinBERT** - Financial text and documents (earnings reports, stock analysis, financial news)
2. **BERTweet** - Twitter and social media content (tweets, posts, social media discussions)
3. **PatentBERT** - Patent documents and legal text (patent claims, legal documents, IP text)
4. **ClinicalBERT** - Medical and clinical records (patient notes, diagnoses, medical reports)
5. **BioBERT** - Biomedical and biological research (genetics, molecular biology, proteins)
6. **SciBERT** - Scientific papers and research (academic papers, research findings, studies)

**Content Type:** {input_type.upper()}

**Content to Analyze:**
```
{content}  
```

**Instructions:**
1. Carefully analyze the content domain, terminology, and context
2. Identify the primary subject matter and field
3. Select the MOST appropriate model from the 6 options above
4. Provide a confidence score (0-100)
5. Explain your reasoning in 1-2 clear sentences

**Response Format (JSON only, no markdown):**
{{
    "model": "modelname",
    "confidence": 95,
    "reasoning": "Your clear explanation here"
}}

**Important:** 
- Return ONLY the JSON object, no additional text or markdown
- Model name must be lowercase: finbert, berttweet, patentbert, clinicalbert, biobert, or scibert
- Confidence must be a number between 0-100
- Reasoning should be concise and specific

Analyze the content now and provide your recommendation:
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse Gemini's response and extract model recommendation.
        
        Args:
            response_text: Raw response from Gemini API
            
        Returns:
            Dictionary with model, confidence, and reasoning
        """
        import json
        import re
        
        try:
            # Clean the response text
            clean_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON object in the text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
            
            # Parse JSON
            parsed = json.loads(clean_text)
            
            # Validate required fields
            if 'model' not in parsed or 'confidence' not in parsed or 'reasoning' not in parsed:
                raise ValueError("Missing required fields in response")
            
            # Normalize model name
            model = parsed['model'].lower().strip()
            
            # Validate model name
            valid_models = ['finbert', 'berttweet', 'patentbert', 'clinicalbert', 'biobert', 'scibert']
            if model not in valid_models:
                # Try to find closest match
                for valid_model in valid_models:
                    if valid_model in model:
                        model = valid_model
                        break
                else:
                    raise ValueError(f"Invalid model name: {model}")
            
            # Validate confidence
            confidence = int(parsed['confidence'])
            if confidence < 0 or confidence > 100:
                confidence = max(0, min(100, confidence))
            
            return {
                'model': model,
                'confidence': confidence,
                'reasoning': parsed['reasoning']
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {response_text}")
            
            # Fallback: Try keyword-based analysis
            return self._fallback_keyword_analysis(response_text)
    
    def _fallback_keyword_analysis(self, content: str) -> Dict[str, any]:
        """
        Fallback method using keyword matching if Gemini response parsing fails.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with model recommendation
        """
        content_lower = content.lower()
        scores = {}
        
        # Calculate scores for each model based on keyword matches
        for model_key, model_data in self.MODEL_INFO.items():
            score = sum(1 for keyword in model_data['keywords'] if keyword in content_lower)
            scores[model_key] = score
        
        # Select model with highest score
        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]
        
        # Calculate confidence based on score
        total_keywords = len(self.MODEL_INFO[best_model]['keywords'])
        confidence = min(95, int((max_score / total_keywords) * 100)) if max_score > 0 else 50
        
        return {
            'model': best_model,
            'confidence': confidence,
            'reasoning': f"Keyword analysis identified {max_score} relevant terms for {self.MODEL_INFO[best_model]['name']} domain."
        }
    
    def analyze_content(self, content: str, input_type: str = 'text', retries: int = 3) -> Dict[str, any]:
        """
        Analyze content and recommend the best BERT model.
        
        Args:
            content: Text or CSV content to analyze
            input_type: Type of input ('text' or 'csv')
            retries: Number of retry attempts for API calls
            
        Returns:
            Dictionary with 'model', 'confidence', and 'reasoning'
            
        Raises:
            Exception: If analysis fails after all retries
        """
        if not content or content.strip() == '':
            raise ValueError("Content cannot be empty")
        
        # Limit content based on input type to avoid quota limits
        if input_type == 'csv':
            # For CSV, limit to first 10 rows (including header)
            lines = content.split('\n')
            if len(lines) > 10:
                content = '\n'.join(lines[:10]) + "\n... (remaining rows truncated for analysis)"
        else:
            # For text, limit to first 200 words
            words = content.split()
            if len(words) > 200:
                content = ' '.join(words[:200]) + "\n... (content truncated for analysis)"
        
        # Also enforce character limit as backup
        if len(content) > 1500:
            content = content[:1500] + "\n... (content truncated)"
        
        prompt = self._construct_analysis_prompt(content, input_type)
        
        for attempt in range(retries):
            try:
                print(f"Analyzing content with Gemini API (attempt {attempt + 1}/{retries})...")
                response = self.model.generate_content(prompt)
                
                if response and response.text:
                    result = self._parse_gemini_response(response.text)
                    print(f"Analysis complete: {result['model']} (confidence: {result['confidence']}%)")
                    return result
                else:
                    print(f"Warning: Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Using fallback analysis.")
                    return self._fallback_keyword_analysis(content)
        
        # Final fallback
        return self._fallback_keyword_analysis(content)


def analyze_content_for_model(content: str, input_type: str = 'text') -> Dict[str, any]:
    """
    Convenience function to analyze content and get model recommendation.
    
    Args:
        content: Text or CSV content to analyze
        input_type: Type of input ('text' or 'csv')
        
    Returns:
        Dictionary with 'model', 'confidence', and 'reasoning'
    """
    analyzer = GeminiContentAnalyzer()
    return analyzer.analyze_content(content, input_type)
