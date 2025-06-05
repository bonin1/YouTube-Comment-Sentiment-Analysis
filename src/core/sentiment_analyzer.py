import re
import string
import logging
from typing import List, Dict, Any, Optional
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataclasses import dataclass
import numpy as np

from ..config.settings import get_settings
from ..utils.decorators import timing, cache_result

@dataclass
class SentimentResult:
    """Data class for sentiment analysis result"""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # detailed scores for each sentiment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'scores': self.scores
        }

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Using basic preprocessing.")
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 
            'vader_lexicon', 'averaged_perceptron_tagger'
        ]
        
        for data in required_data:
            try:
                nltk.download(data, quiet=True)
            except Exception as e:
                self.logger.warning(f"Failed to download NLTK data '{data}': {e}")
    
    @timing
    def preprocess_text(self, text: str, advanced: bool = True) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            advanced: Whether to use advanced preprocessing with spaCy
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = self._basic_clean(text)
        
        if advanced and self.nlp:
            return self._advanced_preprocess(text)
        else:
            return self._basic_preprocess(text)
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions and #hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short text
        if len(text.split()) < 2:
            return ""
        
        return text
    
    def _basic_preprocess(self, text: str) -> str:
        """Basic preprocessing using NLTK"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _advanced_preprocess(self, text: str) -> str:
        """Advanced preprocessing using spaCy"""
        doc = self.nlp(text)
        
        # Extract lemmatized tokens, excluding stop words and punctuation
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.is_space
            and token.is_alpha
            and len(token.text) > 1
        ]
        
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional features from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'has_emoji': bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        }
        
        return features

class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize sentiment analysis models"""
        # VADER (rule-based)
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            self.logger.error(f"Failed to initialize VADER: {e}")
            self.vader = None
        
        # Transformer model (neural)
        try:
            model_name = self.settings.MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize transformer model: {e}")
            self.transformer_pipeline = None
    
    @timing
    def analyze_sentiment(self, text: str, method: str = "ensemble") -> SentimentResult:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            method: Analysis method ('vader', 'transformer', 'ensemble')
            
        Returns:
            SentimentResult object
        """
        if not text or not isinstance(text, str):
            return SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            )
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        if method == "vader":
            return self._analyze_vader(text, processed_text)
        elif method == "transformer":
            return self._analyze_transformer(text, processed_text)
        else:  # ensemble
            return self._analyze_ensemble(text, processed_text)
    
    def _analyze_vader(self, original_text: str, processed_text: str) -> SentimentResult:
        """Analyze sentiment using VADER"""
        if not self.vader:
            raise RuntimeError("VADER model not available")
        
        scores = self.vader.polarity_scores(original_text)
        
        # Determine sentiment
        if scores['compound'] >= 0.05:
            sentiment = "positive"
            confidence = scores['pos']
        elif scores['compound'] <= -0.05:
            sentiment = "negative"
            confidence = scores['neg']
        else:
            sentiment = "neutral"
            confidence = scores['neu']
        
        return SentimentResult(
            text=original_text,
            sentiment=sentiment,
            confidence=confidence,
            scores={
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu']
            }
        )
    
    def _analyze_transformer(self, original_text: str, processed_text: str) -> SentimentResult:
        """Analyze sentiment using transformer model"""
        if not self.transformer_pipeline:
            raise RuntimeError("Transformer model not available")
        
        # Use original text for transformer (it handles preprocessing)
        results = self.transformer_pipeline(original_text)[0]
        
        # Convert to our format
        scores_dict = {result['label'].lower(): result['score'] for result in results}
        
        # Normalize labels
        label_mapping = {
            'label_0': 'negative',
            'label_1': 'neutral', 
            'label_2': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive'
        }
        
        normalized_scores = {}
        for label, score in scores_dict.items():
            normalized_label = label_mapping.get(label, label)
            normalized_scores[normalized_label] = score
        
        # Find dominant sentiment
        sentiment = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[sentiment]
        
        return SentimentResult(
            text=original_text,
            sentiment=sentiment,
            confidence=confidence,
            scores=normalized_scores
        )
    
    def _analyze_ensemble(self, original_text: str, processed_text: str) -> SentimentResult:
        """Analyze sentiment using ensemble of methods"""
        results = []
        
        # Get VADER results
        if self.vader:
            try:
                vader_result = self._analyze_vader(original_text, processed_text)
                results.append(vader_result)
            except Exception as e:
                self.logger.warning(f"VADER analysis failed: {e}")
        
        # Get transformer results
        if self.transformer_pipeline:
            try:
                transformer_result = self._analyze_transformer(original_text, processed_text)
                results.append(transformer_result)
            except Exception as e:
                self.logger.warning(f"Transformer analysis failed: {e}")
        
        if not results:
            # Fallback to neutral
            return SentimentResult(
                text=original_text,
                sentiment="neutral",
                confidence=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            )
        
        # Ensemble voting
        return self._ensemble_vote(original_text, results)
    
    def _ensemble_vote(self, original_text: str, results: List[SentimentResult]) -> SentimentResult:
        """Combine multiple sentiment analysis results"""
        if len(results) == 1:
            return results[0]
        
        # Weighted average of scores
        combined_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_weight = 0.0
        
        for result in results:
            weight = result.confidence
            total_weight += weight
            
            for sentiment, score in result.scores.items():
                if sentiment in combined_scores:
                    combined_scores[sentiment] += score * weight
        
        # Normalize scores
        if total_weight > 0:
            for sentiment in combined_scores:
                combined_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        final_sentiment = max(combined_scores, key=combined_scores.get)
        final_confidence = combined_scores[final_sentiment]
        
        return SentimentResult(
            text=original_text,
            sentiment=final_sentiment,
            confidence=final_confidence,
            scores=combined_scores
        )
    
    @timing
    def analyze_batch(self, texts: List[str], method: str = "ensemble") -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            method: Analysis method
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text, method)
                results.append(result)
                
                # Log progress for large batches
                if len(texts) > 100 and (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze text {i}: {e}")
                # Add neutral result for failed analysis
                results.append(SentimentResult(
                    text=text,
                    sentiment="neutral",
                    confidence=0.0,
                    scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                ))
        
        return results
