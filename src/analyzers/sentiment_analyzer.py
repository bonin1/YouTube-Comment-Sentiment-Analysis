"""
Sentiment Analysis Module

This module provides comprehensive sentiment analysis using multiple ML models
including VADER, TextBlob, and transformer-based models with ensemble predictions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pickle
from pathlib import Path
import joblib

# NLP Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentAnalyzer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Deep Learning (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import get_logger, settings
from src.processors import TextPreprocessor


class SentimentAnalyzer:
    """Advanced sentiment analyzer with multiple models and ensemble predictions."""
    
    def __init__(self, use_transformers: bool = TRANSFORMERS_AVAILABLE):
        """Initialize the sentiment analyzer."""
        self.logger = get_logger(__name__)
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.text_processor = TextPreprocessor()
        
        # Initialize models
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.transformer_pipeline = None
        self.custom_model = None
        self.vectorizer = None
        
        # Model weights for ensemble
        self.model_weights = {
            'vader': 0.3,
            'textblob': 0.2,
            'transformer': 0.4,
            'custom': 0.1
        }
        
        # Sentiment thresholds
        self.sentiment_threshold = settings.SENTIMENT_THRESHOLD
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info("Sentiment analyzer initialized successfully")
    
    def _initialize_models(self):
        """Initialize all sentiment analysis models."""
        # Initialize transformer model if available
        if self.use_transformers:
            try:
                self.logger.info("Loading transformer model...")
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                self.logger.info("Transformer model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load transformer model: {str(e)}")
                self.use_transformers = False
        
        # Initialize custom model if exists
        self._load_custom_model()
    
    def _load_custom_model(self):
        """Load custom trained model if available."""
        model_path = settings.DATA_DIR / "models" / "custom_sentiment_model.pkl"
        vectorizer_path = settings.DATA_DIR / "models" / "sentiment_vectorizer.pkl"
        
        try:
            if model_path.exists() and vectorizer_path.exists():
                self.custom_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.logger.info("Custom sentiment model loaded successfully")
            else:
                self.logger.info("No custom model found, will use pre-trained models only")
        except Exception as e:
            self.logger.warning(f"Failed to load custom model: {str(e)}")
    
    def analyze_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment
            compound = scores['compound']
            if compound >= self.sentiment_threshold:
                sentiment = 'positive'
            elif compound <= -self.sentiment_threshold:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(compound),
                'scores': scores,
                'model': 'vader'
            }
        except Exception as e:
            self.logger.error(f"VADER analysis failed: {str(e)}")
            return self._get_default_result('vader')
    
    def analyze_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment
            if polarity > self.sentiment_threshold:
                sentiment = 'positive'
            elif polarity < -self.sentiment_threshold:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(polarity),
                'scores': {
                    'polarity': polarity,
                    'subjectivity': subjectivity
                },
                'model': 'textblob'
            }
        except Exception as e:
            self.logger.error(f"TextBlob analysis failed: {str(e)}")
            return self._get_default_result('textblob')
    
    def analyze_transformer(self, text: str) -> Dict:
        """Analyze sentiment using transformer model."""
        if not self.use_transformers or not self.transformer_pipeline:
            return self._get_default_result('transformer')
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.transformer_pipeline(text)
            
            # Process results
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                if 'positive' in label or 'pos' in label:
                    sentiment_scores['positive'] = result['score']
                elif 'negative' in label or 'neg' in label:
                    sentiment_scores['negative'] = result['score']
                else:
                    sentiment_scores['neutral'] = result['score']
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[dominant_sentiment]
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': confidence,
                'scores': sentiment_scores,
                'model': 'transformer'
            }
        except Exception as e:
            self.logger.error(f"Transformer analysis failed: {str(e)}")
            return self._get_default_result('transformer')
    
    def analyze_custom(self, text: str) -> Dict:
        """Analyze sentiment using custom trained model."""
        if not self.custom_model or not self.vectorizer:
            return self._get_default_result('custom')
        
        try:
            # Preprocess text
            processed = self.text_processor.preprocess_text(text)
            processed_text = processed['processed_text']
            
            # Vectorize
            features = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.custom_model.predict(features)[0]
            probabilities = self.custom_model.predict_proba(features)[0]
            
            # Get class names
            classes = self.custom_model.classes_
            confidence = max(probabilities)
            
            # Create scores dictionary
            scores = {classes[i]: probabilities[i] for i in range(len(classes))}
            
            return {
                'sentiment': prediction,
                'confidence': confidence,
                'scores': scores,
                'model': 'custom'
            }
        except Exception as e:
            self.logger.error(f"Custom model analysis failed: {str(e)}")
            return self._get_default_result('custom')
    
    def _get_default_result(self, model_name: str) -> Dict:
        """Get default result for failed analysis."""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'scores': {},
            'model': model_name,
            'error': True
        }
    
    def analyze_ensemble(self, text: str) -> Dict:
        """Perform ensemble sentiment analysis using multiple models."""
        if not text or not text.strip():
            return self._get_default_result('ensemble')
        
        # Get predictions from all models
        results = {
            'vader': self.analyze_vader(text),
            'textblob': self.analyze_textblob(text),
            'transformer': self.analyze_transformer(text),
            'custom': self.analyze_custom(text)
        }
        
        # Calculate weighted ensemble
        sentiment_votes = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        
        for model_name, result in results.items():
            if not result.get('error', False):
                weight = self.model_weights.get(model_name, 0)
                confidence = result['confidence']
                sentiment = result['sentiment']
                
                # Weight by both model weight and confidence
                weighted_vote = weight * confidence
                sentiment_votes[sentiment] += weighted_vote
                total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            for sentiment in sentiment_votes:
                sentiment_votes[sentiment] /= total_weight
        
        # Determine final sentiment
        final_sentiment = max(sentiment_votes, key=sentiment_votes.get)
        final_confidence = sentiment_votes[final_sentiment]
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(results)
        
        return {
            'sentiment': final_sentiment,
            'confidence': final_confidence,
            'agreement_score': agreement_score,
            'sentiment_scores': sentiment_votes,
            'individual_results': results,
            'model': 'ensemble',
            'metadata': {
                'text_length': len(text),
                'timestamp': datetime.now().isoformat(),
                'models_used': list(results.keys()),
                'high_confidence': final_confidence >= self.confidence_threshold
            }
        }
    
    def _calculate_agreement(self, results: Dict) -> float:
        """Calculate agreement score between models."""
        sentiments = []
        confidences = []
        
        for result in results.values():
            if not result.get('error', False):
                sentiments.append(result['sentiment'])
                confidences.append(result['confidence'])
        
        if len(sentiments) < 2:
            return 0.0
        
        # Calculate percentage of models that agree with majority
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        majority_sentiment = sentiment_counts.most_common(1)[0][0]
        agreement_count = sentiment_counts[majority_sentiment]
        
        agreement_ratio = agreement_count / len(sentiments)
        
        # Weight by average confidence
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return agreement_ratio * avg_confidence
    
    def analyze_batch(self, texts: List[str], use_ensemble: bool = True) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        for i, text in enumerate(texts):
            try:
                if use_ensemble:
                    result = self.analyze_ensemble(text)
                else:
                    result = self.analyze_vader(text)  # Fallback to VADER for speed
                
                result['batch_index'] = i
                results.append(result)
                
                # Log progress
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Analyzed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing text {i}: {str(e)}")
                results.append({
                    **self._get_default_result('batch'),
                    'batch_index': i,
                    'error_message': str(e)
                })
        
        return results
    
    def get_sentiment_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics from sentiment analysis results."""
        if not results:
            return {}
        
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r]
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        agreements = [r.get('agreement_score', 0) for r in results]
        
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        
        stats = {
            'total_comments': len(results),
            'sentiment_distribution': dict(sentiment_counts),
            'sentiment_percentages': {
                sentiment: (count / len(sentiments)) * 100
                for sentiment, count in sentiment_counts.items()
            } if sentiments else {},
            'average_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'average_agreement': np.mean(agreements) if agreements else 0,
            'high_confidence_count': sum(1 for c in confidences if c >= self.confidence_threshold),
            'high_confidence_percentage': (
                sum(1 for c in confidences if c >= self.confidence_threshold) / len(confidences) * 100
            ) if confidences else 0
        }
        
        return stats
    
    def train_custom_model(self, 
                          texts: List[str], 
                          labels: List[str],
                          model_type: str = 'logistic',
                          test_size: float = 0.2) -> Dict:
        """Train a custom sentiment analysis model."""
        self.logger.info(f"Training custom model with {len(texts)} samples")
        
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                processed = self.text_processor.preprocess_text(text)
                processed_texts.append(processed['processed_text'])
            
            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(processed_texts)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Choose model
            if model_type == 'logistic':
                model = LogisticRegression(random_state=42)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'naive_bayes':
                model = MultinomialNB()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model
            self.custom_model = model
            models_dir = settings.DATA_DIR / "models"
            models_dir.mkdir(exist_ok=True)
            
            joblib.dump(model, models_dir / "custom_sentiment_model.pkl")
            joblib.dump(self.vectorizer, models_dir / "sentiment_vectorizer.pkl")
            
            self.logger.info(f"Custom model trained successfully. Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'model_type': model_type,
                'training_samples': len(texts),
                'test_samples': len(y_test)
            }
            
        except Exception as e:
            self.logger.error(f"Error training custom model: {str(e)}")
            raise
