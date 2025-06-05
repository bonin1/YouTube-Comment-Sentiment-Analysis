"""
Text Preprocessing Module

This module provides comprehensive text preprocessing capabilities including
tokenization, lemmatization, stop word removal, and emoji handling.
"""

import re
import string
from typing import List, Dict, Set, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import emoji
import spacy
from textblob import TextBlob

from config import get_logger, settings


class TextPreprocessor:
    """Advanced text preprocessing with multiple options and configurations."""
    
    def __init__(self, language: str = 'en'):
        """Initialize the text preprocessor."""
        self.logger = get_logger(__name__)
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.spacy_nlp = None
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load stop words
        self.stop_words = self._load_stop_words()
        
        # Initialize spaCy if available
        self._initialize_spacy()
        
        # Emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+"
        )
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'vader_lexicon', 'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                self.logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words for the specified language."""
        try:
            stop_words = set(stopwords.words('english'))
            
            # Add custom stop words
            custom_stop_words = {
                'youtube', 'video', 'watch', 'like', 'subscribe', 'comment',
                'please', 'thanks', 'thank', 'lol', 'haha', 'omg', 'wow',
                'first', 'early', 'notification', 'squad', 'anyone'
            }
            
            stop_words.update(custom_stop_words)
            return stop_words
            
        except Exception as e:
            self.logger.warning(f"Error loading stop words: {str(e)}")
            return set()
    
    def _initialize_spacy(self):
        """Initialize spaCy NLP pipeline."""
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            self.spacy_nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers (optional - based on settings)
        text = re.sub(r'\d+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def handle_emojis(self, text: str, mode: str = 'convert') -> str:
        """
        Handle emojis in text.
        
        Args:
            text: Input text
            mode: 'convert' to text, 'remove', or 'keep'
            
        Returns:
            Processed text
        """
        if not settings.HANDLE_EMOJIS:
            return text
        
        if mode == 'convert':
            # Convert emojis to text descriptions
            return emoji.demojize(text, delimiters=(" ", " "))
        elif mode == 'remove':
            # Remove all emojis
            return self.emoji_pattern.sub('', text)
        else:
            # Keep emojis as is
            return text
    
    def tokenize(self, text: str, method: str = 'nltk') -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            method: 'nltk', 'spacy', or 'simple'
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if method == 'spacy' and self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc if not token.is_space]
        elif method == 'nltk':
            return word_tokenize(text)
        else:
            # Simple tokenization
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not settings.REMOVE_STOPWORDS:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str], method: str = 'nltk') -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            method: 'nltk' or 'spacy'
            
        Returns:
            Lemmatized tokens
        """
        if not settings.ENABLE_LEMMATIZATION:
            return tokens
        
        if method == 'spacy' and self.spacy_nlp:
            doc = self.spacy_nlp(' '.join(tokens))
            return [token.lemma_ for token in doc if not token.is_space]
        else:
            # NLTK lemmatization with POS tagging
            pos_tags = pos_tag(tokens)
            lemmatized = []
            
            for word, pos in pos_tags:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                if wordnet_pos:
                    lemmatized.append(self.lemmatizer.lemmatize(word, pos=wordnet_pos))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(word))
            
            return lemmatized
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert TreeBank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return None
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract various text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        }
        
        # Add emoji count
        features['emoji_count'] = len(self.emoji_pattern.findall(text))
        
        # Add readability score using TextBlob
        try:
            blob = TextBlob(text)
            features['polarity'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['polarity'] = 0.0
            features['subjectivity'] = 0.0
        
        return features
    
    def preprocess_text(self, 
                       text: str, 
                       clean: bool = True,
                       handle_emojis: bool = True,
                       tokenize: bool = True,
                       remove_stopwords: bool = None,
                       lemmatize: bool = None) -> Dict:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text
            clean: Apply basic cleaning
            handle_emojis: Process emojis
            tokenize: Tokenize text
            remove_stopwords: Remove stop words (uses settings default if None)
            lemmatize: Apply lemmatization (uses settings default if None)
            
        Returns:
            Dictionary with processed text and metadata
        """
        if remove_stopwords is None:
            remove_stopwords = settings.REMOVE_STOPWORDS
        if lemmatize is None:
            lemmatize = settings.ENABLE_LEMMATIZATION
        
        original_text = text
        processed_text = text
        
        # Extract original features
        original_features = self.extract_features(original_text)
        
        # Clean text
        if clean:
            processed_text = self.clean_text(processed_text)
        
        # Handle emojis
        if handle_emojis:
            processed_text = self.handle_emojis(processed_text, mode='convert')
        
        # Tokenization
        tokens = []
        if tokenize:
            tokens = self.tokenize(processed_text)
            
            # Remove stop words
            if remove_stopwords:
                tokens = self.remove_stopwords(tokens)
            
            # Lemmatization
            if lemmatize:
                tokens = self.lemmatize(tokens)
            
            # Rejoin tokens
            processed_text = ' '.join(tokens)
        
        # Extract processed features
        processed_features = self.extract_features(processed_text)
        
        return {
            'original_text': original_text,
            'processed_text': processed_text,
            'tokens': tokens,
            'original_features': original_features,
            'processed_features': processed_features,
            'processing_metadata': {
                'cleaned': clean,
                'emojis_handled': handle_emojis,
                'tokenized': tokenize,
                'stopwords_removed': remove_stopwords,
                'lemmatized': lemmatize
            }
        }
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to preprocess_text
            
        Returns:
            List of preprocessing results
        """
        results = []
        for text in texts:
            try:
                result = self.preprocess_text(text, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error preprocessing text: {str(e)}")
                # Return basic result for failed preprocessing
                results.append({
                    'original_text': text,
                    'processed_text': text,
                    'tokens': [],
                    'original_features': {},
                    'processed_features': {},
                    'processing_metadata': {'error': str(e)}
                })
        
        return results
