"""
Advanced text preprocessing utilities for sentiment analysis.
"""
import re
import string
import logging
from typing import List, Set, Dict, Optional
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor.
        
        Args:
            language: Language for stop words and processing
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load stop words
        try:
            self.stop_words = set(stopwords.words(language))
            # Add custom stop words for YouTube comments
            self.stop_words.update([
                'video', 'youtube', 'watch', 'like', 'subscribe', 'comment',
                'channel', 'thanks', 'please', 'first', 'edit', 'lol', 'omg'
            ])
        except Exception as e:
            logger.warning(f"Could not load stop words: {e}")
            self.stop_words = set()
        
        # Emoji mappings for sentiment
        self.emoji_sentiment = {
            # Positive emojis
            '😊': 'happy', '😍': 'love', '👍': 'good', '❤️': 'love', '🔥': 'awesome',
            '⭐': 'great', '🎉': 'celebrate', '👏': 'applause', '💯': 'perfect',
            '😄': 'happy', '😃': 'happy', '😀': 'happy', '🥰': 'love', '😘': 'love',
            '👌': 'good', '✨': 'amazing', '🙌': 'praise', '💖': 'love', '💕': 'love',
            
            # Negative emojis
            '👎': 'bad', '😞': 'sad', '😡': 'angry', '💔': 'heartbroken', '😢': 'cry',
            '🤮': 'disgusting', '😤': 'angry', '😠': 'mad', '💩': 'bad', '❌': 'no',
            '😭': 'cry', '😟': 'worried', '😔': 'sad', '🙄': 'annoyed', '😒': 'annoyed',
            '💀': 'dead', '🤬': 'angry', '😣': 'frustrated', '😖': 'confused'
        }
        
        # Slang and internet speak mappings
        self.slang_mappings = {
            'lol': 'laugh out loud', 'lmao': 'laugh out loud', 'rofl': 'laugh out loud',
            'omg': 'oh my god', 'wtf': 'what the hell', 'tbh': 'to be honest',
            'imo': 'in my opinion', 'imho': 'in my humble opinion', 'afaik': 'as far as i know',
            'u': 'you', 'ur': 'your', 'n': 'and', 'r': 'are', 'ppl': 'people',
            'thx': 'thanks', 'plz': 'please', 'gud': 'good', 'gr8': 'great',
            'luv': 'love', 'hav': 'have', 'cant': 'cannot', 'wont': 'will not',
            'dont': 'do not', 'isnt': 'is not', 'wasnt': 'was not', 'werent': 'were not'
        }
        
        # Initialize word frequency counter
        self.word_frequencies = Counter()
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        
        for dataset in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else 
                              f'corpora/{dataset}' if dataset in ['stopwords', 'wordnet', 'omw-1.4'] else 
                              f'taggers/{dataset}')
            except LookupError:
                try:
                    nltk.download(dataset, quiet=True)
                    logger.info(f"Downloaded NLTK dataset: {dataset}")
                except Exception as e:
                    logger.warning(f"Failed to download {dataset}: {e}")
    
    def preprocess(
        self,
        text: str,
        remove_stop_words: bool = True,
        lemmatize: bool = True,
        handle_emojis: bool = True,
        expand_contractions: bool = True,
        handle_slang: bool = True,
        min_word_length: int = 2
    ) -> Dict[str, any]:
        """
        Comprehensive text preprocessing.
        
        Args:
            text: Input text
            remove_stop_words: Remove stop words
            lemmatize: Apply lemmatization
            handle_emojis: Convert emojis to text
            expand_contractions: Expand contractions
            handle_slang: Handle internet slang
            min_word_length: Minimum word length to keep
            
        Returns:
            Dictionary with processed text and metadata        """
        if not text or not isinstance(text, str):
            return {
                'original': text,
                'processed': '',
                'processed_text': '',  # For backward compatibility
                'tokens': [],
                'word_count': 0,
                'emoji_count': 0,
                'sentiment_indicators': {}
            }
        
        original_text = text
        processed_text = text.lower()
        
        # Track emoji count before processing
        emoji_count = self._count_emojis(processed_text)
        
        # Handle emojis
        if handle_emojis:
            processed_text = self._handle_emojis(processed_text)
        
        # Expand contractions
        if expand_contractions:
            processed_text = self._expand_contractions(processed_text)
        
        # Handle slang
        if handle_slang:
            processed_text = self._handle_slang(processed_text)
        
        # Clean text
        processed_text = self._clean_text(processed_text)
        
        # Tokenize
        tokens = word_tokenize(processed_text)
        
        # Remove punctuation and filter by length
        tokens = [token for token in tokens 
                 if token not in string.punctuation and len(token) >= min_word_length]
        
        # Remove stop words
        if remove_stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = self._lemmatize_tokens(tokens)
        
        # Update word frequencies
        self.word_frequencies.update(tokens)
        
        # Extract sentiment indicators
        sentiment_indicators = self._extract_sentiment_indicators(original_text, tokens)
          # Join processed tokens
        final_text = ' '.join(tokens)
        return {
            'original': original_text,
            'original_text': original_text,  # For test compatibility
            'processed': final_text,
            'processed_text': final_text,  # For backward compatibility
            'tokens': tokens,
            'word_count': len(tokens),
            'emoji_count': emoji_count,
            'sentiment_indicators': sentiment_indicators
        }
    
    def _handle_emojis(self, text: str) -> str:
        """Convert emojis to text representations."""
        for emoji, sentiment_word in self.emoji_sentiment.items():
            text = text.replace(emoji, f' {sentiment_word} ')
        
        # Remove remaining emojis (those not in our mapping)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        return emoji_pattern.sub('', text)
    
    def _count_emojis(self, text: str) -> int:
        """Count emojis in text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        return len(emoji_pattern.findall(text))
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "let's": "let us", "that's": "that is",
            "what's": "what is", "here's": "here is", "there's": "there is",
            "where's": "where is", "how's": "how is", "why's": "why is",
            "who's": "who is", "it's": "it is", "he's": "he is",
            "she's": "she is", "we're": "we are", "they're": "they are"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _handle_slang(self, text: str) -> str:
        """Handle internet slang and abbreviations."""
        words = text.split()
        processed_words = []
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = word.strip(string.punctuation).lower()
            
            if clean_word in self.slang_mappings:
                processed_words.append(self.slang_mappings[clean_word])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def _clean_text(self, text: str) -> str:
        """Clean text from URLs, mentions, and excessive punctuation."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove @ mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove # hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # Remove repeated characters
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens with POS tagging."""
        try:
            # Get POS tags
            pos_tags = pos_tag(tokens)
            
            lemmatized = []
            for word, pos in pos_tags:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                
                if wordnet_pos:
                    lemmatized.append(self.lemmatizer.lemmatize(word, wordnet_pos))
                else:
                    lemmatized.append(self.lemmatizer.lemmatize(word))
            
            return lemmatized
            
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return tokens
    
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
    def _extract_sentiment_indicators(self, original_text: str, tokens: List[str]) -> Dict[str, any]:
        """Extract sentiment indicators from text."""
        positive_emojis = sum(1 for emoji in self.emoji_sentiment if emoji in original_text and 
                             self.emoji_sentiment[emoji] in ['happy', 'love', 'good', 'awesome', 'great', 'perfect'])
        negative_emojis = sum(1 for emoji in self.emoji_sentiment if emoji in original_text and 
                             self.emoji_sentiment[emoji] in ['bad', 'sad', 'angry', 'disgusting', 'mad'])
        
        indicators = {
            'exclamation_marks': original_text.count('!'),
            'question_marks': original_text.count('?'),
            'caps_ratio': sum(1 for c in original_text if c.isupper()) / len(original_text) if original_text else 0,
            'positive_emojis': positive_emojis,
            'negative_emojis': negative_emojis,
            'positive_count': positive_emojis,  # For test compatibility
            'negative_count': negative_emojis,  # For test compatibility
            'word_repetition': self._calculate_word_repetition(tokens),
            'average_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        }
        
        return indicators
    
    def _calculate_word_repetition(self, tokens: List[str]) -> float:
        """Calculate ratio of repeated words."""
        if not tokens:
            return 0.0
        
        word_counts = Counter(tokens)
        repeated_words = sum(count for count in word_counts.values() if count > 1)
        
        return repeated_words / len(tokens)
    
    def get_word_frequencies(self, top_n: int = 50) -> List[tuple]:
        """Get most common words."""
        return self.word_frequencies.most_common(top_n)
    
    def reset_frequencies(self) -> None:
        """Reset word frequency counter."""
        self.word_frequencies.clear()
    
    def batch_preprocess(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict[str, any]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            **kwargs: Arguments for preprocess method
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        for text in texts:
            result = self.preprocess(text, **kwargs)
            results.append(result)
        
        return results
