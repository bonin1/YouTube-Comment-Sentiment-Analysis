"""
Sophisticated rule-based sentiment classification system.
"""
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """Result of sentiment classification."""
    label: SentimentLabel
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    matched_rules: List[str]
    sentiment_words: List[str]

class SentimentRuleEngine:
    """Advanced rule-based sentiment classification engine."""
    
    def __init__(self):
        """Initialize the sentiment rule engine."""
        self._load_sentiment_lexicons()
        self._load_pattern_rules()
        self._load_emoji_rules()
        self._load_contextual_rules()
    
    def _load_sentiment_lexicons(self) -> None:
        """Load sentiment word lexicons."""
        # Positive words with weights
        self.positive_words = {
            # Strong positive (weight: 3)
            'amazing': 3, 'awesome': 3, 'excellent': 3, 'fantastic': 3,
            'incredible': 3, 'outstanding': 3, 'perfect': 3, 'brilliant': 3,
            'magnificent': 3, 'superb': 3, 'extraordinary': 3, 'phenomenal': 3,
            
            # Moderate positive (weight: 2)
            'good': 2, 'great': 2, 'nice': 2, 'wonderful': 2, 'beautiful': 2,
            'lovely': 2, 'pleasant': 2, 'enjoyable': 2, 'impressive': 2,
            'solid': 2, 'cool': 2, 'sweet': 2, 'neat': 2, 'fun': 2,
            
            # Mild positive (weight: 1)
            'ok': 1, 'okay': 1, 'fine': 1, 'decent': 1, 'alright': 1,
            'satisfactory': 1, 'acceptable': 1, 'adequate': 1,
            
            # Emotional positive
            'love': 3, 'adore': 3, 'cherish': 2, 'appreciate': 2, 'like': 2,
            'enjoy': 2, 'happy': 2, 'excited': 2, 'thrilled': 3, 'delighted': 3,
            'pleased': 2, 'glad': 2, 'satisfied': 2, 'content': 1,
            
            # Quality positive
            'best': 3, 'top': 2, 'first': 2, 'premium': 2, 'superior': 2,
            'quality': 2, 'professional': 2, 'expert': 2, 'skilled': 2,
            'talented': 2, 'gifted': 2, 'genius': 3,
            
            # Action positive
            'recommend': 2, 'subscribe': 1, 'share': 1, 'support': 2,
            'helpful': 2, 'useful': 2, 'valuable': 2, 'worth': 2,
            'inspiring': 2, 'motivating': 2, 'encouraging': 2
        }
        
        # Negative words with weights
        self.negative_words = {
            # Strong negative (weight: 3)
            'terrible': 3, 'awful': 3, 'horrible': 3, 'disgusting': 3,
            'pathetic': 3, 'ridiculous': 3, 'stupid': 3, 'idiotic': 3,
            'worst': 3, 'trash': 3, 'garbage': 3, 'crap': 3, 'shit': 3,
            'fucking': 3, 'damn': 3, 'hell': 3, 'disaster': 3,
            
            # Moderate negative (weight: 2)
            'bad': 2, 'poor': 2, 'weak': 2, 'disappointing': 2, 'boring': 2,
            'dull': 2, 'lame': 2, 'sucks': 2, 'annoying': 2, 'irritating': 2,
            'frustrating': 2, 'confusing': 2, 'complicated': 2, 'difficult': 2,
            'hard': 2, 'slow': 2, 'broken': 2, 'failed': 2, 'wrong': 2,
            
            # Mild negative (weight: 1)
            'meh': 1, 'blah': 1, 'whatever': 1, 'nope': 1, 'no': 1,
            'not': 1, 'never': 1, 'nothing': 1, 'none': 1,
            
            # Emotional negative
            'hate': 3, 'despise': 3, 'detest': 3, 'loathe': 3, 'dislike': 2,
            'angry': 2, 'mad': 2, 'furious': 3, 'pissed': 3, 'upset': 2,
            'sad': 2, 'depressed': 2, 'disappointed': 2, 'frustrated': 2,
            'worried': 2, 'scared': 2, 'afraid': 2, 'nervous': 2,
            
            # Quality negative
            'fake': 2, 'cheap': 2, 'low': 2, 'inferior': 2, 'amateur': 2,
            'unprofessional': 2, 'lazy': 2, 'careless': 2, 'sloppy': 2,
            'messy': 2, 'ugly': 2, 'outdated': 2, 'old': 1,
            
            # Action negative
            'quit': 2, 'stop': 2, 'leave': 2, 'unsubscribe': 2, 'dislike': 2,
            'report': 2, 'complain': 2, 'criticize': 2, 'bash': 2, 'attack': 2
        }
        
        # Neutral words (contextual)
        self.neutral_words = {
            'video': 0, 'channel': 0, 'content': 0, 'watch': 0, 'see': 0,
            'think': 0, 'believe': 0, 'consider': 0, 'maybe': 0, 'perhaps': 0,
            'might': 0, 'could': 0, 'would': 0, 'should': 0, 'will': 0,
            'time': 0, 'year': 0, 'day': 0, 'week': 0, 'month': 0,
            'people': 0, 'person': 0, 'guy': 0, 'girl': 0, 'man': 0, 'woman': 0
        }
    
    def _load_pattern_rules(self) -> None:
        """Load pattern-based rules."""
        self.positive_patterns = [
            (r'\b(so|very|really|extremely|incredibly|absolutely)\s+(good|great|amazing|awesome|nice|cool|beautiful|wonderful|perfect|excellent)\b', 2),
            (r'\b(love|adore|absolutely\s+love)\s+(this|it|that)\b', 3),
            (r'\b(best|greatest|finest|most\s+amazing|most\s+awesome)\b', 3),
            (r'\b(definitely|absolutely|totally|completely)\s+(recommend|worth|good|great)\b', 2),
            (r'\b(thank\s+you|thanks|appreciate|grateful)\b', 2),
            (r'\b(keep\s+up|good\s+work|well\s+done|great\s+job)\b', 2),
            (r'\b(looking\s+forward|can\'t\s+wait|excited\s+for)\b', 2),
            (r'!{2,}', 1),  # Multiple exclamation marks
            (r'\b(yes|yeah|yep|yay|woohoo|woot)\b', 1)
        ]
        
        self.negative_patterns = [
            (r'\b(so|very|really|extremely|incredibly|absolutely)\s+(bad|terrible|awful|horrible|boring|stupid|annoying)\b', 2),
            (r'\b(hate|despise|can\'t\s+stand)\s+(this|it|that)\b', 3),
            (r'\b(worst|terrible|awful|horrible|disgusting|pathetic)\b', 3),
            (r'\b(total|complete|absolute)\s+(disaster|failure|waste|crap|shit|garbage)\b', 3),
            (r'\b(what\s+the\s+hell|what\s+the\s+fuck|wtf|omg\s+no)\b', 2),
            (r'\b(don\'t|do\s+not)\s+(like|want|need|recommend)\b', 2),
            (r'\b(never\s+again|waste\s+of\s+time|money\s+back)\b', 2),
            (r'\?\?+', 1),  # Multiple question marks (confusion/frustration)
            (r'\b(ugh|eww|gross|yuck|meh|blah)\b', 1)
        ]
        
        self.neutral_patterns = [
            (r'\b(i\s+think|in\s+my\s+opinion|imo|imho|personally)\b', 0),
            (r'\b(maybe|perhaps|might|could|possibly|probably)\b', 0),
            (r'\b(question|ask|wonder|curious|how|when|where|why|what)\b', 0),
            (r'\b(information|facts|data|statistics|analysis|review)\b', 0),
            (r'\b(first|second|next|then|finally|conclusion)\b', 0)
        ]
    
    def _load_emoji_rules(self) -> None:
        """Load emoji-based sentiment rules."""
        self.positive_emojis = {
            '😊': 2, '😍': 3, '👍': 2, '❤️': 3, '🔥': 2, '⭐': 2, '🎉': 2,
            '👏': 2, '💯': 3, '😄': 2, '😃': 2, '😀': 2, '🥰': 3, '😘': 2,
            '👌': 2, '✨': 2, '🙌': 2, '💖': 3, '💕': 2, '🤩': 2, '😎': 2,
            '🎊': 2, '🎈': 1, '🌟': 2, '💪': 2, '🏆': 2, '🥇': 2, '🎯': 2
        }
        
        self.negative_emojis = {
            '👎': 2, '😞': 2, '😡': 3, '💔': 3, '😢': 2, '🤮': 3, '😤': 2,
            '😠': 3, '💩': 3, '❌': 2, '😭': 2, '😟': 2, '😔': 2, '🙄': 2,
            '😒': 2, '💀': 2, '🤬': 3, '😣': 2, '😖': 2, '😫': 2, '😩': 2,
            '🤦': 2, '🤷': 1, '😴': 1, '🥱': 1, '😵': 2, '🤯': 2
        }
    
    def _load_contextual_rules(self) -> None:
        """Load contextual sentiment rules."""
        # Negation words that can flip sentiment
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
            'nor', 'none', 'without', 'lack', 'lacking', 'missing', 'absent',
            'avoid', 'prevent', 'stop', 'quit', 'cease', 'end', 'finish'
        }
        
        # Intensifiers that amplify sentiment
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
            'absolutely': 2.0, 'totally': 1.8, 'completely': 1.8, 'so': 1.3,
            'quite': 1.2, 'rather': 1.2, 'pretty': 1.2, 'fairly': 1.1,
            'somewhat': 0.8, 'slightly': 0.7, 'a bit': 0.7, 'kind of': 0.8,
            'sort of': 0.8, 'too': 1.3, 'way': 1.5, 'super': 1.8
        }
        
        # Diminishers that reduce sentiment
        self.diminishers = {
            'barely': 0.5, 'hardly': 0.5, 'scarcely': 0.5, 'rarely': 0.6,
            'seldom': 0.6, 'little': 0.7, 'somewhat': 0.8, 'slightly': 0.7,
            'a little': 0.7, 'a bit': 0.7, 'kind of': 0.8, 'sort of': 0.8
        }
    
    def classify_sentiment(
        self,
        text: str,
        processed_tokens: List[str],
        sentiment_indicators: Dict[str, any]
    ) -> SentimentResult:
        """
        Classify sentiment using rule-based approach.
        
        Args:
            text: Original text
            processed_tokens: Preprocessed tokens
            sentiment_indicators: Additional sentiment indicators
            
        Returns:
            SentimentResult with classification details
        """
        # Calculate individual scores
        word_scores = self._calculate_word_scores(processed_tokens)
        pattern_scores = self._calculate_pattern_scores(text)
        emoji_scores = self._calculate_emoji_scores(text)
        contextual_scores = self._calculate_contextual_scores(text, processed_tokens)
        indicator_scores = self._calculate_indicator_scores(sentiment_indicators)
        
        # Combine scores
        positive_score = (
            word_scores['positive'] +
            pattern_scores['positive'] +
            emoji_scores['positive'] +
            contextual_scores['positive'] +
            indicator_scores['positive']
        )
        
        negative_score = (
            word_scores['negative'] +
            pattern_scores['negative'] +
            emoji_scores['negative'] +
            contextual_scores['negative'] +
            indicator_scores['negative']
        )
        
        neutral_score = (
            word_scores['neutral'] +
            pattern_scores['neutral'] +
            indicator_scores['neutral']
        )
        
        # Apply contextual adjustments
        positive_score, negative_score = self._apply_contextual_adjustments(
            text, processed_tokens, positive_score, negative_score
        )
        
        # Normalize scores
        total_score = positive_score + negative_score + neutral_score
        if total_score > 0:
            positive_score /= total_score
            negative_score /= total_score
            neutral_score /= total_score
        else:
            # Default to neutral if no sentiment detected
            positive_score, negative_score, neutral_score = 0.0, 0.0, 1.0
        
        # Determine final label and confidence
        scores = {
            SentimentLabel.POSITIVE: positive_score,
            SentimentLabel.NEGATIVE: negative_score,
            SentimentLabel.NEUTRAL: neutral_score
        }
        
        final_label = max(scores, key=scores.get)
        confidence = scores[final_label]
        
        # Collect matched rules and sentiment words
        matched_rules = self._get_matched_rules(text, processed_tokens)
        sentiment_words = self._get_sentiment_words(processed_tokens)
        
        return SentimentResult(
            label=final_label,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            matched_rules=matched_rules,
            sentiment_words=sentiment_words
        )
    
    def _calculate_word_scores(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate sentiment scores based on word lexicons."""
        positive_score = sum(self.positive_words.get(token, 0) for token in tokens)
        negative_score = sum(self.negative_words.get(token, 0) for token in tokens)
        neutral_score = sum(1 for token in tokens if token in self.neutral_words)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def _calculate_pattern_scores(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores based on pattern matching."""
        positive_score = sum(weight for pattern, weight in self.positive_patterns
                           if re.search(pattern, text, re.IGNORECASE))
        
        negative_score = sum(weight for pattern, weight in self.negative_patterns
                           if re.search(pattern, text, re.IGNORECASE))
        
        neutral_score = sum(weight for pattern, weight in self.neutral_patterns
                          if re.search(pattern, text, re.IGNORECASE))
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def _calculate_emoji_scores(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores based on emojis."""
        positive_score = sum(weight for emoji, weight in self.positive_emojis.items()
                           if emoji in text)
        negative_score = sum(weight for emoji, weight in self.negative_emojis.items()
                           if emoji in text)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': 0
        }
    
    def _calculate_contextual_scores(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """Calculate contextual sentiment adjustments."""
        # Check for sarcasm indicators (more sophisticated detection)
        text_lower = text.lower()
        sarcasm_score = 0
        
        # Only detect sarcasm with specific contextual patterns
        sarcasm_patterns = [
            r'\byeah\s+right\b',
            r'\bsure\s+(?:thing|whatever)\b',
            r'\boh\s+great\b',
            r'\bjust\s+(?:fantastic|wonderful|perfect)\b',
            r'\breal(?:ly)?\s+(?:fantastic|wonderful|perfect)\b'
        ]
        
        for pattern in sarcasm_patterns:
            if re.search(pattern, text_lower):
                sarcasm_score += 1
        
        # Check for comparison patterns
        comparison_patterns = [
            r'\bbetter\s+than\b', r'\bworse\s+than\b', r'\bcompared\s+to\b',
            r'\bunlike\b', r'\binstead\s+of\b'
        ]
        comparison_score = sum(1 for pattern in comparison_patterns
                             if re.search(pattern, text, re.IGNORECASE))
        
        return {
            'positive': -sarcasm_score,  # Sarcasm reduces positive sentiment
            'negative': sarcasm_score,
            'neutral': comparison_score
        }
    
    def _calculate_indicator_scores(self, indicators: Dict[str, any]) -> Dict[str, float]:
        """Calculate scores based on text indicators."""
        positive_score = 0
        negative_score = 0
        neutral_score = 0
        
        # Exclamation marks (positive indicator)
        positive_score += min(indicators.get('exclamation_marks', 0) * 0.5, 2)
        
        # Question marks (neutral/negative indicator)
        question_marks = indicators.get('question_marks', 0)
        if question_marks > 2:
            negative_score += 1  # Too many questions suggest confusion/frustration
        else:
            neutral_score += question_marks * 0.3
        
        # Caps ratio (can be positive or negative)
        caps_ratio = indicators.get('caps_ratio', 0)
        if caps_ratio > 0.5:  # Excessive caps
            negative_score += 1
        elif caps_ratio > 0.2:  # Moderate caps
            positive_score += 0.5
        
        # Emoji counts
        positive_score += indicators.get('positive_emojis', 0) * 0.5
        negative_score += indicators.get('negative_emojis', 0) * 0.5
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def _apply_contextual_adjustments(
        self,
        text: str,
        tokens: List[str],
        positive_score: float,
        negative_score: float
    ) -> Tuple[float, float]:
        """Apply contextual adjustments to sentiment scores."""
        text_lower = text.lower()
        
        # Handle negation
        negation_multiplier = 1.0
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                # Check if negation affects sentiment words in the next 3 words
                window = tokens[i+1:i+4]
                if any(word in self.positive_words or word in self.negative_words for word in window):
                    negation_multiplier = -0.5
                    break
        
        # Apply negation
        if negation_multiplier < 0:
            positive_score, negative_score = negative_score * abs(negation_multiplier), positive_score * abs(negation_multiplier)
        
        # Handle intensifiers and diminishers
        intensity_multiplier = 1.0
        for token in tokens:
            if token in self.intensifiers:
                intensity_multiplier = max(intensity_multiplier, self.intensifiers[token])
            elif token in self.diminishers:
                intensity_multiplier = min(intensity_multiplier, self.diminishers[token])
        
        # Apply intensity adjustment
        positive_score *= intensity_multiplier
        negative_score *= intensity_multiplier
        
        return positive_score, negative_score
    
    def _get_matched_rules(self, text: str, tokens: List[str]) -> List[str]:
        """Get list of matched sentiment rules."""
        matched_rules = []
        
        # Check word matches
        for token in tokens:
            if token in self.positive_words:
                matched_rules.append(f"positive_word:{token}")
            elif token in self.negative_words:
                matched_rules.append(f"negative_word:{token}")
        
        # Check pattern matches
        for pattern, _ in self.positive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_rules.append(f"positive_pattern:{pattern}")
        
        for pattern, _ in self.negative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_rules.append(f"negative_pattern:{pattern}")
        
        # Check emoji matches
        for emoji in self.positive_emojis:
            if emoji in text:
                matched_rules.append(f"positive_emoji:{emoji}")
        
        for emoji in self.negative_emojis:
            if emoji in text:
                matched_rules.append(f"negative_emoji:{emoji}")
        
        return matched_rules
    
    def _get_sentiment_words(self, tokens: List[str]) -> List[str]:
        """Get list of sentiment-bearing words."""
        sentiment_words = []
        
        for token in tokens:
            if token in self.positive_words or token in self.negative_words:
                sentiment_words.append(token)
        
        return sentiment_words
    
    def get_sentiment_stats(self) -> Dict[str, int]:
        """Get statistics about the sentiment lexicons."""
        return {
            'positive_words': len(self.positive_words),
            'negative_words': len(self.negative_words),
            'neutral_words': len(self.neutral_words),
            'positive_patterns': len(self.positive_patterns),
            'negative_patterns': len(self.negative_patterns),
            'positive_emojis': len(self.positive_emojis),
            'negative_emojis': len(self.negative_emojis)
        }
    
    def classify(self, text: str) -> Dict[str, any]:
        """
        Simple classify method for backward compatibility with tests.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with sentiment and confidence
        """
        # Import here to avoid circular imports
        from .preprocessor import TextPreprocessor
        
        # Preprocess the text
        preprocessor = TextPreprocessor()
        processed_data = preprocessor.preprocess(text)
        processed_tokens = processed_data['processed']
        sentiment_indicators = processed_data.get('sentiment_indicators', {})
        
        # Get full sentiment result
        result = self.classify_sentiment(text, processed_tokens, sentiment_indicators)
        
        # Return simplified format expected by tests
        return {
            'sentiment': result.label.value,
            'confidence': result.confidence,
            'positive_score': result.positive_score,
            'negative_score': result.negative_score,
            'neutral_score': result.neutral_score,
            'matched_rules': result.matched_rules,
            'sentiment_words': result.sentiment_words
        }
