# API Documentation

## YouTube Comment Sentiment Analysis API

This document provides comprehensive API documentation for the YouTube Comment Sentiment Analysis project.

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Utility Functions](#utility-functions)
4. [Configuration](#configuration)
5. [Examples](#examples)

## Overview

The YouTube Comment Sentiment Analysis system is built with a modular architecture consisting of several key components:

- **SentimentAnalyzer**: Main orchestrator class
- **CommentScraper**: Handles YouTube comment extraction
- **TextPreprocessor**: Processes and cleans text data
- **SentimentRuleEngine**: Performs rule-based sentiment classification
- **SentimentVisualizer**: Generates charts and visualizations

## Core Classes

### SentimentAnalyzer

The main class that orchestrates the entire sentiment analysis pipeline.

```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    video_id="dQw4w9WgXcQ",
    max_comments=1000,
    scraping_method="auto",
    sort_by="top",
    output_dir="data/outputs",
    use_cache=True
)

# Run analysis
results = await analyzer.analyze()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_id` | str | Required | YouTube video ID to analyze |
| `max_comments` | int | 1000 | Maximum number of comments to scrape |
| `scraping_method` | str | "auto" | Method for scraping ("auto", "requests", "selenium") |
| `sort_by` | str | "top" | Comment sorting ("top", "new") |
| `output_dir` | str | "data/outputs" | Directory for saving results |
| `use_cache` | bool | True | Enable/disable result caching |

#### Methods

##### `analyze() -> dict`

Runs the complete sentiment analysis pipeline.

**Returns:** Dictionary containing analysis results with the following structure:

```python
{
    "statistics": {
        "total_comments": int,
        "sentiment_distribution": {
            "positive": int,
            "negative": int,
            "neutral": int
        },
        "processing_time": float,
        "confidence_statistics": {
            "mean": float,
            "median": float,
            "std": float,
            "min": float,
            "max": float
        }
    },
    "comments": [
        {
            "text": str,
            "processed_text": str,
            "sentiment": str,
            "confidence": float,
            "author": str,
            "timestamp": str
        }
    ],
    "word_frequency": {
        "positive": {"word": count},
        "negative": {"word": count},
        "neutral": {"word": count}
    }
}
```

##### `export_results(results: dict, format: str = "csv") -> list`

Exports analysis results to files.

**Parameters:**
- `results`: Results dictionary from `analyze()`
- `format`: Export format ("csv", "json", "excel", "all")

**Returns:** List of file paths where results were saved.

##### `generate_visualizations(results: dict, interactive: bool = False) -> list`

Generates visualization charts.

**Parameters:**
- `results`: Results dictionary from `analyze()`
- `interactive`: Generate interactive plots using Plotly

**Returns:** List of file paths where visualizations were saved.

### CommentScraper

Handles scraping comments from YouTube videos.

```python
from src.scraper import CommentScraper

scraper = CommentScraper(
    max_comments=1000,
    method="auto",
    sort_by="top"
)

comments = await scraper.scrape_comments(video_id)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_comments` | int | 1000 | Maximum comments to scrape |
| `method` | str | "auto" | Scraping method |
| `sort_by` | str | "top" | Comment sorting method |
| `filter_spam` | bool | True | Filter spam/duplicate comments |

#### Methods

##### `scrape_comments(video_id: str) -> list`

Scrapes comments from a YouTube video.

**Parameters:**
- `video_id`: YouTube video ID

**Returns:** List of comment dictionaries:

```python
[
    {
        "text": str,
        "author": str,
        "likes": int,
        "timestamp": str,
        "replies": int
    }
]
```

### TextPreprocessor

Processes and cleans text data for analysis.

```python
from src.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    handle_emojis=True
)

result = preprocessor.preprocess(text)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_stopwords` | bool | True | Remove common stop words |
| `lemmatize` | bool | True | Apply lemmatization |
| `handle_emojis` | bool | True | Convert emojis to text |
| `expand_contractions` | bool | True | Expand contractions |
| `remove_special_chars` | bool | False | Remove special characters |

#### Methods

##### `preprocess(text: str) -> dict`

Preprocesses text for sentiment analysis.

**Parameters:**
- `text`: Raw text to preprocess

**Returns:** Preprocessing results:

```python
{
    "original_text": str,
    "processed_text": str,
    "tokens": list,
    "sentiment_indicators": {
        "positive_count": int,
        "negative_count": int,
        "neutral_count": int,
        "emoji_sentiment": str,
        "has_intensifiers": bool,
        "has_negation": bool
    }
}
```

### SentimentRuleEngine

Performs rule-based sentiment classification.

```python
from src.sentiment_rules import SentimentRuleEngine

engine = SentimentRuleEngine(
    use_emoji_sentiment=True,
    handle_negation=True,
    apply_intensifiers=True
)

result = engine.classify(text)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_emoji_sentiment` | bool | True | Consider emoji sentiment |
| `handle_negation` | bool | True | Handle negation patterns |
| `apply_intensifiers` | bool | True | Apply intensifier weights |
| `custom_lexicon` | dict | None | Custom sentiment lexicon |

#### Methods

##### `classify(text: str) -> dict`

Classifies text sentiment using rule-based approach.

**Parameters:**
- `text`: Text to classify

**Returns:** Classification results:

```python
{
    "sentiment": str,  # "positive", "negative", or "neutral"
    "confidence": float,  # 0.0 to 1.0
    "scores": {
        "positive": float,
        "negative": float,
        "neutral": float
    },
    "features": {
        "word_matches": list,
        "emoji_sentiment": str,
        "negation_detected": bool,
        "intensifiers": list
    }
}
```

### SentimentVisualizer

Generates charts and visualizations.

```python
from src.visualizer import SentimentVisualizer

visualizer = SentimentVisualizer(
    output_dir="data/outputs",
    style="modern"
)

# Generate pie chart
chart_path = visualizer.create_sentiment_pie_chart(sentiment_counts)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | "data/outputs" | Output directory for charts |
| `style` | str | "modern" | Chart style theme |
| `color_scheme` | str | "default" | Color scheme for charts |

#### Methods

##### `create_sentiment_pie_chart(sentiment_counts: dict) -> str`

Creates a pie chart showing sentiment distribution.

##### `create_word_frequency_chart(word_freq: dict) -> str`

Creates word frequency charts by sentiment.

##### `create_confidence_distribution(confidences: list) -> str`

Creates confidence score distribution chart.

##### `create_interactive_dashboard(results: dict) -> str`

Creates interactive dashboard with multiple visualizations.

## Utility Functions

### Video ID Extraction

```python
from src.utils import extract_video_id

video_id = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
# Returns: "dQw4w9WgXcQ"
```

### Text Cleaning

```python
from src.utils import clean_text

cleaned = clean_text("  Text with   extra  spaces  ")
# Returns: "Text with extra spaces"
```

### Progress Tracking

```python
from src.utils import ProgressTracker

tracker = ProgressTracker(total=100, description="Processing")
for i in range(100):
    # Do work
    tracker.update(1)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/sentiment_analysis.log

# Output Configuration
DEFAULT_OUTPUT_DIR=data/outputs
DEFAULT_MAX_COMMENTS=1000

# Scraping Configuration
DEFAULT_SCRAPING_METHOD=auto
SELENIUM_HEADLESS=true

# Cache Configuration
ENABLE_CACHE=true
CACHE_DIRECTORY=data/cache
```

### Settings

```python
from config.settings import Settings

# Access configuration
max_comments = Settings.DEFAULT_MAX_COMMENTS
output_dir = Settings.DEFAULT_OUTPUT_DIR
```

## Examples

### Basic Usage

```python
import asyncio
from src.sentiment_analyzer import SentimentAnalyzer

async def analyze_video():
    analyzer = SentimentAnalyzer(video_id="dQw4w9WgXcQ")
    results = await analyzer.analyze()
    
    # Export results
    analyzer.export_results(results, format="csv")
    
    # Generate visualizations
    analyzer.generate_visualizations(results)

asyncio.run(analyze_video())
```

### Batch Processing

```python
async def batch_analyze(video_ids):
    results = []
    
    for video_id in video_ids:
        analyzer = SentimentAnalyzer(video_id=video_id)
        result = await analyzer.analyze()
        results.append(result)
    
    return results
```

### Custom Configuration

```python
analyzer = SentimentAnalyzer(
    video_id="VIDEO_ID",
    max_comments=500,
    scraping_method="selenium",
    sort_by="new",
    output_dir="custom_output",
    use_cache=False
)
```

### Error Handling

```python
try:
    results = await analyzer.analyze()
    if results:
        print("Analysis successful!")
    else:
        print("Analysis failed - no results")
except Exception as e:
    print(f"Error: {e}")
```

## Return Codes

When using the command-line interface:

- `0`: Success
- `1`: General error or user interruption
- `2`: Invalid arguments or configuration

## Logging

The system uses structured logging with different levels:

- `DEBUG`: Detailed debugging information
- `INFO`: General information about progress
- `WARNING`: Warning messages for non-critical issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors that stop execution

Configure logging level through environment variables or command-line arguments.
