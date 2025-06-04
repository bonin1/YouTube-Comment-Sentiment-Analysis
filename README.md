# YouTube Comment Sentiment Analysis

A comprehensive tool for scraping YouTube comments and analyzing their sentiment using advanced rule-based classification techniques.

## 🚀 Features

- **Multi-method Comment Scraping**: Support for both youtube-comment-scraper and Selenium WebDriver
- **Advanced Text Preprocessing**: Stop word removal, tokenization, lemmatization, and emoji handling
- **Sophisticated Rule-based Sentiment Analysis**: Multi-layered classification with positive, negative, and neutral categories
- **Rich Visualizations**: Interactive pie charts, word clouds, and frequency analysis
- **Comprehensive Analytics**: Word frequency analysis per sentiment class
- **Professional Logging**: Detailed logging with multiple levels
- **Error Handling**: Robust error handling and retry mechanisms
- **Configurable Settings**: Easy configuration through environment variables

## 📋 Requirements

- Python 3.8+
- Chrome browser (for Selenium option)
- Internet connection for YouTube access

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YouTube-Comment-Sentiment-Analysis.git
cd YouTube-Comment-Sentiment-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

## 🚀 Usage

### Basic Usage

```python
from src.sentiment_analyzer import YouTubeSentimentAnalyzer

# Initialize analyzer
analyzer = YouTubeSentimentAnalyzer()

# Analyze comments from a YouTube video
video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
results = analyzer.analyze_video(video_url, max_comments=500)

# Generate visualizations
analyzer.create_sentiment_pie_chart(results)
analyzer.create_word_frequency_analysis(results)
```

### Advanced Usage

```python
# Custom configuration
analyzer = YouTubeSentimentAnalyzer(
    scraping_method='selenium',  # or 'youtube-comment-scraper'
    max_retries=3,
    timeout=30
)

# Analyze with custom parameters
results = analyzer.analyze_video(
    video_url,
    max_comments=1000,
    include_replies=True,
    filter_spam=True
)

# Export results
analyzer.export_results(results, format='csv')  # or 'json'
```

## 📊 Output

The tool generates:
- **Sentiment Distribution**: Interactive pie chart showing sentiment percentages
- **Word Clouds**: Visual representation of most frequent words per sentiment
- **Frequency Analysis**: Bar charts of top words for each sentiment category
- **Detailed Reports**: CSV/JSON exports with individual comment analysis

## 🏗️ Project Structure

```
YouTube-Comment-Sentiment-Analysis/
├── src/
│   ├── __init__.py
│   ├── scraper.py              # Comment scraping functionality
│   ├── preprocessor.py         # Text preprocessing utilities
│   ├── sentiment_rules.py      # Rule-based sentiment classification
│   ├── visualizer.py          # Data visualization components
│   ├── sentiment_analyzer.py   # Main analyzer class
│   └── utils.py               # Utility functions
├── config/
│   ├── __init__.py
│   ├── settings.py            # Configuration settings
│   └── logging_config.py      # Logging configuration
├── data/
│   ├── outputs/               # Generated reports and visualizations
│   └── cache/                 # Cached data
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_preprocessor.py
│   ├── test_sentiment_rules.py
│   └── test_analyzer.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_analysis.py
├── docs/
│   └── API.md
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── main.py
```

## 🔧 Configuration

Create a `.env` file based on `.env.example`:

```env
# Scraping settings
DEFAULT_SCRAPING_METHOD=youtube-comment-scraper
MAX_COMMENTS=500
TIMEOUT=30
MAX_RETRIES=3

# Output settings
OUTPUT_DIR=data/outputs
CACHE_DIR=data/cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/sentiment_analysis.log
```

## 📈 Sentiment Classification Rules

The system uses sophisticated rule-based classification:

### Positive Indicators
- Keywords: love, awesome, amazing, great, excellent, fantastic, wonderful, perfect, best, brilliant
- Emojis: 😊, 😍, 👍, ❤️, 🔥, ⭐, 🎉, 👏, 💯, 😄
- Patterns: Multiple exclamation marks, all caps positive words

### Negative Indicators
- Keywords: hate, boring, terrible, awful, worst, bad, horrible, disgusting, stupid, annoying
- Emojis: 👎, 😞, 😡, 💔, 😢, 🤮, 😤, 😠, 💩, ❌
- Patterns: Multiple question marks indicating confusion/frustration

### Neutral Indicators
- Informational content
- Questions without emotional context
- Balanced statements

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Selenium project for web automation
- YouTube for providing accessible comment data
- The open-source community for inspiration and tools

## 📞 Support

If you encounter any issues or have questions, please [create an issue](https://github.com/yourusername/YouTube-Comment-Sentiment-Analysis/issues) on GitHub.