# YouTube Comment Sentiment Analysis

An advanced, professional-grade sentiment analysis tool for YouTube comments with machine learning capabilities, modern web interface, and interactive dashboards.

## 🌟 Two Interface Options

### 🌐 **Streamlit Web App** (Recommended)
Modern, interactive web-based interface with advanced visualizations
```bash
python run_streamlit.py
# or
streamlit run streamlit_app.py
```

### 🖥️ **Desktop GUI** (Traditional)
Full-featured desktop application with tkinter interface
```bash
python main.py
```

## Features

### 🚀 Core Capabilities
- **Advanced Comment Scraping**: Robust YouTube comment extraction with rate limiting and error handling
- **Multi-Model Sentiment Analysis**: Ensemble approach using VADER, Transformers, and custom models
- **Professional Text Processing**: Advanced NLP preprocessing with spaCy and NLTK
- **Real-time Progress Tracking**: Live updates during scraping and analysis

### 🎨 User Interface
- **🌐 Streamlit Web App**: Modern, responsive web interface with interactive dashboards
- **🖥️ Desktop GUI**: Professional tkinter-based interface with rich styling
- **📊 Interactive Visualizations**: Rich charts with Plotly, Altair, and Matplotlib
- **☁️ Word Clouds**: Beautiful text visualizations by sentiment
- **📋 Data Tables**: Searchable, filterable data exploration
- **💾 Export Capabilities**: Multiple format support (CSV, JSON, Excel)

### 📊 Visualizations
- Sentiment distribution charts (pie, bar, treemap)
- Confidence score histograms and distributions
- Interactive scatter plots and correlation analysis
- Comment length and engagement analysis
- Top authors and engagement metrics
- Word cloud dashboards by sentiment
- Advanced statistical charts and heatmaps

### 🛠 Technical Excellence
- **Async Processing**: Non-blocking operations for smooth user experience
- **Database Integration**: SQLite storage for data persistence
- **Configuration Management**: Environment-based settings with validation
- **Comprehensive Logging**: Structured logging with Rich formatting
- **Error Handling**: Robust exception handling and recovery
- **Type Hints**: Full type annotation for better code quality

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows/macOS/Linux

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/youtube-comment-sentiment-analysis.git
cd youtube-comment-sentiment-analysis
```

2. **Create virtual environment**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download additional models**:
```bash
python -c "import nltk; nltk.download('all')"
python -m spacy download en_core_web_sm
```

5. **Run the application**:
```bash
python main.py
```

## Usage

### Basic Usage

1. **Launch the Application**:
   ```bash
   python main.py
   ```

2. **Enter YouTube URL**: Paste any valid YouTube video URL

3. **Configure Settings**:
   - Set number of comments to scrape (10-10,000)
   - Choose sorting method (top, new, time)
   - Enable/disable reply inclusion

4. **Start Analysis**: Click "Start Analysis" and monitor progress

5. **View Results**: Examine sentiment summary and detailed statistics

6. **Open Dashboard**: Access interactive visualizations

### Advanced Configuration

Create a `.env` file in the project root:

```env
# Application Settings
DEBUG=false
THEME=light

# Scraping Configuration
DEFAULT_COMMENT_LIMIT=500
RATE_LIMIT_DELAY=1.0
MAX_RETRIES=3

# ML Model Settings
MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
BATCH_SIZE=32
CONFIDENCE_THRESHOLD=0.7

# GUI Settings
WINDOW_WIDTH=1200
WINDOW_HEIGHT=800

# Logging
LOG_LEVEL=INFO
```

## Architecture

### Project Structure
```
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── src/
│   ├── config/           # Configuration management
│   │   ├── settings.py   # Settings with validation
│   │   └── __init__.py
│   ├── core/             # Core business logic
│   │   ├── comment_scraper.py    # YouTube API integration
│   │   ├── sentiment_analyzer.py # ML sentiment analysis
│   │   ├── data_processor.py     # Data processing pipeline
│   │   └── __init__.py
│   ├── gui/              # User interface
│   │   ├── main_window.py        # Main GUI application
│   │   ├── dashboard.py          # Interactive dashboard
│   │   └── __init__.py
│   └── utils/            # Utility functions
│       ├── logger.py             # Logging configuration
│       ├── validators.py         # URL and data validation
│       ├── decorators.py         # Retry, timing decorators
│       └── __init__.py
├── data/                 # Data storage
├── logs/                 # Application logs
├── exports/              # Exported data files
└── models/               # Cached ML models
```

### Key Components

#### 1. Comment Scraper (`comment_scraper.py`)
- **YouTube Integration**: Uses `youtube-comment-downloader` for robust scraping
- **Rate Limiting**: Prevents API abuse with configurable delays
- **Error Recovery**: Automatic retry with exponential backoff
- **Data Validation**: Ensures comment data integrity

#### 2. Sentiment Analyzer (`sentiment_analyzer.py`)
- **Multi-Model Approach**: Combines VADER and Transformer models
- **Advanced Preprocessing**: Text cleaning, tokenization, lemmatization
- **Ensemble Voting**: Weighted combination of multiple predictions
- **Confidence Scoring**: Provides prediction confidence metrics

#### 3. Data Processor (`data_processor.py`)
- **Database Integration**: SQLite for persistent storage
- **Batch Processing**: Efficient handling of large datasets
- **Export Capabilities**: Multiple format support
- **Statistical Analysis**: Comprehensive summary generation

#### 4. GUI Application (`main_window.py`)
- **Modern Interface**: Professional tkinter-based GUI
- **Async Operations**: Non-blocking UI with threading
- **Progress Tracking**: Real-time status updates
- **Error Handling**: User-friendly error messages

#### 5. Interactive Dashboard (`dashboard.py`)
- **Rich Visualizations**: matplotlib, seaborn, plotly integration
- **Multiple Views**: Overview, charts, detailed data, interactive
- **Export Options**: Save charts and data
- **Responsive Design**: Scalable interface elements

## Machine Learning Models

### Sentiment Analysis Pipeline

1. **Text Preprocessing**:
   - URL and mention removal
   - Tokenization and lemmatization
   - Stop word filtering
   - Special character handling

2. **Feature Extraction**:
   - Text length and structure metrics
   - Emoji and punctuation analysis
   - Capital letter ratios
   - Word count statistics

3. **Model Ensemble**:
   - **VADER**: Rule-based sentiment analysis
   - **RoBERTa**: Transformer-based neural model
   - **Custom Weights**: Confidence-based combination

4. **Post-processing**:
   - Confidence thresholding
   - Result validation
   - Statistical aggregation

### Supported Models

- `cardiffnlp/twitter-roberta-base-sentiment-latest` (default)
- `nlptown/bert-base-multilingual-uncased-sentiment`
- `microsoft/DialoGPT-medium`
- Custom fine-tuned models (configurable)

## Visualization Features

### Overview Dashboard
- **Statistics Grid**: Key metrics and counts
- **Sentiment Distribution**: Pie charts and bar graphs
- **Confidence Analysis**: Histogram distributions
- **Top Contributors**: Most active authors

### Advanced Charts
- **Treemap Visualization**: Hierarchical sentiment representation
- **Scatter Plots**: Likes vs confidence correlation
- **Timeline Analysis**: Sentiment trends over time
- **3D Visualizations**: Multi-dimensional data exploration

### Interactive Features
- **Hover Tooltips**: Detailed information on hover
- **Zoom and Pan**: Navigate large datasets
- **Filter Controls**: Dynamic data filtering
- **Export Options**: Save charts as images

## Data Export

### Supported Formats
- **CSV**: Comma-separated values for Excel/Google Sheets
- **JSON**: Structured data for APIs and applications
- **Excel**: Native XLSX format with formatting
- **SQLite**: Database export for analysis tools

### Export Data Structure
```json
{
  "id": "comment_unique_id",
  "video_id": "youtube_video_id",
  "text": "comment_text",
  "author": "commenter_name",
  "likes": 42,
  "reply_count": 5,
  "time_parsed": "2025-01-01T12:00:00Z",
  "sentiment": "positive",
  "sentiment_confidence": 0.857,
  "sentiment_scores": {
    "positive": 0.857,
    "negative": 0.089,
    "neutral": 0.054
  }
}
```

## Performance Optimization

### Scalability Features
- **Batch Processing**: Handle thousands of comments efficiently
- **Memory Management**: Optimized data structures
- **Async Operations**: Non-blocking UI operations
- **Caching**: Model and result caching
- **Database Indexing**: Fast data retrieval

### Resource Usage
- **Memory**: ~200-500MB typical usage
- **CPU**: Multi-threaded processing
- **Storage**: Configurable database size
- **Network**: Rate-limited API requests

## Troubleshooting

### Common Issues

1. **Model Download Errors**:
   ```bash
   python -c "import transformers; transformers.AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')"
   ```

2. **NLTK Data Missing**:
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

3. **spaCy Model Issues**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Permission Errors**:
   - Run as administrator (Windows)
   - Check file permissions (Unix)

### Debug Mode

Enable debug logging in `.env`:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Performance Issues

For large datasets (>1000 comments):
- Increase batch size in settings
- Enable database indexing
- Use filtered data views

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Install development dependencies**: `pip install -r requirements-dev.txt`
4. **Run tests**: `pytest tests/`
5. **Submit pull request**

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Formatting**: Black code formatting
- **Linting**: Flake8 compliance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **youtube-comment-downloader**: Comment scraping functionality
- **Transformers**: Hugging Face transformer models
- **NLTK**: Natural language processing toolkit
- **spaCy**: Advanced NLP processing
- **Plotly**: Interactive visualizations
- **tkinter**: GUI framework

---

**Built with ❤️ for YouTube content creators and researchers**
