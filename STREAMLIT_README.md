# 🎬 YouTube Comment Sentiment Analysis - Streamlit Edition

## 🚀 Quick Start

### Method 1: Using the launcher script
```bash
python run_streamlit.py
```

### Method 2: Direct Streamlit command
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## ✨ Features

### 🎯 Core Functionality
- **YouTube Comment Scraping** - Extract comments from any public YouTube video
- **Advanced Sentiment Analysis** - ML-powered sentiment classification using transformer models
- **Real-time Progress Tracking** - Live updates during analysis
- **Interactive Visualizations** - Multiple chart types and dashboards

### 📊 Visualizations & Analytics
- **Sentiment Overview** - Pie charts and distribution analysis
- **Comment Analysis** - Detailed comment inspection with filtering
- **Word Clouds** - Visual representation of common terms by sentiment
- **Advanced Charts** - Correlation analysis, engagement metrics, author statistics
- **Data Table** - Searchable and filterable data table

### 💾 Data Export
- **CSV Format** - Standard comma-separated values
- **JSON Format** - Structured data format
- **Excel Format** - Spreadsheet format with formatting

## 🎮 How to Use

1. **Enter YouTube URL** - Paste a valid YouTube video URL in the sidebar
2. **Configure Parameters** - Set number of comments and whether to include replies
3. **Start Analysis** - Click "Start Analysis" and wait for completion
4. **Explore Results** - Navigate through different tabs to view insights
5. **Export Data** - Download results in your preferred format

## 📋 Interface Layout

### 🎛️ Sidebar Controls
- URL input and validation
- Analysis parameters (comment limit, include replies)
- Export options and data download

### 📊 Main Dashboard Tabs

#### 1. **📈 Sentiment Overview**
- Overall sentiment distribution pie chart
- Confidence score distribution
- Sentiment vs engagement analysis
- Average likes by sentiment

#### 2. **💬 Comment Analysis**
- Detailed comment viewer with filtering
- Sort by likes, confidence, or text length
- Individual comment cards with sentiment indicators

#### 3. **☁️ Word Clouds**
- Positive sentiment word cloud
- Negative sentiment word cloud
- Overall word cloud for all comments

#### 4. **📋 Data Table**
- Searchable and filterable data table
- Customizable column selection
- Data summary statistics

#### 5. **📊 Advanced Charts**
- Comment length analysis by sentiment
- Correlation heatmap
- Top authors by engagement
- Text length distribution

## 🎨 Visual Features

### 🎯 Color Coding
- **Green** - Positive sentiment
- **Red** - Negative sentiment  
- **Gray** - Neutral sentiment

### 📱 Responsive Design
- Optimized for desktop and tablet viewing
- Clean, modern interface
- Interactive hover effects and tooltips

## ⚡ Performance Tips

- **Start Small** - Begin with 50-100 comments for quick testing
- **Popular Videos** - Use videos with active comment sections
- **Include Replies** - Enable for more comprehensive analysis
- **Export Early** - Download data after analysis for backup

## 🔧 Technical Details

### 🧠 ML Models
- **Transformer-based** sentiment analysis
- **Pre-trained** RoBERTa model optimized for social media text
- **Confidence scoring** for result reliability

### 📊 Visualization Libraries
- **Plotly** - Interactive charts and graphs
- **Altair** - Statistical visualizations
- **WordCloud** - Text visualization
- **Matplotlib/Seaborn** - Additional chart support

### 🔗 Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Async processing** - Non-blocking comment scraping

## 🆘 Troubleshooting

### Common Issues

**"No comments found"**
- Video may have comments disabled
- Video may be private or age-restricted
- Try a different video

**"Analysis failed"**
- Check internet connection
- Verify YouTube URL format
- Try reducing comment limit

**Slow performance**
- Reduce number of comments
- Check system resources
- Restart the application

## 🔮 Future Enhancements

- **Real-time analysis** - Live comment monitoring
- **Comparative analysis** - Multiple video comparison
- **Emotion detection** - Beyond positive/negative/neutral
- **Export templates** - Pre-formatted reports
- **Batch processing** - Multiple videos at once

---

**Built with ❤️ using Streamlit, transformers, and modern ML techniques**
