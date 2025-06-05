#!/usr/bin/env python3
"""
Setup script for YouTube Comment Sentiment Analysis
Handles initial setup, dependency installation, and model downloads
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup basic logging for setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data",
        "logs", 
        "exports",
        "models"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    logger = logging.getLogger(__name__)
    
    try:
        import nltk
        logger.info("Downloading NLTK data...")
        
        nltk_downloads = [
            'punkt',
            'stopwords', 
            'wordnet',
            'vader_lexicon',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
                logger.info(f"Downloaded NLTK data: {item}")
            except Exception as e:
                logger.warning(f"Failed to download {item}: {e}")
        
        return True
    except ImportError:
        logger.error("NLTK not available. Please install dependencies first.")
        return False
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        logger.info("spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model: {e}")
        logger.info("You can download it manually later with: python -m spacy download en_core_web_sm")
        return False

def setup_environment():
    """Setup environment configuration"""
    logger = logging.getLogger(__name__)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        try:
            with open(env_example, 'r') as source:
                content = source.read()
            
            with open(env_file, 'w') as target:
                target.write(content)
            
            logger.info("Created .env file from .env.example")
        except Exception as e:
            logger.warning(f"Failed to create .env file: {e}")
    else:
        logger.info(".env file already exists or .env.example not found")

def test_installation():
    """Test if installation was successful"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test core imports
        logger.info("Testing installation...")
        
        # Test basic imports
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import plotly
        
        logger.info("Basic packages imported successfully")
        
        # Test NLP packages
        import nltk
        import spacy
        
        # Test if spacy model is available
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not available, but installation can continue")
        
        # Test transformers
        try:
            import transformers
            logger.info("Transformers package available")
        except ImportError:
            logger.warning("Transformers not available")
        
        logger.info("Installation test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger = setup_logging()
    
    print("=" * 60)
    print("YouTube Comment Sentiment Analysis - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    print("\n" + "=" * 40)
    print("Installing Dependencies")
    print("=" * 40)
    
    if not install_dependencies():
        logger.error("Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download NLTK data
    print("\n" + "=" * 40)
    print("Downloading NLTK Data")
    print("=" * 40)
    
    download_nltk_data()
    
    # Download spaCy model
    print("\n" + "=" * 40)
    print("Downloading spaCy Model")
    print("=" * 40)
    
    download_spacy_model()
    
    # Test installation
    print("\n" + "=" * 40)
    print("Testing Installation")
    print("=" * 40)
    
    if test_installation():
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nYou can now run the application with:")
        print("python main.py")
        print("\nFor more information, see README.md")
    else:
        print("\n" + "=" * 60)
        print("Setup completed with warnings")
        print("=" * 60)
        print("\nSome components may not work properly.")
        print("Please check the error messages above and install missing dependencies manually.")

if __name__ == "__main__":
    main()
