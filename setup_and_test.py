#!/usr/bin/env python3
"""
Setup and Test Script for YouTube Comment Sentiment Analysis

This script sets up the environment and runs comprehensive tests to ensure
the sentiment analysis system is working correctly.

Usage:
    python setup_and_test.py
    python setup_and_test.py --quick-test
    python setup_and_test.py --install-only
"""

import argparse
import asyncio
import sys
import subprocess
import os
from pathlib import Path
import logging

def setup_logging():
    """Setup logging for the setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_banner():
    """Print setup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              YouTube Comment Sentiment Analysis - Setup & Test               ║
║                                                                              ║
║  This script will set up your environment and test the system components    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required!")
        logger.error(f"Current version: {sys.version}")
        return False
    
    logger.info(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install project requirements."""
    logger = logging.getLogger(__name__)
    
    requirements_files = [
        "requirements.txt",
        "tests/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            logger.info(f"📦 Installing requirements from {req_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", req_file
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ Successfully installed requirements from {req_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install requirements from {req_file}")
                logger.error(f"Error output: {e.stderr}")
                return False
        else:
            logger.warning(f"⚠️  Requirements file not found: {req_file}")
    
    return True

def download_nltk_data():
    """Download required NLTK data."""
    logger = logging.getLogger(__name__)
    
    try:
        import nltk
        
        logger.info("📚 Downloading NLTK data...")
        
        # Download required NLTK data
        nltk_downloads = [
            'punkt',
            'stopwords', 
            'averaged_perceptron_tagger',
            'wordnet',
            'vader_lexicon'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
                logger.info(f"✅ Downloaded NLTK data: {item}")
            except Exception as e:
                logger.warning(f"⚠️  Could not download {item}: {e}")
        
        return True
        
    except ImportError:
        logger.error("❌ NLTK not installed - please install requirements first")
        return False

def create_directories():
    """Create necessary directories."""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data/outputs",
        "data/cache", 
        "data/raw",
        "logs",
        "tests/output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    logger = logging.getLogger(__name__)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        logger.info("📝 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        logger.info("✅ Created .env file")
    elif env_file.exists():
        logger.info("✅ .env file already exists")
    else:
        logger.warning("⚠️  No .env.example template found")
    
    return True

def test_imports():
    """Test if all modules can be imported."""
    logger = logging.getLogger(__name__)
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    modules_to_test = [
        "src.utils",
        "src.scraper", 
        "src.preprocessor",
        "src.sentiment_rules",
        "src.visualizer",
        "src.sentiment_analyzer",
        "config.settings",
        "config.logging_config"
    ]
    
    logger.info("🔍 Testing module imports...")
    
    for module in modules_to_test:
        try:
            __import__(module)
            logger.info(f"✅ Successfully imported: {module}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            return False
    
    return True

async def test_basic_functionality():
    """Test basic functionality of core components."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing basic functionality...")
    
    try:
        # Test utility functions
        from src.utils import extract_video_id, clean_text, validate_video_id
        
        # Test video ID extraction
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(test_url)
        assert video_id == "dQw4w9WgXcQ", f"Video ID extraction failed: {video_id}"
        logger.info("✅ Video ID extraction working")
        
        # Test text cleaning
        dirty_text = "  Hello   World!  \n\n  "
        clean = clean_text(dirty_text)
        assert clean == "Hello World!", f"Text cleaning failed: {clean}"
        logger.info("✅ Text cleaning working")
        
        # Test video ID validation
        assert validate_video_id("dQw4w9WgXcQ"), "Valid video ID rejected"
        assert not validate_video_id("invalid"), "Invalid video ID accepted"
        logger.info("✅ Video ID validation working")
        
        # Test text preprocessor
        from src.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        test_text = "This is AMAZING! I love it! 😊"
        result = preprocessor.preprocess(test_text)
        
        assert 'processed_text' in result, "Preprocessor missing processed_text"
        assert 'tokens' in result, "Preprocessor missing tokens"
        assert 'sentiment_indicators' in result, "Preprocessor missing sentiment_indicators"
        logger.info("✅ Text preprocessing working")
        
        # Test sentiment engine
        from src.sentiment_rules import SentimentRuleEngine
        engine = SentimentRuleEngine()
        
        test_cases = [
            ("I love this amazing video!", "positive"),
            ("This is terrible and boring", "negative"),
            ("This is okay, nothing special", "neutral")
        ]
        
        for text, expected in test_cases:
            result = engine.classify(text)
            assert result['sentiment'] == expected, f"Sentiment classification failed for: {text}"
        
        logger.info("✅ Sentiment classification working")
        
        # Test visualizer (basic initialization)
        from src.visualizer import SentimentVisualizer
        visualizer = SentimentVisualizer(output_dir="tests/output")
        logger.info("✅ Visualizer initialization working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False

async def test_full_pipeline():
    """Test the complete analysis pipeline with mock data."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Testing full analysis pipeline...")
    
    try:
        from src.sentiment_analyzer import SentimentAnalyzer
        
        # Test with a reliable video ID
        video_id = "dQw4w9WgXcQ"  # Rick Roll - should always exist
        
        analyzer = SentimentAnalyzer(
            video_id=video_id,
            max_comments=10,  # Small number for testing
            output_dir="tests/output/pipeline_test",
            use_cache=True
        )
        
        # Override scraper to return mock data for testing
        mock_comments = [
            {"text": "This is amazing! I love it! 😊", "author": "user1"},
            {"text": "Terrible video, waste of time 😞", "author": "user2"},
            {"text": "It's okay, nothing special", "author": "user3"},
            {"text": "Absolutely fantastic!", "author": "user4"},
            {"text": "I hate this so much", "author": "user5"}
        ]
        
        async def mock_scrape():
            return mock_comments
        
        analyzer.scraper.scrape_comments = mock_scrape
        
        # Run analysis
        results = await analyzer.analyze()
        
        # Verify results
        assert results is not None, "Analysis returned no results"
        assert 'statistics' in results, "Missing statistics in results"
        assert 'comments' in results, "Missing comments in results"
        assert results['statistics']['total_comments'] == len(mock_comments), "Comment count mismatch"
        
        # Test export
        export_files = analyzer.export_results(results, format="csv")
        assert len(export_files) > 0, "No export files generated"
        assert Path(export_files[0]).exists(), "Export file not created"
        
        logger.info("✅ Full pipeline test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        return False

def run_pytest():
    """Run the test suite using pytest."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Running pytest test suite...")
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--maxfail=5"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ All tests passed!")
            logger.info("Test output:")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Some tests failed!")
            logger.error("Test output:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        logger.warning("⚠️  pytest not found - skipping automated tests")
        return True

def print_summary(success_steps, total_steps):
    """Print setup summary."""
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    success_rate = (success_steps / total_steps) * 100
    
    print(f"✅ Successful steps: {success_steps}/{total_steps}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if success_steps == total_steps:
        print("\n🎉 Setup completed successfully!")
        print("You can now use the sentiment analysis system.")
        print("\nNext steps:")
        print("1. Run: python main.py --help")
        print("2. Try: python examples/basic_usage.py")
        print("3. Check: python examples/advanced_usage.py")
    else:
        print(f"\n⚠️  Setup completed with {total_steps - success_steps} issues.")
        print("Please check the errors above and resolve them.")
    
    print("="*60)

async def main():
    """Main setup and test function."""
    parser = argparse.ArgumentParser(description="Setup and test the sentiment analysis system")
    parser.add_argument("--quick-test", action="store_true", help="Run only quick tests")
    parser.add_argument("--install-only", action="store_true", help="Only install requirements")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Print banner
    print_banner()
    
    success_steps = 0
    total_steps = 0
    
    # Setup steps
    steps = [
        ("Check Python version", check_python_version),
        ("Install requirements", install_requirements),
        ("Download NLTK data", download_nltk_data),
        ("Create directories", create_directories),
        ("Create .env file", create_env_file),
    ]
    
    if not args.install_only:
        steps.extend([
            ("Test imports", test_imports),
            ("Test basic functionality", test_basic_functionality),
        ])
        
        if not args.quick_test and not args.skip_tests:
            steps.extend([
                ("Test full pipeline", test_full_pipeline),
                ("Run pytest suite", run_pytest),
            ])
    
    total_steps = len(steps)
    
    # Execute steps
    for step_name, step_func in steps:
        logger.info(f"\n🔄 {step_name}...")
        try:
            if asyncio.iscoroutinefunction(step_func):
                success = await step_func()
            else:
                success = step_func()
            
            if success:
                success_steps += 1
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
    
    # Print summary
    print_summary(success_steps, total_steps)
    
    return 0 if success_steps == total_steps else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
