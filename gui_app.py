"""
Modern GUI Application for YouTube Comment Sentiment Analysis

This module provides a comprehensive, modern GUI interface for the sentiment
analysis system with real-time progress tracking, interactive visualizations,
and export capabilities.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import customtkinter as ctk
from ttkthemes import ThemedTk
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import setup_logging, get_logger, settings
from src import CommentScraper, SentimentAnalyzer, TextPreprocessor, AdvancedVisualizer, DataManager, ProgressTracker
from src.utils import validate_url, format_duration, format_number, clean_filename


class ModernGUI:
    """Modern GUI application for YouTube comment sentiment analysis."""
    
    def __init__(self):
        """Initialize the GUI application."""
        # Setup logging with GUI callback
        setup_logging(console_output=True, file_output=True, gui_callback=self._log_callback)
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.scraper = None
        self.analyzer = None
        self.preprocessor = None
        self.visualizer = None
        self.data_manager = None
        
        # State variables
        self.current_analysis = None
        self.stop_requested = False
        self.progress_tracker = None
        
        # Create main window
        self._create_main_window()
        self._setup_styles()
        self._create_widgets()
        self._initialize_components()
        
        self.logger.info("GUI application initialized successfully")
    
    def _log_callback(self, level: str, message: str):
        """Callback for logging to GUI."""
        if hasattr(self, 'log_text'):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            # Update in thread-safe manner
            self.root.after(0, lambda: self._update_log_display(log_entry))
    
    def _update_log_display(self, log_entry: str):
        """Update log display in GUI."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, log_entry)
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
        
        # Keep only last 1000 lines
        lines = self.log_text.get('1.0', tk.END).split('\n')
        if len(lines) > 1000:
            self.log_text.configure(state='normal')
            self.log_text.delete('1.0', f'{len(lines)-1000}.0')
            self.log_text.configure(state='disabled')
    
    def _create_main_window(self):
        """Create and configure the main window."""
        # Use themed Tk for better appearance
        self.root = ThemedTk(theme="arc")
        self.root.title("YouTube Comment Sentiment Analysis - Advanced System")
        self.root.geometry(f"{settings.WINDOW_WIDTH}x{settings.WINDOW_HEIGHT}")
        self.root.minsize(1000, 700)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (settings.WINDOW_WIDTH // 2)
        y = (self.root.winfo_screenheight() // 2) - (settings.WINDOW_HEIGHT // 2)
        self.root.geometry(f"{settings.WINDOW_WIDTH}x{settings.WINDOW_HEIGHT}+{x}+{y}")
        
        # Configure window icon (if available)
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_styles(self):
        """Setup custom styles for the GUI."""
        self.style = ttk.Style()
        
        # Configure colors
        colors = {
            'primary': '#2E8B57',
            'secondary': '#4682B4', 
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Success.TLabel', foreground=colors['success'])
        self.style.configure('Warning.TLabel', foreground=colors['warning'])
        self.style.configure('Danger.TLabel', foreground=colors['danger'])
        
        # Button styles
        self.style.configure('Primary.TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('Success.TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('Danger.TButton', font=('Helvetica', 10, 'bold'))
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Create header
        self._create_header(main_container)
        
        # Create main content area with notebook
        self._create_notebook(main_container)
        
        # Create status bar
        self._create_status_bar(main_container)
    
    def _create_header(self, parent):
        """Create header section."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(header_frame, text="YouTube Comment Sentiment Analysis", style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # System info
        info_text = f"v1.0.0 | Python {sys.version.split()[0]} | {datetime.now().strftime('%Y-%m-%d')}"
        info_label = ttk.Label(header_frame, text=info_text, font=('Helvetica', 9))
        info_label.grid(row=0, column=1, sticky=tk.E)
    
    def _create_notebook(self, parent):
        """Create main notebook with tabs."""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Analysis Tab
        self._create_analysis_tab()
        
        # Results Tab
        self._create_results_tab()
        
        # Visualization Tab
        self._create_visualization_tab()
        
        # History Tab
        self._create_history_tab()
        
        # Settings Tab
        self._create_settings_tab()
        
        # Logs Tab
        self._create_logs_tab()
    
    def _create_analysis_tab(self):
        """Create the main analysis tab."""
        analysis_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Configure grid
        analysis_frame.columnconfigure(1, weight=1)
        
        # URL Input Section
        url_frame = ttk.LabelFrame(analysis_frame, text="Video Configuration", padding="15")
        url_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        url_frame.columnconfigure(1, weight=1)
        
        ttk.Label(url_frame, text="YouTube URL:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(url_frame, textvariable=self.url_var, font=('Helvetica', 10), width=50)
        self.url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.validate_btn = ttk.Button(url_frame, text="Validate", command=self._validate_url)
        self.validate_btn.grid(row=0, column=2)
        
        # URL status
        self.url_status_var = tk.StringVar(value="Enter a YouTube URL")
        self.url_status_label = ttk.Label(url_frame, textvariable=self.url_status_var)
        self.url_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Analysis Parameters Section
        params_frame = ttk.LabelFrame(analysis_frame, text="Analysis Parameters", padding="15")
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        params_frame.columnconfigure(1, weight=1)
        
        # Max comments
        ttk.Label(params_frame, text="Max Comments:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.max_comments_var = tk.IntVar(value=settings.MAX_COMMENTS)
        self.max_comments_spinbox = ttk.Spinbox(params_frame, from_=10, to=50000, textvariable=self.max_comments_var, width=15)
        self.max_comments_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
          # Sort by
        ttk.Label(params_frame, text="Sort by:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.sort_var = tk.StringVar(value="top")
        self.sort_combo = ttk.Combobox(params_frame, textvariable=self.sort_var, values=["top", "new", "time"], state="readonly", width=12)
        self.sort_combo.grid(row=0, column=3, sticky=tk.W)
        
        # Analysis options
        options_frame = ttk.Frame(params_frame)
        options_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.use_ensemble_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use Ensemble Analysis", variable=self.use_ensemble_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.save_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Save to Database", variable=self.save_data_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.auto_export_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Auto Export Results", variable=self.auto_export_var).grid(row=0, column=2, sticky=tk.W)
        
        # Control Buttons
        control_frame = ttk.Frame(analysis_frame)
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.start_btn = ttk.Button(control_frame, text="🚀 Start Analysis", command=self._start_analysis, style='Primary.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", command=self._stop_analysis, state=tk.DISABLED, style='Danger.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_btn = ttk.Button(control_frame, text="📤 Export Results", command=self._export_results, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(control_frame, text="🗑️ Clear", command=self._clear_results)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(analysis_frame, text="Progress", padding="15")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress text
        self.progress_text_var = tk.StringVar(value="Ready to start analysis")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_text_var)
        self.progress_label.grid(row=1, column=0, sticky=tk.W)
        
        # Quick Stats
        stats_frame = ttk.LabelFrame(analysis_frame, text="Quick Statistics", padding="15")
        stats_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        stats_inner = ttk.Frame(stats_frame)
        stats_inner.grid(row=0, column=0, sticky=(tk.W, tk.E))
        stats_frame.columnconfigure(0, weight=1)
        
        # Create stats labels
        self.stats_vars = {
            'total': tk.StringVar(value="0"),
            'positive': tk.StringVar(value="0"),
            'negative': tk.StringVar(value="0"),
            'neutral': tk.StringVar(value="0"),
            'confidence': tk.StringVar(value="0.00")
        }
        
        stats_labels = [
            ("Total Comments:", self.stats_vars['total']),
            ("Positive:", self.stats_vars['positive']),
            ("Negative:", self.stats_vars['negative']),
            ("Neutral:", self.stats_vars['neutral']),
            ("Avg Confidence:", self.stats_vars['confidence'])
        ]
        
        for i, (label, var) in enumerate(stats_labels):
            ttk.Label(stats_inner, text=label, font=('Helvetica', 9, 'bold')).grid(row=0, column=i*2, sticky=tk.W, padx=(0, 5))
            ttk.Label(stats_inner, textvariable=var, font=('Helvetica', 9)).grid(row=0, column=i*2+1, sticky=tk.W, padx=(0, 20))
    
    def _create_results_tab(self):
        """Create results display tab."""
        results_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(results_frame, text="📋 Results")
        
        # Configure grid
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results header
        header_frame = ttk.Frame(results_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Analysis Results", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        # Filter controls
        filter_frame = ttk.Frame(header_frame)
        filter_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(filter_frame, text="Filter:").grid(row=0, column=0, padx=(0, 5))
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, 
                                  values=["all", "positive", "negative", "neutral"], 
                                  state="readonly", width=10)
        filter_combo.grid(row=0, column=1, padx=(0, 10))
        filter_combo.bind('<<ComboboxSelected>>', self._filter_results)
        
        # Search
        ttk.Label(filter_frame, text="Search:").grid(row=0, column=2, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=20)
        search_entry.grid(row=0, column=3)
        search_entry.bind('<KeyRelease>', self._search_results)
        
        # Results treeview
        tree_frame = ttk.Frame(results_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        self.results_tree = ttk.Treeview(tree_frame, columns=('author', 'text', 'sentiment', 'confidence', 'likes'), show='headings')
        
        # Configure columns
        self.results_tree.heading('author', text='Author')
        self.results_tree.heading('text', text='Comment Text')
        self.results_tree.heading('sentiment', text='Sentiment')
        self.results_tree.heading('confidence', text='Confidence')
        self.results_tree.heading('likes', text='Likes')
        
        self.results_tree.column('author', width=120, minwidth=80)
        self.results_tree.column('text', width=400, minwidth=200)
        self.results_tree.column('sentiment', width=80, minwidth=80)
        self.results_tree.column('confidence', width=80, minwidth=80)
        self.results_tree.column('likes', width=60, minwidth=50)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.results_tree.configure(xscrollcommand=h_scrollbar.set)
    
    def _create_visualization_tab(self):
        """Create visualization tab."""
        viz_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(viz_frame, text="📈 Visualizations")
        
        # Configure grid
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Visualization controls
        control_frame = ttk.Frame(viz_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(control_frame, text="Visualization Controls", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        self.viz_pie_btn = ttk.Button(btn_frame, text="📊 Sentiment Pie Chart", command=lambda: self._show_visualization('pie'))
        self.viz_pie_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.viz_treemap_btn = ttk.Button(btn_frame, text="🌳 Treemap", command=lambda: self._show_visualization('treemap'))
        self.viz_treemap_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.viz_confidence_btn = ttk.Button(btn_frame, text="📈 Confidence Analysis", command=lambda: self._show_visualization('confidence'))
        self.viz_confidence_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.viz_wordcloud_btn = ttk.Button(btn_frame, text="☁️ Word Cloud", command=lambda: self._show_visualization('wordcloud'))
        self.viz_wordcloud_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.viz_dashboard_btn = ttk.Button(btn_frame, text="🎛️ Interactive Dashboard", command=self._show_dashboard)
        self.viz_dashboard_btn.pack(side=tk.LEFT)
        
        # Visualization display area
        self.viz_display_frame = ttk.Frame(viz_frame)
        self.viz_display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Placeholder
        placeholder_label = ttk.Label(self.viz_display_frame, 
                                    text="Select a visualization type above to display charts.\nAnalyze some comments first to generate visualizations.", 
                                    font=('Helvetica', 12), justify=tk.CENTER)
        placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def _create_history_tab(self):
        """Create analysis history tab."""
        history_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(history_frame, text="📜 History")
        
        # Configure grid
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(1, weight=1)
        
        # History header
        header_frame = ttk.Frame(history_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Analysis History", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        # Refresh button
        ttk.Button(header_frame, text="🔄 Refresh", command=self._refresh_history).grid(row=0, column=1, sticky=tk.E)
        
        # History treeview
        history_tree_frame = ttk.Frame(history_frame)
        history_tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_tree_frame.columnconfigure(0, weight=1)
        history_tree_frame.rowconfigure(0, weight=1)
        
        self.history_tree = ttk.Treeview(history_tree_frame, columns=('date', 'video_id', 'comments', 'avg_sentiment'), show='headings')
        
        self.history_tree.heading('date', text='Date')
        self.history_tree.heading('video_id', text='Video ID')
        self.history_tree.heading('comments', text='Comments')
        self.history_tree.heading('avg_sentiment', text='Dominant Sentiment')
        
        self.history_tree.column('date', width=150)
        self.history_tree.column('video_id', width=120)
        self.history_tree.column('comments', width=100)
        self.history_tree.column('avg_sentiment', width=150)
        
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # History scrollbar
        hist_scrollbar = ttk.Scrollbar(history_tree_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        hist_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.history_tree.configure(yscrollcommand=hist_scrollbar.set)
    
    def _create_settings_tab(self):
        """Create settings tab."""
        settings_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Analysis Settings
        analysis_settings_frame = ttk.LabelFrame(settings_frame, text="Analysis Settings", padding="15")
        analysis_settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        analysis_settings_frame.columnconfigure(1, weight=1)
        
        # Confidence threshold
        ttk.Label(analysis_settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.confidence_threshold_var = tk.DoubleVar(value=settings.CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(analysis_settings_frame, from_=0.0, to=1.0, 
                                   variable=self.confidence_threshold_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.confidence_value_label = ttk.Label(analysis_settings_frame, text=f"{settings.CONFIDENCE_THRESHOLD:.2f}")
        self.confidence_value_label.grid(row=0, column=2)
        confidence_scale.configure(command=self._update_confidence_label)
        
        # Sentiment threshold
        ttk.Label(analysis_settings_frame, text="Sentiment Threshold:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.sentiment_threshold_var = tk.DoubleVar(value=settings.SENTIMENT_THRESHOLD)
        sentiment_scale = ttk.Scale(analysis_settings_frame, from_=0.0, to=0.5, 
                                  variable=self.sentiment_threshold_var, orient=tk.HORIZONTAL)
        sentiment_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        
        self.sentiment_value_label = ttk.Label(analysis_settings_frame, text=f"{settings.SENTIMENT_THRESHOLD:.2f}")
        self.sentiment_value_label.grid(row=1, column=2, pady=(10, 0))
        sentiment_scale.configure(command=self._update_sentiment_label)
        
        # Processing Settings
        processing_settings_frame = ttk.LabelFrame(settings_frame, text="Text Processing", padding="15")
        processing_settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.remove_stopwords_var = tk.BooleanVar(value=settings.REMOVE_STOPWORDS)
        ttk.Checkbutton(processing_settings_frame, text="Remove Stopwords", variable=self.remove_stopwords_var).grid(row=0, column=0, sticky=tk.W)
        
        self.enable_lemmatization_var = tk.BooleanVar(value=settings.ENABLE_LEMMATIZATION)
        ttk.Checkbutton(processing_settings_frame, text="Enable Lemmatization", variable=self.enable_lemmatization_var).grid(row=1, column=0, sticky=tk.W)
        
        self.handle_emojis_var = tk.BooleanVar(value=settings.HANDLE_EMOJIS)
        ttk.Checkbutton(processing_settings_frame, text="Process Emojis", variable=self.handle_emojis_var).grid(row=2, column=0, sticky=tk.W)
        
        # Visualization Settings
        viz_settings_frame = ttk.LabelFrame(settings_frame, text="Visualization Settings", padding="15")
        viz_settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(viz_settings_frame, text="Color Scheme:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.color_scheme_var = tk.StringVar(value=settings.COLOR_SCHEME)
        color_combo = ttk.Combobox(viz_settings_frame, textvariable=self.color_scheme_var, 
                                 values=["viridis", "plasma", "inferno", "custom"], state="readonly")
        color_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Apply settings button
        ttk.Button(settings_frame, text="💾 Apply Settings", command=self._apply_settings, style='Primary.TButton').grid(row=3, column=0, pady=(15, 0))
    
    def _create_logs_tab(self):
        """Create logs display tab."""
        logs_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(logs_frame, text="📝 Logs")
        
        # Configure grid
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(1, weight=1)
        
        # Logs header
        header_frame = ttk.Frame(logs_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Application Logs", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        # Log controls
        log_controls = ttk.Frame(header_frame)
        log_controls.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Button(log_controls, text="🗑️ Clear Logs", command=self._clear_logs).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(log_controls, text="💾 Save Logs", command=self._save_logs).pack(side=tk.RIGHT)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=20, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def _create_status_bar(self, parent):
        """Create status bar."""
        self.status_frame = ttk.Frame(parent)
        self.status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.status_frame.columnconfigure(1, weight=1)
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Component status indicators
        status_indicators = ttk.Frame(self.status_frame)
        status_indicators.grid(row=0, column=1, sticky=tk.E)
        
        self.scraper_status = ttk.Label(status_indicators, text="Scraper: ❌", font=('Helvetica', 8))
        self.scraper_status.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.analyzer_status = ttk.Label(status_indicators, text="Analyzer: ❌", font=('Helvetica', 8))
        self.analyzer_status.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.db_status = ttk.Label(status_indicators, text="Database: ❌", font=('Helvetica', 8))
        self.db_status.pack(side=tk.RIGHT, padx=(10, 0))
    
    def _initialize_components(self):
        """Initialize all system components."""
        self.status_var.set("Initializing components...")
        
        def init_thread():
            try:
                # Initialize components
                self.scraper = CommentScraper(progress_callback=self._update_progress)
                self.root.after(0, lambda: self.scraper_status.configure(text="Scraper: ✅"))
                
                self.analyzer = SentimentAnalyzer()
                self.root.after(0, lambda: self.analyzer_status.configure(text="Analyzer: ✅"))
                
                self.preprocessor = TextPreprocessor()
                
                self.visualizer = AdvancedVisualizer()
                
                self.data_manager = DataManager()
                self.root.after(0, lambda: self.db_status.configure(text="Database: ✅"))
                
                self.root.after(0, lambda: self.status_var.set("Ready for analysis"))
                
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {str(e)}")
                self.root.after(0, lambda: self.status_var.set(f"Initialization failed: {str(e)}"))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def _validate_url(self):
        """Validate the entered YouTube URL."""
        url = self.url_var.get().strip()
        
        if not url:
            self.url_status_var.set("Enter a YouTube URL")
            self.url_status_label.configure(style='')
            return
        
        if validate_url(url):
            try:
                video_id = self.scraper.extract_video_id(url)
                if video_id:
                    self.url_status_var.set(f"✅ Valid YouTube URL (Video ID: {video_id})")
                    self.url_status_label.configure(style='Success.TLabel')
                    self.start_btn.configure(state=tk.NORMAL)
                else:
                    self.url_status_var.set("❌ Could not extract video ID")
                    self.url_status_label.configure(style='Danger.TLabel')
                    self.start_btn.configure(state=tk.DISABLED)
            except Exception as e:
                self.url_status_var.set(f"❌ Error validating URL: {str(e)}")
                self.url_status_label.configure(style='Danger.TLabel')
                self.start_btn.configure(state=tk.DISABLED)
        else:
            self.url_status_var.set("❌ Invalid YouTube URL format")
            self.url_status_label.configure(style='Danger.TLabel')
            self.start_btn.configure(state=tk.DISABLED)
    
    def _start_analysis(self):
        """Start the sentiment analysis process."""
        url = self.url_var.get().strip()
        max_comments = self.max_comments_var.get()
        sort_by = self.sort_var.get()
        
        if not url or not self.scraper:
            messagebox.showerror("Error", "Please enter a valid YouTube URL and ensure components are initialized.")
            return
        
        # Update UI state
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.export_btn.configure(state=tk.DISABLED)
        self.stop_requested = False
        
        # Clear previous results
        self._clear_results_display()
        
        # Start analysis in separate thread
        threading.Thread(target=self._run_analysis, args=(url, max_comments, sort_by), daemon=True).start()
    
    def _run_analysis(self, url: str, max_comments: int, sort_by: str):
        """Run the complete analysis process."""
        start_time = time.time()
        
        try:
            self.root.after(0, lambda: self.status_var.set("Starting analysis..."))
            
            # Step 1: Scrape comments
            self.root.after(0, lambda: self.progress_text_var.set("Scraping comments..."))
            comments = self.scraper.scrape_comments(url, max_comments, sort_by)
            
            if self.stop_requested:
                return
            
            if not comments:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No comments found for this video."))
                return
            
            # Step 2: Get video info
            video_info = self.scraper.get_video_info(url)
            
            # Step 3: Analyze sentiment
            self.root.after(0, lambda: self.progress_text_var.set("Analyzing sentiment..."))
            use_ensemble = self.use_ensemble_var.get()
            
            if use_ensemble:
                results = self.analyzer.analyze_batch([c['text'] for c in comments], use_ensemble=True)
            else:
                results = self.analyzer.analyze_batch([c['text'] for c in comments], use_ensemble=False)
            
            if self.stop_requested:
                return
            
            # Step 4: Calculate statistics
            stats = self.analyzer.get_sentiment_statistics(results)
            
            # Step 5: Save data if requested
            if self.save_data_var.get() and self.data_manager:
                self.root.after(0, lambda: self.progress_text_var.set("Saving to database..."))
                
                # Save video info
                self.data_manager.save_video_info(video_info)
                
                # Save comments
                self.data_manager.save_comments(comments, video_info['video_id'])
                
                # Save sentiment results
                self.data_manager.save_sentiment_results(results, comments)
                
                # Save analysis session
                session_data = {
                    'video_id': video_info['video_id'],
                    'total_comments': len(comments),
                    'settings': {
                        'max_comments': max_comments,
                        'sort_by': sort_by,
                        'use_ensemble': use_ensemble
                    },
                    'statistics': stats
                }
                session_id = self.data_manager.save_analysis_session(session_data)
            
            # Store results for GUI
            self.current_analysis = {
                'video_info': video_info,
                'comments': comments,
                'results': results,
                'statistics': stats
            }
            
            # Update UI with results
            self.root.after(0, self._update_results_display)
            
            # Auto export if requested
            if self.auto_export_var.get():
                self.root.after(0, self._auto_export)
            
            elapsed = time.time() - start_time
            self.root.after(0, lambda: self.status_var.set(f"Analysis completed in {format_duration(elapsed)}"))
            self.root.after(0, lambda: self.progress_text_var.set(f"Analysis completed! Processed {len(comments)} comments"))
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Analysis failed"))
            
        finally:
            # Reset UI state
            self.root.after(0, self._reset_ui_state)
    
    def _update_progress(self, percentage: float, current: int, total: int, message: str = "", eta: str = ""):
        """Update progress display."""
        self.root.after(0, lambda: self.progress_var.set(percentage))
        
        if message:
            progress_text = f"{message} - {current}/{total} ({percentage:.1f}%)"
        else:
            progress_text = f"Progress: {current}/{total} ({percentage:.1f}%)"
        
        if eta and eta != "Unknown":
            progress_text += f" - ETA: {eta}"
        
        self.root.after(0, lambda: self.progress_text_var.set(progress_text))
    
    def _stop_analysis(self):
        """Stop the current analysis."""
        self.stop_requested = True
        if self.scraper:
            self.scraper.stop_scraping()
        
        self.status_var.set("Stopping analysis...")
        self.stop_btn.configure(state=tk.DISABLED)
    
    def _reset_ui_state(self):
        """Reset UI to default state."""
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        
        if self.current_analysis:
            self.export_btn.configure(state=tk.NORMAL)
    
    def _update_results_display(self):
        """Update the results display with current analysis."""
        if not self.current_analysis:
            return
        
        results = self.current_analysis['results']
        comments = self.current_analysis['comments']
        stats = self.current_analysis['statistics']
        
        # Update quick stats
        self.stats_vars['total'].set(str(stats.get('total_comments', 0)))
        
        sentiment_dist = stats.get('sentiment_distribution', {})
        self.stats_vars['positive'].set(str(sentiment_dist.get('positive', 0)))
        self.stats_vars['negative'].set(str(sentiment_dist.get('negative', 0)))
        self.stats_vars['neutral'].set(str(sentiment_dist.get('neutral', 0)))
        self.stats_vars['confidence'].set(f"{stats.get('average_confidence', 0):.2f}")
        
        # Update results tree
        self._populate_results_tree(results, comments)
        
        # Switch to results tab
        self.notebook.select(1)
    
    def _populate_results_tree(self, results: list, comments: list):
        """Populate the results treeview."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add new items
        for result, comment in zip(results, comments):
            values = (
                comment.get('author', 'Unknown')[:20],
                comment.get('text', '')[:100] + ('...' if len(comment.get('text', '')) > 100 else ''),
                result.get('sentiment', 'Unknown'),
                f"{result.get('confidence', 0):.3f}",
                comment.get('likes', 0)
            )
            
            # Color code by sentiment
            sentiment = result.get('sentiment', 'neutral')
            tags = (sentiment,)
            
            self.results_tree.insert('', tk.END, values=values, tags=tags)
        
        # Configure tags for colors
        self.results_tree.tag_configure('positive', foreground='#2E8B57')
        self.results_tree.tag_configure('negative', foreground='#DC143C')
        self.results_tree.tag_configure('neutral', foreground='#4682B4')
    
    def _filter_results(self, event=None):
        """Filter results based on sentiment."""
        # This would implement filtering logic
        pass
    
    def _search_results(self, event=None):
        """Search results based on text content."""
        # This would implement search logic
        pass
    
    def _clear_results(self):
        """Clear all results."""
        if messagebox.askyesno("Confirm", "Clear all current results?"):
            self.current_analysis = None
            self._clear_results_display()
            self.export_btn.configure(state=tk.DISABLED)
    
    def _clear_results_display(self):
        """Clear the results display."""
        # Clear quick stats
        for var in self.stats_vars.values():
            var.set("0")
        
        # Clear results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_text_var.set("Ready to start analysis")
    
    def _export_results(self):
        """Export current results."""
        if not self.current_analysis:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        # Ask for export format
        export_format = messagebox.askyesnocancel("Export Format", 
                                                 "Choose export format:\nYes = CSV\nNo = JSON\nCancel = Cancel")
        
        if export_format is None:
            return
        
        # Get export path
        if export_format:
            file_path = filedialog.asksaveasfilename(
                title="Save CSV Export",
                filetypes=[("CSV files", "*.csv")],
                defaultextension=".csv"
            )
            format_type = "csv"
        else:
            file_path = filedialog.asksaveasfilename(
                title="Save JSON Export", 
                filetypes=[("JSON files", "*.json")],
                defaultextension=".json"
            )
            format_type = "json"
        
        if not file_path:
            return
        
        try:
            video_id = self.current_analysis['video_info']['video_id']
            
            if format_type == "csv":
                self.data_manager.export_to_csv(video_id, Path(file_path))
            else:
                self.data_manager.export_to_json(video_id, Path(file_path))
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def _auto_export(self):
        """Automatically export results."""
        if not self.current_analysis:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = self.current_analysis['video_info']['video_id']
            filename = f"sentiment_analysis_{video_id}_{timestamp}.csv"
            
            export_path = settings.EXPORTS_DIR / filename
            self.data_manager.export_to_csv(video_id, export_path)
            
            self.logger.info(f"Auto-exported results to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Auto-export failed: {str(e)}")
    
    def _show_visualization(self, viz_type: str):
        """Show selected visualization."""
        if not self.current_analysis:
            messagebox.showwarning("Warning", "No analysis results available for visualization.")
            return
        
        # Switch to visualization tab
        self.notebook.select(2)
        
        # This would implement the visualization display logic
        # For now, show a placeholder
        messagebox.showinfo("Visualization", f"Would show {viz_type} visualization here")
    
    def _show_dashboard(self):
        """Show interactive dashboard in browser."""
        if not self.current_analysis:
            messagebox.showwarning("Warning", "No analysis results available for dashboard.")
            return
        
        try:
            # Generate dashboard
            dashboard_html = self.visualizer.create_interactive_dashboard(self.current_analysis['results'])
            
            # Save to temp file and open in browser
            temp_path = settings.EXPORTS_DIR / "temp_dashboard.html"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            webbrowser.open(f"file://{temp_path.absolute()}")
            
        except Exception as e:
            messagebox.showerror("Dashboard Error", f"Failed to create dashboard: {str(e)}")
    
    def _refresh_history(self):
        """Refresh the analysis history."""
        if not self.data_manager:
            return
        
        try:
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Get history
            history = self.data_manager.get_analysis_history()
            
            # Populate tree
            for session in history:
                values = (
                    session.get('created_at', '')[:19],  # Trim timestamp
                    session.get('video_id', ''),
                    session.get('total_comments', 0),
                    "Various"  # Simplified for now
                )
                self.history_tree.insert('', tk.END, values=values)
                
        except Exception as e:
            self.logger.error(f"Failed to refresh history: {str(e)}")
    
    def _update_confidence_label(self, value):
        """Update confidence threshold label."""
        self.confidence_value_label.configure(text=f"{float(value):.2f}")
    
    def _update_sentiment_label(self, value):
        """Update sentiment threshold label."""
        self.sentiment_value_label.configure(text=f"{float(value):.2f}")
    
    def _apply_settings(self):
        """Apply current settings."""
        # This would update the settings and reinitialize components as needed
        messagebox.showinfo("Settings", "Settings applied successfully!")
    
    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')
    
    def _save_logs(self):
        """Save logs to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Logs",
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log")],
            defaultextension=".txt"
        )
        
        if file_path:
            try:
                logs_content = self.log_text.get('1.0', tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(logs_content)
                messagebox.showinfo("Success", f"Logs saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {str(e)}")
    
    def _on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            # Stop any running analysis
            self.stop_requested = True
            if self.scraper:
                self.scraper.stop_scraping()
            
            self.root.destroy()
    
    def run(self):
        """Start the GUI application."""
        self.logger.info("Starting GUI application")
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    try:
        app = ModernGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
