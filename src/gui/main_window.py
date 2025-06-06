import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import asyncio
import logging
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import webbrowser

from ..core.comment_scraper import YouTubeCommentScraper
from ..core.sentiment_analyzer import SentimentAnalyzer
from ..core.data_processor import DataProcessor
from ..utils.validators import validate_youtube_url
from ..config.settings import get_settings
from .dashboard import Dashboard

class SentimentAnalysisGUI:
    """Main GUI application for YouTube Comment Sentiment Analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scraper = None
        self.analyzer = None
        self.processor = None
        self.current_data = None
        
        # GUI components
        self.root = None
        self.dashboard = None
        
        # Progress tracking
        self.progress_var = None
        self.status_var = None
        self.is_running = False
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Initialize the GUI"""
        self.root = tk.Tk()
        self.root.title(f"{self.settings.APP_NAME} v{self.settings.APP_VERSION}")
        self.root.geometry(f"{self.settings.WINDOW_WIDTH}x{self.settings.WINDOW_HEIGHT}")
        
        # Configure style
        self._setup_style()
        
        # Create main layout
        self._create_main_layout()
        
        # Initialize components in background
        self._init_components_async()
    
    def _setup_style(self):
        """Setup GUI styling"""
        style = ttk.Style()
        
        # Configure colors and fonts
        if self.settings.THEME == "dark":
            style.theme_use("clam")
            # Add dark theme customizations here
        else:
            style.theme_use("clam")
        
        # Custom styles
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Heading.TLabel", font=("Arial", 12, "bold"))
        style.configure("Status.TLabel", font=("Arial", 10))
    
    def _create_main_layout(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="YouTube Comment Sentiment Analysis",
            style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input section
        self._create_input_section(main_frame)
        
        # Progress section
        self._create_progress_section(main_frame)
        
        # Results section
        self._create_results_section(main_frame)
        
        # Control buttons
        self._create_control_buttons(main_frame)
    
    def _create_input_section(self, parent):
        """Create input section"""
        # Input frame
        input_frame = ttk.LabelFrame(parent, text="Input Settings", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # URL input
        ttk.Label(input_frame, text="YouTube URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(input_frame, textvariable=self.url_var, width=60)
        url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Validate button
        validate_btn = ttk.Button(input_frame, text="Validate URL", command=self._validate_url)
        validate_btn.grid(row=0, column=2)
        
        # Number of comments
        ttk.Label(input_frame, text="Number of Comments:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.limit_var = tk.IntVar(value=self.settings.DEFAULT_COMMENT_LIMIT)
        limit_spinbox = ttk.Spinbox(
            input_frame, 
            from_=10, 
            to=10000, 
            textvariable=self.limit_var,
            width=10
        )
        limit_spinbox.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
          # Sort options
        ttk.Label(input_frame, text="Sort by:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.sort_var = tk.StringVar(value="Top Liked")
        sort_combo = ttk.Combobox(
            input_frame,
            textvariable=self.sort_var,
            values=["Top Liked", "Most Recent", "Oldest"],
            state="readonly",
            width=12
        )
        sort_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Include replies checkbox
        self.include_replies_var = tk.BooleanVar(value=False)
        replies_check = ttk.Checkbutton(
            input_frame,
            text="Include replies",
            variable=self.include_replies_var
        )
        replies_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    def _create_progress_section(self, parent):
        """Create progress tracking section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, style="Status.TLabel")
        status_label.grid(row=1, column=0, sticky=tk.W)
    
    def _create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="Results Summary", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(
            text_frame,
            height=10,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def _create_control_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
        # Start Analysis button
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Analysis",
            command=self._start_analysis,
            style="Accent.TButton"
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop",
            command=self._stop_analysis,
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Dashboard button
        self.dashboard_btn = ttk.Button(
            button_frame,
            text="Open Dashboard",
            command=self._open_dashboard,
            state="disabled"
        )
        self.dashboard_btn.grid(row=0, column=2, padx=(0, 10))
        
        # Export button
        self.export_btn = ttk.Button(
            button_frame,
            text="Export Data",
            command=self._export_data,
            state="disabled"
        )
        self.export_btn.grid(row=0, column=3, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="Clear Results",
            command=self._clear_results
        )
        clear_btn.grid(row=0, column=4)
    
    def _init_components_async(self):
        """Initialize components in background thread"""
        def init_thread():
            try:
                self.status_var.set("Initializing components...")
                
                # Initialize scraper
                self.scraper = YouTubeCommentScraper(progress_callback=self._update_progress)
                
                # Initialize analyzer
                self.analyzer = SentimentAnalyzer()
                
                # Initialize processor
                self.processor = DataProcessor()
                
                self.status_var.set("Ready")
                self.logger.info("Components initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {e}")
                self.status_var.set(f"Initialization failed: {e}")
        
        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
    
    def _validate_url(self):
        """Validate the YouTube URL"""
        url = self.url_var.get().strip()
        
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        if validate_youtube_url(url):
            messagebox.showinfo("Success", "Valid YouTube URL!")
        else:
            messagebox.showerror("Error", "Invalid YouTube URL format")
    
    def _update_progress(self, progress: float, message: str = ""):
        """Update progress bar and status"""
        self.progress_var.set(progress * 100)
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()
    
    def _start_analysis(self):
        """Start the sentiment analysis process"""
        # Validate inputs
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
        
        if not validate_youtube_url(url):
            messagebox.showerror("Error", "Invalid YouTube URL format")
            return
        
        if not all([self.scraper, self.analyzer, self.processor]):
            messagebox.showerror("Error", "Components not initialized. Please wait...")
            return
        
        # Disable controls
        self._set_controls_state(False)
        self.is_running = True
          # Start analysis in background thread
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
    
    def _run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Get parameters
            url = self.url_var.get().strip()
            limit = self.limit_var.get()
            include_replies = self.include_replies_var.get()
            sort_by = self.sort_var.get()
              # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Scrape comments
                self.status_var.set("Scraping comments...")
                
                comments = loop.run_until_complete(
                    self.scraper.scrape_comments(
                        url=url,
                        limit=limit,
                        include_replies=include_replies,
                        sort_by=sort_by
                    )
                )
                
                if not comments:
                    self._show_error("No comments found or video is not accessible")
                    return
                
                # Process comments with sentiment analysis
                self.status_var.set("Analyzing sentiment...")
                video_id = self.scraper.get_video_info(url)['video_id']
                
                self.current_data = loop.run_until_complete(
                    self.processor.process_comments(
                        comments=comments,
                        video_id=video_id,
                        progress_callback=self._update_progress
                    )
                )
                
                # Generate summary
                summary = self.processor.get_sentiment_summary(self.current_data)
                
                # Update UI
                self._display_results(summary)
                self._enable_export_buttons()
                
                self.status_var.set(f"Analysis complete! Processed {len(comments)} comments")
                
            finally:
                loop.close()
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self._show_error(f"Analysis failed: {str(e)}")
        
        finally:
            self.is_running = False
            self._set_controls_state(True)
    
    def _stop_analysis(self):
        """Stop the current analysis"""
        self.is_running = False
        self.status_var.set("Stopping analysis...")
        # Note: Actual stopping implementation would need to be added to the async methods
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        if not summary:
            self.results_text.insert(tk.END, "No results to display.")
            return
        
        # Format and display summary
        result_text = f"""SENTIMENT ANALYSIS RESULTS
{'='*50}

Total Comments Analyzed: {summary.get('total_comments', 0)}

SENTIMENT DISTRIBUTION:
• Positive: {summary.get('sentiment_distribution', {}).get('positive', 0)} ({summary.get('sentiment_percentages', {}).get('positive', 0):.1f}%)
• Negative: {summary.get('sentiment_distribution', {}).get('negative', 0)} ({summary.get('sentiment_percentages', {}).get('negative', 0):.1f}%)
• Neutral: {summary.get('sentiment_distribution', {}).get('neutral', 0)} ({summary.get('sentiment_percentages', {}).get('neutral', 0):.1f}%)

Average Confidence: {summary.get('average_confidence', 0):.3f}

TOP POSITIVE COMMENTS:
{self._format_comments(summary.get('top_positive_comments', []))}

TOP NEGATIVE COMMENTS:
{self._format_comments(summary.get('top_negative_comments', []))}

MOST LIKED COMMENTS:
{self._format_liked_comments(summary.get('most_liked_comments', []))}
"""
        
        self.results_text.insert(tk.END, result_text)
    
    def _format_comments(self, comments: list) -> str:
        """Format comments for display"""
        if not comments:
            return "None found.\n"
        
        formatted = ""
        for i, comment in enumerate(comments[:5], 1):
            text = comment.get('text', '')[:100] + "..." if len(comment.get('text', '')) > 100 else comment.get('text', '')
            confidence = comment.get('sentiment_confidence', 0)
            likes = comment.get('likes', 0)
            formatted += f"{i}. {text} (Confidence: {confidence:.3f}, Likes: {likes})\n\n"
        
        return formatted
    
    def _format_liked_comments(self, comments: list) -> str:
        """Format most liked comments for display"""
        if not comments:
            return "None found.\n"
        
        formatted = ""
        for i, comment in enumerate(comments[:5], 1):
            text = comment.get('text', '')[:100] + "..." if len(comment.get('text', '')) > 100 else comment.get('text', '')
            sentiment = comment.get('sentiment', 'unknown')
            likes = comment.get('likes', 0)
            formatted += f"{i}. [{sentiment.upper()}] {text} ({likes} likes)\n\n"
        
        return formatted
    
    def _open_dashboard(self):
        """Open the interactive dashboard"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("Warning", "No data available. Please run analysis first.")
            return
        
        try:
            if not self.dashboard:
                self.dashboard = Dashboard(self.current_data)
            else:
                self.dashboard.update_data(self.current_data)
            
            self.dashboard.show()
        except Exception as e:
            self.logger.error(f"Failed to open dashboard: {e}")
            messagebox.showerror("Error", f"Failed to open dashboard: {str(e)}")
    
    def _export_data(self):
        """Export current data"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("Warning", "No data to export. Please run analysis first.")
            return
        
        # Choose export format
        format_options = ["CSV", "JSON", "Excel"]
        format_dialog = tk.Toplevel(self.root)
        format_dialog.title("Export Format")
        format_dialog.geometry("300x150")
        format_dialog.transient(self.root)
        format_dialog.grab_set()
        
        selected_format = tk.StringVar(value="CSV")
        
        ttk.Label(format_dialog, text="Select export format:").pack(pady=10)
        
        for fmt in format_options:
            ttk.Radiobutton(
                format_dialog, 
                text=fmt, 
                variable=selected_format, 
                value=fmt
            ).pack(anchor=tk.W, padx=20)
        
        def do_export():
            try:
                format_map = {"CSV": "csv", "JSON": "json", "Excel": "xlsx"}
                export_format = format_map[selected_format.get()]
                
                filepath = self.processor.export_data(self.current_data, export_format)
                messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
                format_dialog.destroy()
                
            except Exception as e:
                self.logger.error(f"Export failed: {e}")
                messagebox.showerror("Error", f"Export failed: {str(e)}")
        
        ttk.Button(format_dialog, text="Export", command=do_export).pack(pady=10)
        ttk.Button(format_dialog, text="Cancel", command=format_dialog.destroy).pack()
    
    def _clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.current_data = None
        self.progress_var.set(0)
        self.status_var.set("Ready")
        self._disable_export_buttons()
    
    def _set_controls_state(self, enabled: bool):
        """Enable/disable control buttons"""
        state = "normal" if enabled else "disabled"
        
        self.start_btn.config(state=state)
        self.stop_btn.config(state="disabled" if enabled else "normal")
    
    def _enable_export_buttons(self):
        """Enable export-related buttons"""
        self.dashboard_btn.config(state="normal")
        self.export_btn.config(state="normal")
    
    def _disable_export_buttons(self):
        """Disable export-related buttons"""
        self.dashboard_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
    
    def _show_error(self, message: str):
        """Show error message in a thread-safe way"""
        self.root.after(0, lambda: messagebox.showerror("Error", message))
    
    def run(self):
        """Start the GUI application"""
        try:
            self.logger.info("Starting GUI application")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI application failed: {e}")
            raise
