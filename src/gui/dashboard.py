import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import pandas as pd
import numpy as np
import squarify
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import webbrowser
import tempfile
import os
from typing import Dict, Any, Optional
import logging

class Dashboard:
    """Interactive dashboard for visualizing sentiment analysis results"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.window = None
        self.notebook = None
        
    def show(self):
        """Show the dashboard window"""
        if self.window is None:
            self._create_window()
        else:
            self.window.lift()
            self.window.focus_force()
    
    def _create_window(self):
        """Create the dashboard window"""
        self.window = tk.Toplevel()
        self.window.title("Sentiment Analysis Dashboard")
        self.window.geometry("1400x900")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_overview_tab()
        self._create_charts_tab()
        self._create_detailed_tab()
        self._create_interactive_tab()
    
    def _create_overview_tab(self):
        """Create overview statistics tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        
        # Create main container with scrollbar
        canvas = tk.Canvas(overview_frame)
        scrollbar = ttk.Scrollbar(overview_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Statistics grid
        self._create_stats_grid(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_stats_grid(self, parent):
        """Create statistics grid"""
        # Title
        title_label = ttk.Label(parent, text="Sentiment Analysis Overview", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20), sticky="w")
        
        # Basic statistics
        stats = self._calculate_basic_stats()
        
        row = 1
        for category, values in stats.items():
            # Category header
            cat_label = ttk.Label(parent, text=category, font=("Arial", 12, "bold"))
            cat_label.grid(row=row, column=0, columnspan=4, sticky="w", pady=(10, 5))
            row += 1
            
            # Values
            col = 0
            for key, value in values.items():
                if col >= 4:
                    col = 0
                    row += 1
                
                # Create stat card
                card_frame = ttk.LabelFrame(parent, text=key, padding="10")
                card_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
                
                value_label = ttk.Label(card_frame, text=str(value), font=("Arial", 14, "bold"))
                value_label.pack()
                
                col += 1
            
            row += 1 if col == 0 else 2
    
    def _calculate_basic_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate basic statistics"""
        if self.data.empty:
            return {}
        
        sentiment_counts = self.data['sentiment'].value_counts()
        total_comments = len(self.data)
        
        stats = {
            "General Statistics": {
                "Total Comments": f"{total_comments:,}",
                "Unique Authors": f"{self.data['author'].nunique():,}",
                "Average Likes": f"{self.data['likes'].mean():.1f}",
                "Total Likes": f"{self.data['likes'].sum():,}"
            },
            "Sentiment Distribution": {
                "Positive": f"{sentiment_counts.get('positive', 0):,} ({sentiment_counts.get('positive', 0)/total_comments*100:.1f}%)",
                "Negative": f"{sentiment_counts.get('negative', 0):,} ({sentiment_counts.get('negative', 0)/total_comments*100:.1f}%)",
                "Neutral": f"{sentiment_counts.get('neutral', 0):,} ({sentiment_counts.get('neutral', 0)/total_comments*100:.1f}%)",
                "Avg Confidence": f"{self.data['sentiment_confidence'].mean():.3f}"
            },
            "Content Analysis": {
                "Avg Comment Length": f"{self.data['text'].str.len().mean():.0f} chars",
                "Most Active Author": self.data['author'].value_counts().index[0] if not self.data.empty else "N/A",
                "Replies Count": f"{self.data['reply_count'].sum():,}",
                "Top Sentiment Score": f"{self.data['sentiment_confidence'].max():.3f}"
            }
        }
        
        return stats
    
    def _create_charts_tab(self):
        """Create charts visualization tab"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="Charts")
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(16, 12))
        
        # Sentiment distribution pie chart
        ax1 = plt.subplot(2, 3, 1)
        self._create_sentiment_pie_chart(ax1)
        
        # Sentiment distribution bar chart
        ax2 = plt.subplot(2, 3, 2)
        self._create_sentiment_bar_chart(ax2)
        
        # Confidence distribution histogram
        ax3 = plt.subplot(2, 3, 3)
        self._create_confidence_histogram(ax3)
        
        # Likes vs Sentiment scatter plot
        ax4 = plt.subplot(2, 3, 4)
        self._create_likes_sentiment_scatter(ax4)
        
        # Top authors bar chart
        ax5 = plt.subplot(2, 3, 5)
        self._create_top_authors_chart(ax5)
        
        # Sentiment over time (if time data available)
        ax6 = plt.subplot(2, 3, 6)
        self._create_sentiment_treemap(ax6)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, charts_frame)
        toolbar.update()
    
    def _create_sentiment_pie_chart(self, ax):
        """Create sentiment distribution pie chart"""
        sentiment_counts = self.data['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _create_sentiment_bar_chart(self, ax):
        """Create sentiment distribution bar chart"""
        sentiment_counts = self.data['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        
        ax.set_title('Sentiment Counts', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Comments')
        ax.set_xlabel('Sentiment')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
    
    def _create_confidence_histogram(self, ax):
        """Create confidence score histogram"""
        ax.hist(self.data['sentiment_confidence'], bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.axvline(self.data['sentiment_confidence'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.data["sentiment_confidence"].mean():.3f}')
        ax.legend()
    
    def _create_likes_sentiment_scatter(self, ax):
        """Create likes vs sentiment scatter plot"""
        sentiment_colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        
        for sentiment in sentiment_colors:
            data_subset = self.data[self.data['sentiment'] == sentiment]
            ax.scatter(
                data_subset['likes'], 
                data_subset['sentiment_confidence'],
                c=sentiment_colors[sentiment],
                label=sentiment.capitalize(),
                alpha=0.6
            )
        
        ax.set_title('Likes vs Confidence by Sentiment', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Likes')
        ax.set_ylabel('Confidence Score')
        ax.legend()
        ax.set_xscale('log')  # Log scale for likes if there's a wide range
    
    def _create_top_authors_chart(self, ax):
        """Create top authors chart"""
        top_authors = self.data['author'].value_counts().head(10)
        
        bars = ax.barh(range(len(top_authors)), top_authors.values, color='#9b59b6')
        ax.set_yticks(range(len(top_authors)))
        ax.set_yticklabels([f"{author[:20]}..." if len(author) > 20 else author 
                           for author in top_authors.index])
        
        ax.set_title('Top 10 Most Active Authors', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Comments')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}',
                   ha='left', va='center', fontweight='bold')
    
    def _create_sentiment_treemap(self, ax):
        """Create sentiment treemap using squarify"""
        sentiment_counts = self.data['sentiment'].value_counts()
        
        colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(sentiment_counts)]
        
        squarify.plot(
            sizes=sentiment_counts.values,
            label=[f"{idx}\\n{val}" for idx, val in sentiment_counts.items()],
            color=colors,
            ax=ax,
            text_kwargs={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        ax.set_title('Sentiment Distribution (Treemap)', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _create_detailed_tab(self):
        """Create detailed data view tab"""
        detailed_frame = ttk.Frame(self.notebook)
        self.notebook.add(detailed_frame, text="Detailed Data")
        
        # Create treeview for data display
        columns = ['text', 'author', 'sentiment', 'confidence', 'likes']
        tree = ttk.Treeview(detailed_frame, columns=columns, show='headings', height=20)
        
        # Define headings
        tree.heading('text', text='Comment Text')
        tree.heading('author', text='Author')
        tree.heading('sentiment', text='Sentiment')
        tree.heading('confidence', text='Confidence')
        tree.heading('likes', text='Likes')
        
        # Configure column widths
        tree.column('text', width=400)
        tree.column('author', width=150)
        tree.column('sentiment', width=100)
        tree.column('confidence', width=100)
        tree.column('likes', width=80)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(detailed_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(detailed_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Populate data (first 1000 rows for performance)
        display_data = self.data.head(1000)
        for idx, row in display_data.iterrows():
            text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            tree.insert('', tk.END, values=(
                text,
                row['author'],
                row['sentiment'],
                f"{row['sentiment_confidence']:.3f}",
                row['likes']
            ))
        
        # Pack elements
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        detailed_frame.grid_rowconfigure(0, weight=1)
        detailed_frame.grid_columnconfigure(0, weight=1)
        
        # Add info label
        info_label = ttk.Label(
            detailed_frame, 
            text=f"Showing first 1000 of {len(self.data)} comments"
        )
        info_label.grid(row=2, column=0, columnspan=2, pady=5)
    
    def _create_interactive_tab(self):
        """Create interactive Plotly visualizations tab"""
        interactive_frame = ttk.Frame(self.notebook)
        self.notebook.add(interactive_frame, text="Interactive")
        
        # Create buttons for different interactive charts
        button_frame = ttk.Frame(interactive_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Sentiment Distribution (Interactive)",
            command=self._show_interactive_pie
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Sentiment Timeline",
            command=self._show_sentiment_timeline
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="3D Sentiment Analysis",
            command=self._show_3d_analysis
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Word Cloud Dashboard",
            command=self._show_wordcloud_dashboard
        ).pack(side=tk.LEFT, padx=5)
        
        # Info text
        info_text = tk.Text(interactive_frame, height=20, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_content = """
Interactive Visualizations:

1. Sentiment Distribution (Interactive)
   - Interactive pie chart with hover details
   - Click to filter data by sentiment

2. Sentiment Timeline
   - Shows sentiment distribution over time
   - Hover for detailed information

3. 3D Sentiment Analysis
   - 3D scatter plot of likes, confidence, and sentiment
   - Rotate and zoom for different perspectives

4. Word Cloud Dashboard
   - Interactive word clouds for each sentiment
   - Shows most common words in positive/negative comments

Click the buttons above to open interactive visualizations in your web browser.
These charts are powered by Plotly and provide rich interactivity including:
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Data filtering and selection
- Animation effects
        """
        
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
    
    def _show_interactive_pie(self):
        """Show interactive pie chart"""
        sentiment_counts = self.data['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Interactive Sentiment Distribution",
            color_discrete_sequence=['#2ecc71', '#e74c3c', '#95a5a6']
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        self._show_plotly_chart(fig)
    
    def _show_sentiment_timeline(self):
        """Show sentiment timeline"""
        if 'time_parsed' not in self.data.columns:
            self._show_error("Time data not available")
            return
        
        # Convert time_parsed to datetime if it's string
        time_data = pd.to_datetime(self.data['time_parsed'])
        timeline_data = self.data.copy()
        timeline_data['date'] = time_data.dt.date
        
        # Group by date and sentiment
        daily_sentiment = timeline_data.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(
            daily_sentiment,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Distribution Over Time',
            color_discrete_sequence=['#2ecc71', '#e74c3c', '#95a5a6']
        )
        
        self._show_plotly_chart(fig)
    
    def _show_3d_analysis(self):
        """Show 3D sentiment analysis"""
        fig = px.scatter_3d(
            self.data,
            x='likes',
            y='sentiment_confidence',
            z='reply_count',
            color='sentiment',
            title='3D Sentiment Analysis: Likes vs Confidence vs Replies',
            color_discrete_sequence=['#2ecc71', '#e74c3c', '#95a5a6'],
            hover_data=['author', 'text']
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Number of Likes',
                yaxis_title='Confidence Score',
                zaxis_title='Reply Count'
            )
        )
        
        self._show_plotly_chart(fig)
    
    def _show_wordcloud_dashboard(self):
        """Show word cloud dashboard (placeholder)"""
        # This would require additional libraries like wordcloud
        # For now, show a message
        tk.messagebox.showinfo(
            "Word Cloud Dashboard",
            "Word Cloud functionality requires additional setup.\n"
            "This feature would show:\n"
            "- Most common words in positive comments\n"
            "- Most common words in negative comments\n"
            "- Interactive filtering and selection"
        )
    
    def _show_plotly_chart(self, fig):
        """Show Plotly chart in web browser"""
        try:
            # Create temporary HTML file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            
            # Generate HTML
            html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            
            # Write to temp file
            with open(temp_file.name, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Interactive Chart</title>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """)
            
            # Open in browser
            webbrowser.open(f'file://{temp_file.name}')
            
        except Exception as e:
            self.logger.error(f"Failed to show interactive chart: {e}")
            tk.messagebox.showerror("Error", f"Failed to open interactive chart: {str(e)}")
    
    def _show_error(self, message: str):
        """Show error message"""
        tk.messagebox.showerror("Error", message)
    
    def update_data(self, new_data: pd.DataFrame):
        """Update dashboard with new data"""
        self.data = new_data
        
        # Refresh the current view
        if self.window and self.window.winfo_exists():
            # Close and recreate window
            self.window.destroy()
            self.window = None
            self.show()
