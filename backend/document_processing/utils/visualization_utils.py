"""
Visualization Utilities for Document Processing

This module provides utilities for rendering and drawing identified figures, tables, and images
from the document processing pipeline.
"""

import json
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ContentVisualizer:
    """Utility class for visualizing extracted content (figures, tables, images)."""
    
    def __init__(self, output_dir: str = "output/visualizations"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_content_summary_chart(self, analysis_data: Dict[str, Any]) -> str:
        """Create a summary chart of all extracted content.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            Path to saved chart
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Content Extraction Summary', fontsize=16, fontweight='bold')
            
            # Pie chart of content types
            content_types = ['Tables', 'Figures', 'Text', 'Code']
            content_counts = [
                analysis_data.get('total_tables', 0),
                analysis_data.get('total_figures', 0),
                analysis_data.get('total_text', 0),
                analysis_data.get('total_code', 0)
            ]
            
            ax1.pie(content_counts, labels=content_types, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Content Type Distribution')
            
            # Bar chart of top documents by figures
            figures_by_doc = analysis_data.get('figures_by_document', {})
            if figures_by_doc:
                top_figure_docs = sorted(figures_by_doc.items(), key=lambda x: x[1], reverse=True)[:10]
                doc_names = [doc.split('_')[0] for doc, count in top_figure_docs]
                figure_counts = [count for doc, count in top_figure_docs]
                
                ax2.barh(range(len(doc_names)), figure_counts)
                ax2.set_yticks(range(len(doc_names)))
                ax2.set_yticklabels(doc_names, fontsize=8)
                ax2.set_xlabel('Number of Figures')
                ax2.set_title('Top Documents by Figure Count')
            
            # Bar chart of top documents by tables
            tables_by_doc = analysis_data.get('tables_by_document', {})
            if tables_by_doc:
                top_table_docs = sorted(tables_by_doc.items(), key=lambda x: x[1], reverse=True)[:10]
                doc_names = [doc.split('_')[0] for doc, count in top_table_docs]
                table_counts = [count for doc, count in top_table_docs]
                
                ax3.barh(range(len(doc_names)), table_counts, color='orange')
                ax3.set_yticks(range(len(doc_names)))
                ax3.set_yticklabels(doc_names, fontsize=8)
                ax3.set_xlabel('Number of Tables')
                ax3.set_title('Top Documents by Table Count')
            
            # Timeline of processing
            ax4.text(0.1, 0.5, f"Processing Summary:\n"
                                f"• Total Documents: {len(analysis_data.get('figures_by_document', {})) + len(analysis_data.get('tables_by_document', {}))}\n"
                                f"• Documents with Figures: {len(analysis_data.get('figures_by_document', {}))}\n"
                                f"• Documents with Tables: {len(analysis_data.get('tables_by_document', {}))}\n"
                                f"• Total Figures: {analysis_data.get('total_figures', 0)}\n"
                                f"• Total Tables: {analysis_data.get('total_tables', 0)}",
                    transform=ax4.transAxes, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            ax4.set_title('Processing Statistics')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save chart
            output_path = self.output_dir / "content_summary_chart.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating content summary chart: {e}")
            return ""
    
    def create_table_visualization(self, table_data: List[Dict[str, Any]]) -> str:
        """Create visualizations for extracted tables.
        
        Args:
            table_data: List of table chunks with metadata
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Table Content Analysis', fontsize=16, fontweight='bold')
            
            # Table 1: Sample table content
            if table_data:
                sample_table = table_data[0]
                content = sample_table.get('content_preview', 'No content')
                metadata = sample_table.get('metadata', {})
                
                # Create a text box for the sample table
                axes[0, 0].text(0.1, 0.5, f"Sample Table Content:\n\n{content}",
                               transform=axes[0, 0].transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                axes[0, 0].set_title('Sample Table')
                axes[0, 0].axis('off')
                
                # Table metadata analysis
                word_counts = [t.get('metadata', {}).get('word_count', 0) for t in table_data]
                complexity_scores = [t.get('metadata', {}).get('complexity_score', 0) for t in table_data]
                
                axes[0, 1].hist(word_counts, bins=10, alpha=0.7, color='skyblue')
                axes[0, 1].set_xlabel('Word Count')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Table Word Count Distribution')
                
                axes[1, 0].hist(complexity_scores, bins=10, alpha=0.7, color='lightcoral')
                axes[1, 0].set_xlabel('Complexity Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Table Complexity Distribution')
                
                # Table structure analysis
                table_rows = [t.get('metadata', {}).get('table_rows', 0) for t in table_data]
                has_headers = [t.get('metadata', {}).get('has_header', False) for t in table_data]
                
                header_count = sum(has_headers)
                no_header_count = len(has_headers) - header_count
                
                axes[1, 1].pie([header_count, no_header_count], 
                              labels=['With Headers', 'Without Headers'],
                              autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Table Header Analysis')
            
            plt.tight_layout()
            
            # Save visualization
            output_path = self.output_dir / "table_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating table visualization: {e}")
            return ""
    
    def create_figure_visualization(self, figure_data: List[Dict[str, Any]]) -> str:
        """Create visualizations for extracted figures.
        
        Args:
            figure_data: List of figure chunks with metadata
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Figure Content Analysis', fontsize=16, fontweight='bold')
            
            # Figure 1: Sample figure content
            if figure_data:
                sample_figure = figure_data[0]
                content = sample_figure.get('content_preview', 'No content')
                metadata = sample_figure.get('metadata', {})
                
                # Create a text box for the sample figure
                axes[0, 0].text(0.1, 0.5, f"Sample Figure Content:\n\n{content}",
                               transform=axes[0, 0].transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
                axes[0, 0].set_title('Sample Figure')
                axes[0, 0].axis('off')
                
                # Figure metadata analysis
                word_counts = [f.get('metadata', {}).get('word_count', 0) for f in figure_data]
                complexity_scores = [f.get('metadata', {}).get('complexity_score', 0) for f in figure_data]
                
                axes[0, 1].hist(word_counts, bins=15, alpha=0.7, color='lightgreen')
                axes[0, 1].set_xlabel('Word Count')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Figure Word Count Distribution')
                
                axes[1, 0].hist(complexity_scores, bins=15, alpha=0.7, color='lightblue')
                axes[1, 0].set_xlabel('Complexity Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Figure Complexity Distribution')
                
                # Figure type analysis
                figure_types = []
                for f in figure_data:
                    content = f.get('content_preview', '').lower()
                    if 'figure' in content:
                        figure_types.append('Figure Reference')
                    elif 'diagram' in content:
                        figure_types.append('Diagram')
                    elif 'chart' in content:
                        figure_types.append('Chart')
                    else:
                        figure_types.append('Other')
                
                type_counts = pd.Series(figure_types).value_counts()
                axes[1, 1].pie(type_counts.values, labels=type_counts.index,
                              autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Figure Type Distribution')
            
            plt.tight_layout()
            
            # Save visualization
            output_path = self.output_dir / "figure_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating figure visualization: {e}")
            return ""
    
    def create_interactive_dashboard(self, analysis_data: Dict[str, Any]) -> str:
        """Create an interactive Plotly dashboard.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            Path to saved HTML dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Content Distribution', 'Figures by Document', 
                              'Tables by Document', 'Processing Timeline'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Pie chart for content distribution
            content_types = ['Tables', 'Figures', 'Text', 'Code']
            content_counts = [
                analysis_data.get('total_tables', 0),
                analysis_data.get('total_figures', 0),
                analysis_data.get('total_text', 0),
                analysis_data.get('total_code', 0)
            ]
            
            fig.add_trace(
                go.Pie(labels=content_types, values=content_counts, name="Content Types"),
                row=1, col=1
            )
            
            # Bar chart for figures by document
            figures_by_doc = analysis_data.get('figures_by_document', {})
            if figures_by_doc:
                top_figure_docs = sorted(figures_by_doc.items(), key=lambda x: x[1], reverse=True)[:10]
                doc_names = [doc.split('_')[0] for doc, count in top_figure_docs]
                figure_counts = [count for doc, count in top_figure_docs]
                
                fig.add_trace(
                    go.Bar(x=figure_counts, y=doc_names, orientation='h', name="Figures"),
                    row=1, col=2
                )
            
            # Bar chart for tables by document
            tables_by_doc = analysis_data.get('tables_by_document', {})
            if tables_by_doc:
                top_table_docs = sorted(tables_by_doc.items(), key=lambda x: x[1], reverse=True)[:10]
                doc_names = [doc.split('_')[0] for doc, count in top_table_docs]
                table_counts = [count for doc, count in top_table_docs]
                
                fig.add_trace(
                    go.Bar(x=table_counts, y=doc_names, orientation='h', name="Tables"),
                    row=2, col=1
                )
            
            # Timeline scatter plot
            processing_times = []
            for i in range(len(figures_by_doc) + len(tables_by_doc)):
                processing_times.append(datetime.now().timestamp() + i * 3600)  # Simulated times
            
            fig.add_trace(
                go.Scatter(x=processing_times, y=list(range(len(processing_times))),
                          mode='lines+markers', name="Processing Timeline"),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Document Processing Dashboard",
                showlegend=True,
                height=800
            )
            
            # Save as HTML
            output_path = self.output_dir / "interactive_dashboard.html"
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return ""
    
    def create_content_heatmap(self, analysis_data: Dict[str, Any]) -> str:
        """Create a heatmap showing content distribution across documents.
        
        Args:
            analysis_data: Dictionary containing analysis results
            
        Returns:
            Path to saved heatmap
        """
        try:
            # Prepare data for heatmap
            figures_by_doc = analysis_data.get('figures_by_document', {})
            tables_by_doc = analysis_data.get('tables_by_document', {})
            
            # Get all unique documents
            all_docs = set(figures_by_doc.keys()) | set(tables_by_doc.keys())
            
            # Create data matrix
            data = []
            doc_names = []
            
            for doc in sorted(all_docs):
                doc_names.append(doc.split('_')[0])
                figures = figures_by_doc.get(doc, 0)
                tables = tables_by_doc.get(doc, 0)
                data.append([figures, tables])
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, max(8, len(data) * 0.3)))
            
            # Create DataFrame for seaborn
            df = pd.DataFrame(data, columns=['Figures', 'Tables'], index=doc_names)
            
            # Create heatmap
            sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title('Content Distribution Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Content Type')
            ax.set_ylabel('Documents')
            
            plt.tight_layout()
            
            # Save heatmap
            output_path = self.output_dir / "content_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating content heatmap: {e}")
            return ""
    
    def create_sample_content_cards(self, table_data: List[Dict], figure_data: List[Dict]) -> str:
        """Create sample content cards showing extracted content.
        
        Args:
            table_data: List of table chunks
            figure_data: List of figure chunks
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Sample Extracted Content', fontsize=16, fontweight='bold')
            
            # Sample table cards
            for i in range(min(3, len(table_data))):
                table = table_data[i]
                content = table.get('content_preview', 'No content')
                metadata = table.get('metadata', {})
                
                # Create card-like visualization
                ax = axes[0, i]
                card = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor="lightyellow", alpha=0.8)
                ax.add_patch(card)
                
                ax.text(0.5, 0.7, f"Table {i+1}", transform=ax.transAxes,
                       ha='center', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, content[:100] + "..." if len(content) > 100 else content,
                       transform=ax.transAxes, ha='center', fontsize=8, wrap=True)
                ax.text(0.5, 0.2, f"Words: {metadata.get('word_count', 0)} | "
                                  f"Complexity: {metadata.get('complexity_score', 0):.2f}",
                       transform=ax.transAxes, ha='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # Sample figure cards
            for i in range(min(3, len(figure_data))):
                figure = figure_data[i]
                content = figure.get('content_preview', 'No content')
                metadata = figure.get('metadata', {})
                
                # Create card-like visualization
                ax = axes[1, i]
                card = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor="lightgreen", alpha=0.8)
                ax.add_patch(card)
                
                ax.text(0.5, 0.7, f"Figure {i+1}", transform=ax.transAxes,
                       ha='center', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, content[:100] + "..." if len(content) > 100 else content,
                       transform=ax.transAxes, ha='center', fontsize=8, wrap=True)
                ax.text(0.5, 0.2, f"Words: {metadata.get('word_count', 0)} | "
                                  f"Complexity: {metadata.get('complexity_score', 0):.2f}",
                       transform=ax.transAxes, ha='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            output_path = self.output_dir / "sample_content_cards.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating sample content cards: {e}")
            return ""
    
    def generate_all_visualizations(self, analysis_file: str) -> Dict[str, str]:
        """Generate all visualizations from analysis data.
        
        Args:
            analysis_file: Path to analysis JSON file
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        try:
            # Load analysis data
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            results = {}
            
            # Generate all visualizations
            results['summary_chart'] = self.create_content_summary_chart(analysis_data)
            
            if 'sample_table_chunks' in analysis_data:
                results['table_visualization'] = self.create_table_visualization(
                    analysis_data['sample_table_chunks']
                )
            
            if 'sample_figure_chunks' in analysis_data:
                results['figure_visualization'] = self.create_figure_visualization(
                    analysis_data['sample_figure_chunks']
                )
            
            results['interactive_dashboard'] = self.create_interactive_dashboard(analysis_data)
            results['content_heatmap'] = self.create_content_heatmap(analysis_data)
            
            if 'sample_table_chunks' in analysis_data and 'sample_figure_chunks' in analysis_data:
                results['sample_content_cards'] = self.create_sample_content_cards(
                    analysis_data['sample_table_chunks'],
                    analysis_data['sample_figure_chunks']
                )
            
            # Create summary report
            self._create_visualization_report(results, analysis_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}
    
    def _create_visualization_report(self, results: Dict[str, str], analysis_data: Dict[str, Any]):
        """Create a summary report of all visualizations.
        
        Args:
            results: Dictionary of visualization results
            analysis_data: Analysis data
        """
        try:
            report_path = self.output_dir / "visualization_report.md"
            
            with open(report_path, 'w') as f:
                f.write("# Visualization Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Summary Statistics\n\n")
                f.write(f"- Total Tables: {analysis_data.get('total_tables', 0)}\n")
                f.write(f"- Total Figures: {analysis_data.get('total_figures', 0)}\n")
                f.write(f"- Documents with Tables: {len(analysis_data.get('tables_by_document', {}))}\n")
                f.write(f"- Documents with Figures: {len(analysis_data.get('figures_by_document', {}))}\n\n")
                
                f.write("## Generated Visualizations\n\n")
                for viz_name, viz_path in results.items():
                    if viz_path:
                        f.write(f"- **{viz_name.replace('_', ' ').title()}**: `{viz_path}`\n")
                
                f.write("\n## Usage Instructions\n\n")
                f.write("1. **Summary Chart**: Overview of content distribution\n")
                f.write("2. **Table Visualization**: Analysis of extracted tables\n")
                f.write("3. **Figure Visualization**: Analysis of extracted figures\n")
                f.write("4. **Interactive Dashboard**: Interactive Plotly dashboard\n")
                f.write("5. **Content Heatmap**: Content distribution across documents\n")
                f.write("6. **Sample Content Cards**: Sample extracted content\n")
            
            logger.info(f"Visualization report created: {report_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization report: {e}")


def create_visualization_from_analysis(analysis_file: str, output_dir: str = "output/visualizations") -> Dict[str, str]:
    """Convenience function to create visualizations from analysis file.
    
    Args:
        analysis_file: Path to analysis JSON file
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary of generated visualization paths
    """
    visualizer = ContentVisualizer(output_dir)
    return visualizer.generate_all_visualizations(analysis_file)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        analysis_file = sys.argv[1]
        if Path(analysis_file).exists():
            results = create_visualization_from_analysis(analysis_file)
            print("Generated visualizations:")
            for name, path in results.items():
                print(f"  {name}: {path}")
        else:
            print(f"Analysis file not found: {analysis_file}")
    else:
        print("Usage: python visualization_utils.py <analysis_file.json>") 