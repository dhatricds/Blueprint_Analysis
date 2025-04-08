#!/usr/bin/env python3
"""
Generate detailed report from blueprint analysis results.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)

def create_summary_plots(results, output_dir):
    """Create summary visualizations."""
    # Confidence distribution
    plt.figure(figsize=(10, 6))
    confidences = [match["confidence"] for match in results["matches"]]
    sns.histplot(confidences, bins=20)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.savefig(output_dir / "confidence_dist.png")
    plt.close()
    
    # Symbol locations
    plt.figure(figsize=(10, 10))
    x = [match["x"] for match in results["matches"]]
    y = [match["y"] for match in results["matches"]]
    plt.scatter(x, y, alpha=0.5)
    plt.title("Symbol Locations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig(output_dir / "symbol_locations.png")
    plt.close()

def generate_html_report(results, output_dir):
    """Generate HTML report with results and visualizations."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Blueprint Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Blueprint Analysis Report</h1>
            <p>Generated on: {now}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Total Symbols</th>
                        <td>{results["total_matches"]}</td>
                    </tr>
                    <tr>
                        <th>Blueprint</th>
                        <td>{results["tile_path"]}</td>
                    </tr>
                    <tr>
                        <th>Reference Symbol</th>
                        <td>{results["symbol_path"]}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <h3>Confidence Distribution</h3>
                <img src="confidence_dist.png" alt="Confidence Distribution">
                
                <h3>Symbol Locations</h3>
                <img src="symbol_locations.png" alt="Symbol Locations">
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>X</th>
                        <th>Y</th>
                        <th>Confidence</th>
                        <th>Scale</th>
                    </tr>
    """
    
    for match in results["matches"]:
        html += f"""
                    <tr>
                        <td>{match["x"]}</td>
                        <td>{match["y"]}</td>
                        <td>{match["confidence"]:.3f}</td>
                        <td>{match["scale"]}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / "report.html", "w") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description="Generate report from blueprint analysis results")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--output-dir", default="report", help="Output directory for report")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results
        results = load_results(args.results)
        
        # Create visualizations
        create_summary_plots(results, output_dir)
        
        # Generate HTML report
        generate_html_report(results, output_dir)
        
        print(f"Report generated in {output_dir}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())