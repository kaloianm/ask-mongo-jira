#!/usr/bin/env python3
"""
interactive_graph.py - Show only the interactive scatter plot with hover tooltips.

This script displays the epic duration vs catalog percentage graph in an interactive
matplotlib window where you can hover over points to see epic names.

Usage:
    python3 interactive_graph.py

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
"""

from generate_graphs import GraphDataGenerator

import asyncio
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Import our existing generator
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_interactive_plot(data):
    """Create an interactive scatter plot with hover tooltips."""
    if not data:
        logger.warning("No data available for interactive plot")
        return

    # Extract data for plotting
    durations = [item['duration_weeks'] for item in data]
    catalog_percentages = [item['catalog_percentage'] for item in data]
    epic_names = [
        f"{item['epic_key']}: {item['name'][:50]}..."
        if len(item['name']) > 50 else f"{item['epic_key']}: {item['name']}" for item in data
    ]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(durations,
                         catalog_percentages,
                         alpha=0.7,
                         s=120,
                         c='blue',
                         edgecolors='black')

    # Customize the plot
    ax.set_xlabel('Epic Duration (Weeks)', fontsize=14)
    ax.set_ylabel('Catalog Work Percentage (%)', fontsize=14)
    ax.set_title(
        'Interactive Epic Duration vs Catalog Work Percentage\n(Hover over points to see epic details)',
        fontsize=16,
        fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create tooltip annotations with smart positioning
    annotations = []
    for i, (dur, catalog, name) in enumerate(zip(durations, catalog_percentages, epic_names)):
        annotation = ax.annotate(f"{name}\nDuration: {dur:.1f} weeks\nCatalog: {catalog:.1f}%",
                                 xy=(dur, catalog),
                                 xytext=(20, 20),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.5",
                                           fc="lightyellow",
                                           alpha=0.9,
                                           edgecolor="black"),
                                 arrowprops=dict(arrowstyle="->",
                                                 connectionstyle="arc3,rad=0.1",
                                                 color="black"),
                                 fontsize=10,
                                 visible=False,
                                 zorder=100,
                                 ha='left')
        annotations.append(annotation)

    # Function to handle mouse motion with smart positioning
    def on_hover(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            # Find the closest point
            if len(durations) > 0:
                distances = [
                    ((event.xdata - dur) / max(durations))**2 + ((event.ydata - catalog) / 100)**2
                    for dur, catalog in zip(durations, catalog_percentages)
                ]
                closest_idx = distances.index(min(distances))

                # Show annotation for closest point if mouse is close enough
                threshold = 0.01  # Adjusted threshold
                if min(distances) < threshold:
                    # Hide all annotations first
                    for ann in annotations:
                        ann.set_visible(False)

                    # Get the annotation to show
                    ann = annotations[closest_idx]
                    point_dur, point_catalog = durations[closest_idx], catalog_percentages[
                        closest_idx]

                    # Smart positioning based on point location in plot
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x_range = xlim[1] - xlim[0]
                    y_range = ylim[1] - ylim[0]

                    # Determine if point is near edges (within 25% of range)
                    near_right = point_dur > xlim[0] + 0.75 * x_range
                    near_top = point_catalog > ylim[0] + 0.75 * y_range
                    near_left = point_dur < xlim[0] + 0.25 * x_range
                    near_bottom = point_catalog < ylim[0] + 0.25 * y_range

                    # Adjust offset based on position
                    if near_right and near_top:
                        offset = (-80, -40)  # Top-left of point
                        ha = 'right'
                    elif near_right and near_bottom:
                        offset = (-80, 40)  # Bottom-left of point
                        ha = 'right'
                    elif near_right:
                        offset = (-80, 20)  # Left of point
                        ha = 'right'
                    elif near_top and near_left:
                        offset = (40, -40)  # Top-right of point
                        ha = 'left'
                    elif near_bottom and near_left:
                        offset = (40, 40)  # Bottom-right of point
                        ha = 'left'
                    elif near_top:
                        offset = (20, -40)  # Below point
                        ha = 'left'
                    elif near_bottom:
                        offset = (20, 40)  # Above point
                        ha = 'left'
                    elif near_left:
                        offset = (40, 20)  # Right of point
                        ha = 'left'
                    else:
                        offset = (20, 20)  # Default: top-right
                        ha = 'left'

                    # Update annotation position and alignment
                    ann.xytext = offset
                    ann.set_ha(ha)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    # Hide all annotations if not close to any point
                    for ann in annotations:
                        ann.set_visible(False)
                    fig.canvas.draw_idle()

    # Connect the hover function
    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    # Add trend line
    if len(durations) > 1:
        z = np.polyfit(durations, catalog_percentages, 1)
        p = np.poly1d(z)
        ax.plot(durations,
                p(durations),
                "r--",
                alpha=0.8,
                linewidth=2,
                label=f'Trend line (slope: {z[0]:.2f})')
        ax.legend(fontsize=12)

    # Add some permanent annotations for extreme points
    for i, (dur, catalog, name) in enumerate(zip(durations, catalog_percentages, epic_names)):
        if catalog > 80 or dur > max(durations) * 0.8:
            ax.annotate(name.split(':')[0], (dur, catalog),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        ha='left',
                        alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.6),
                        zorder=50)

    # Add instructions
    ax.text(0.02,
            0.98,
            'Hover over data points to see epic details',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    # Print summary statistics
    print(f"\n=== Interactive Plot Summary ===")
    print(f"Total epics: {len(data)}")
    print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} weeks")
    print(
        f"Catalog percentage range: {min(catalog_percentages):.1f}% - {max(catalog_percentages):.1f}%"
    )
    print(f"Average duration: {np.mean(durations):.1f} weeks")
    print(f"Average catalog percentage: {np.mean(catalog_percentages):.1f}%")
    print(f"\nHover over the data points to see epic names and details!")
    print("Close the window when done exploring.")

    # Show the interactive plot
    plt.show(block=True)


async def main():
    """Main function to create interactive plot."""
    mongodb_url = os.getenv('MONGODB_URL')

    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return

    logger.info("Connecting to MongoDB for interactive plot...")
    generator = GraphDataGenerator(mongodb_url)

    try:
        # Get data for the plot
        logger.info("Fetching epic duration vs catalog percentage data...")
        graph1_data = await generator.get_epic_duration_vs_crud_percentage()

        # Create interactive plot
        create_interactive_plot(graph1_data)

    except Exception as e:
        logger.error(f"Error creating interactive plot: {e}")
        raise
    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main())
