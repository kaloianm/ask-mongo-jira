#!/usr/bin/env python3
"""
generate_graphs.py - Generate data for two analysis graphs from MongoDB collections.

This script connects to MongoDB and creates data for:
1. Graph showing epic duration (weeks) vs percentage of CRUD operations
2. Histogram of epics categorized by catalog change percentages in buckets

Usage:
    python3 generate_graphs.py

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphDataGenerator:
    """Generate data for MongoDB analysis graphs."""
    
    def __init__(self, mongodb_url: str):
        """Initialize with MongoDB connection."""
        self.mongodb_url = mongodb_url
        self.mongodb_client = AsyncIOMotorClient(mongodb_url)
        self.db = self.mongodb_client['ask-mongo-jira']
    
    async def get_epic_duration_vs_crud_percentage(self) -> List[Dict[str, Any]]:
        """
        Generate data for Graph 1: Epic duration (weeks) vs CRUD percentage.
        
        Formula: CRUD% = code_analysis[Catalog CRUD] / (code_analysis[Catalog CRUD] + 
                        code_analysis[Catalog DDL] + code_analysis[Catalog Implementation Change])
        """
        logger.info("Generating data for epic duration vs CRUD percentage...")
        
        pipeline = [
            # Join jira_epics with code_analysis (filtered for categorize v2)
            {
                '$lookup': {
                    'from': 'code_analysis',
                    'localField': 'epic',
                    'foreignField': 'epic_key',
                    'as': 'analyses',
                    'pipeline': [
                        {
                            '$match': {
                                'analysis_type': 'categorize',
                                'analysis_version': 2
                            }
                        }
                    ]
                }
            },
            # Filter out epics with no analysis data and valid dates
            {
                '$match': {
                    'analyses': {'$ne': []},
                    'start_date': {'$exists': True, '$ne': None, '$ne': ''},
                    'end_date': {'$exists': True, '$ne': None, '$ne': ''}
                }
            },
            # Project the basic fields we need
            {
                '$project': {
                    'epic_key': '$epic',
                    'name': '$summary',
                    'start_date_raw': '$start_date',
                    'end_date_raw': '$end_date',
                    'analyses': '$analyses'
                }
            },
            # Unwind analyses to process each classification
            {
                '$unwind': '$analyses'
            },
            # Group by epic and count classifications
            {
                '$group': {
                    '_id': {
                        'epic_key': '$epic_key',
                        'name': '$name',
                        'start_date_raw': '$start_date_raw',
                        'end_date_raw': '$end_date_raw'
                    },
                    'catalog_crud': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog CRUD']},
                                1, 0
                            ]
                        }
                    },
                    'catalog_ddl': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog DDL']},
                                1, 0
                            ]
                        }
                    },
                    'catalog_implementation_change': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog Implementation Change']},
                                1, 0
                            ]
                        }
                    },
                    'data_movement': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Data Movement']},
                                1, 0
                            ]
                        }
                    },
                    'something_else': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Something Else']},
                                1, 0
                            ]
                        }
                    },
                    'total_classifications': {'$sum': 1}
                }
            },
            # Calculate percentages and prepare for Python date processing
            {
                '$project': {
                    'epic_key': '$_id.epic_key',
                    'name': '$_id.name',
                    'start_date_raw': '$_id.start_date_raw',
                    'end_date_raw': '$_id.end_date_raw',
                    'catalog_crud': '$catalog_crud',
                    'catalog_ddl': '$catalog_ddl',
                    'catalog_implementation_change': '$catalog_implementation_change',
                    'data_movement': '$data_movement',
                    'something_else': '$something_else',
                    'total_classifications': '$total_classifications',
                    'catalog_total': {
                        '$add': ['$catalog_crud', '$catalog_ddl', '$catalog_implementation_change']
                    }
                }
            },
            {
                '$project': {
                    'epic_key': 1,
                    'name': 1,
                    'start_date_raw': 1,
                    'end_date_raw': 1,
                    'catalog_crud': 1,
                    'catalog_ddl': 1,
                    'catalog_implementation_change': 1,
                    'data_movement': 1,
                    'something_else': 1,
                    'catalog_total': 1,
                    'total_classifications': 1,
                    'catalog_percentage': {
                        '$cond': [
                            {'$gt': ['$total_classifications', 0]},
                            {'$multiply': [{'$divide': ['$catalog_total', '$total_classifications']}, 100]},
                            0
                        ]
                    }
                }
            }
        ]
        
        cursor = self.db.jira_epics.aggregate(pipeline)
        results = []
        async for doc in cursor:
            # Handle date parsing in Python
            try:
                start_date = doc['start_date_raw']
                end_date = doc['end_date_raw']
                
                # Parse dates - handle both string and datetime objects
                if isinstance(start_date, str):
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                else:
                    start_dt = start_date
                    
                if isinstance(end_date, str):
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
                else:
                    end_dt = end_date
                
                # Calculate duration in weeks
                duration_days = (end_dt - start_dt).days
                duration_weeks = duration_days / 7
                
                # Add calculated fields to the document
                doc['duration_weeks'] = duration_weeks
                doc['start_date'] = start_dt
                doc['end_date'] = end_dt
                
                results.append(doc)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping epic {doc.get('epic_key')} due to date parsing error: {e}")
                continue
        
        logger.info(f"Found {len(results)} epics with valid duration and catalog data")
        return results
    
    async def get_catalog_change_histogram_data(self) -> List[Dict[str, Any]]:
        """
        Generate data for Graph 2: Histogram of epics by catalog change percentage buckets.
        
        Buckets: [0-10%], [10-30%], [30-50%], [50-100%]
        """
        logger.info("Generating data for catalog change histogram...")
        
        pipeline = [
            # Join jira_epics with code_analysis (filtered for categorize v2)
            {
                '$lookup': {
                    'from': 'code_analysis',
                    'localField': 'epic',
                    'foreignField': 'epic_key',
                    'as': 'analyses',
                    'pipeline': [
                        {
                            '$match': {
                                'analysis_type': 'categorize',
                                'analysis_version': 2
                            }
                        }
                    ]
                }
            },
            # Filter out epics with no analysis data
            {
                '$match': {
                    'analyses': {'$ne': []}
                }
            },
            # Unwind analyses to process each classification
            {
                '$unwind': '$analyses'
            },
            # Group by epic and count classifications
            {
                '$group': {
                    '_id': {
                        'epic_key': '$epic',
                        'name': '$summary'
                    },
                    'catalog_crud': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog CRUD']},
                                1, 0
                            ]
                        }
                    },
                    'catalog_ddl': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog DDL']},
                                1, 0
                            ]
                        }
                    },
                    'catalog_implementation_change': {
                        '$sum': {
                            '$cond': [
                                {'$eq': ['$analyses.classification', 'Catalog Implementation Change']},
                                1, 0
                            ]
                        }
                    },
                    'total_classifications': {'$sum': 1}
                }
            },
            # Calculate catalog change percentage
            {
                '$project': {
                    'epic_key': '$_id.epic_key',
                    'name': '$_id.name',
                    'catalog_total': {
                        '$add': ['$catalog_crud', '$catalog_ddl', '$catalog_implementation_change']
                    },
                    'total_classifications': '$total_classifications'
                }
            },
            {
                '$project': {
                    'epic_key': 1,
                    'name': 1,
                    'catalog_total': 1,
                    'total_classifications': 1,
                    'catalog_percentage': {
                        '$multiply': [
                            {'$divide': ['$catalog_total', '$total_classifications']},
                            100
                        ]
                    }
                }
            },
            # Categorize into buckets
            {
                '$project': {
                    'epic_key': 1,
                    'name': 1,
                    'catalog_percentage': 1,
                    'bucket': {
                        '$switch': {
                            'branches': [
                                {
                                    'case': {'$lte': ['$catalog_percentage', 10]},
                                    'then': '0-10%'
                                },
                                {
                                    'case': {'$lte': ['$catalog_percentage', 30]},
                                    'then': '10-30%'
                                },
                                {
                                    'case': {'$lte': ['$catalog_percentage', 50]},
                                    'then': '30-50%'
                                }
                            ],
                            'default': '50-100%'
                        }
                    }
                }
            },
            {
                '$sort': {'catalog_percentage': 1}
            }
        ]
        
        cursor = self.db.jira_epics.aggregate(pipeline)
        results = []
        async for doc in cursor:
            results.append(doc)
        
        logger.info(f"Found {len(results)} epics with catalog change data")
        return results
    
    def generate_graph_1(self, data: List[Dict[str, Any]]):
        """Generate Graph 1: Epic Duration vs CRUD Percentage."""
        if not data:
            logger.warning("No data available for Graph 1")
            return
        
        # Extract data for plotting
        durations = [item['duration_weeks'] for item in data]
        catalog_percentages = [item['catalog_percentage'] for item in data]
        epic_names = [f"{item['epic_key']}: {item['name'][:30]}..." if len(item['name']) > 30 
                      else f"{item['epic_key']}: {item['name']}" for item in data]
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(durations, catalog_percentages, alpha=0.6, s=100, c='blue', edgecolors='black')
        
        # Customize the plot
        ax.set_xlabel('Epic Duration (Weeks)', fontsize=12)
        ax.set_ylabel('Catalog Work Percentage (%)', fontsize=12)
        ax.set_title('Epic Duration vs Catalog Work Percentage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create tooltip annotations
        annotations = []
        for i, (dur, catalog, name) in enumerate(zip(durations, catalog_percentages, epic_names)):
            annotation = ax.annotate(name, 
                                   xy=(dur, catalog), 
                                   xytext=(20, 20), 
                                   textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                   fontsize=9,
                                   visible=False)
            annotations.append(annotation)
        
        # Function to handle mouse motion
        def on_hover(event):
            if event.inaxes == ax:
                # Find the closest point
                if len(durations) > 0:
                    distances = [(event.xdata - dur)**2 + (event.ydata - catalog)**2 
                               for dur, catalog in zip(durations, catalog_percentages)]
                    closest_idx = distances.index(min(distances))
                    
                    # Show annotation for closest point if mouse is close enough
                    if min(distances) < (max(durations) * 0.05)**2:  # Threshold for activation
                        # Hide all annotations first
                        for ann in annotations:
                            ann.set_visible(False)
                        # Show the closest one
                        annotations[closest_idx].set_visible(True)
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
            ax.plot(durations, p(durations), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.2f})')
            ax.legend()
        
        # Add some permanent annotations for extreme points
        for i, (dur, catalog, name) in enumerate(zip(durations, catalog_percentages, epic_names)):
            if catalog > 80 or dur > max(durations) * 0.8:
                ax.annotate(name, (dur, catalog), xytext=(10, 10), textcoords='offset points',
                           fontsize=8, ha='left', alpha=0.7, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('epic_duration_vs_catalog_percentage.png', dpi=300, bbox_inches='tight')
        
        # Show interactive plot with tooltips (keep window open)
        print("Interactive plot opened - hover over data points to see epic names!")
        print("Close the plot window to continue...")
        plt.show(block=True)
        
        # Print summary statistics
        print(f"\nGraph 1 Summary:")
        print(f"Total epics: {len(data)}")
        print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} weeks")
        print(f"Catalog percentage range: {min(catalog_percentages):.1f}% - {max(catalog_percentages):.1f}%")
        print(f"Average duration: {np.mean(durations):.1f} weeks")
        print(f"Average catalog percentage: {np.mean(catalog_percentages):.1f}%")
    
    def generate_graph_2(self, data: List[Dict[str, Any]]):
        """Generate Graph 2: Histogram of Catalog Change Percentages."""
        if not data:
            logger.warning("No data available for Graph 2")
            return
        
        # Count epics in each bucket
        bucket_counts = {'0-10%': 0, '10-30%': 0, '30-50%': 0, '50-100%': 0}
        for item in data:
            bucket = item['bucket']
            bucket_counts[bucket] += 1
        
        # Create the histogram
        plt.figure(figsize=(10, 6))
        buckets = list(bucket_counts.keys())
        counts = list(bucket_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = plt.bar(buckets, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Customize the plot
        plt.xlabel('Catalog Change Percentage Buckets', fontsize=12)
        plt.ylabel('Number of Epics', fontsize=12)
        plt.title('Distribution of Epics by Catalog Change Percentage', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('catalog_change_histogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed breakdown
        print(f"\nGraph 2 Summary:")
        print(f"Total epics: {sum(counts)}")
        for bucket, count in bucket_counts.items():
            percentage = (count / sum(counts)) * 100 if sum(counts) > 0 else 0
            print(f"{bucket}: {count} epics ({percentage:.1f}%)")
        
        # Show some examples from each bucket
        print(f"\nExample epics from each bucket:")
        bucket_examples = {bucket: [] for bucket in buckets}
        for item in data:
            bucket = item['bucket']
            if len(bucket_examples[bucket]) < 3:  # Show up to 3 examples per bucket
                bucket_examples[bucket].append(f"{item['epic_key']} ({item['catalog_percentage']:.1f}%)")
        
        for bucket, examples in bucket_examples.items():
            if examples:
                print(f"{bucket}: {', '.join(examples)}")
    
    async def close(self):
        """Close the MongoDB connection."""
        self.mongodb_client.close()


async def main():
    """Main function to generate both graphs."""
    mongodb_url = os.getenv('MONGODB_URL')
    
    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return
    
    logger.info(f"Connecting to MongoDB...")
    generator = GraphDataGenerator(mongodb_url)
    
    try:
        # Generate data for both graphs
        logger.info("Fetching data for Graph 1: Epic Duration vs CRUD Percentage")
        graph1_data = await generator.get_epic_duration_vs_crud_percentage()
        
        logger.info("Fetching data for Graph 2: Catalog Change Histogram")
        graph2_data = await generator.get_catalog_change_histogram_data()
        
        # Generate the graphs
        logger.info("Generating Graph 1...")
        generator.generate_graph_1(graph1_data)
        
        logger.info("Generating Graph 2...")
        generator.generate_graph_2(graph2_data)
        
        logger.info("Both graphs generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating graphs: {e}")
        raise
    finally:
        await generator.close()


if __name__ == "__main__":
    # Install matplotlib if not available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib", "numpy"])
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    
    asyncio.run(main())