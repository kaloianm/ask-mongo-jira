#!/usr/bin/env python3
"""
generate_graph_data.py - Generate CSV data files for MongoDB analysis graphs.

This script creates CSV files with the data used for the graphs, making it easy to 
import into other tools or perform additional analysis.

Usage:
    python3 generate_graph_data.py

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
"""

import asyncio
import csv
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from generate_graphs import GraphDataGenerator

async def main():
    """Main function to generate CSV data files."""
    mongodb_url = os.getenv('MONGODB_URL')
    
    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return
    
    logger.info("Connecting to MongoDB...")
    generator = GraphDataGenerator(mongodb_url)
    
    try:
        # Generate data for both graphs
        logger.info("Fetching data for Graph 1: Epic Duration vs CRUD Percentage")
        graph1_data = await generator.get_epic_duration_vs_crud_percentage()
        
        logger.info("Fetching data for Graph 2: Catalog Change Histogram")
        graph2_data = await generator.get_catalog_change_histogram_data()
        
        # Save Graph 1 data to CSV
        with open('epic_duration_vs_catalog_data.csv', 'w', newline='') as csvfile:
            fieldnames = ['epic_key', 'name', 'duration_weeks', 'catalog_percentage', 'catalog_crud', 'catalog_ddl', 'catalog_implementation_change', 'catalog_total', 'data_movement', 'something_else', 'total_classifications', 'start_date', 'end_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in graph1_data:
                writer.writerow({
                    'epic_key': row['epic_key'],
                    'name': row['name'][:100],  # Truncate long names
                    'duration_weeks': round(row['duration_weeks'], 2),
                    'catalog_percentage': round(row['catalog_percentage'], 2),
                    'catalog_crud': row['catalog_crud'],
                    'catalog_ddl': row['catalog_ddl'],
                    'catalog_implementation_change': row['catalog_implementation_change'],
                    'catalog_total': row['catalog_total'],
                    'data_movement': row['data_movement'],
                    'something_else': row['something_else'],
                    'total_classifications': row['total_classifications'],
                    'start_date': row['start_date'].strftime('%Y-%m-%d') if row['start_date'] else '',
                    'end_date': row['end_date'].strftime('%Y-%m-%d') if row['end_date'] else ''
                })
        
        logger.info(f"Saved {len(graph1_data)} records to epic_duration_vs_catalog_data.csv")
        
        # Save Graph 2 data to CSV
        with open('catalog_change_histogram_data.csv', 'w', newline='') as csvfile:
            fieldnames = ['epic_key', 'name', 'catalog_percentage', 'bucket']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in graph2_data:
                writer.writerow({
                    'epic_key': row['epic_key'],
                    'name': row['name'][:100],  # Truncate long names
                    'catalog_percentage': round(row['catalog_percentage'], 2),
                    'bucket': row['bucket']
                })
        
        logger.info(f"Saved {len(graph2_data)} records to catalog_change_histogram_data.csv")
        
        # Print summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"\nGraph 1: Epic Duration vs Catalog Percentage")
        print(f"- Total epics analyzed: {len(graph1_data)}")
        if graph1_data:
            durations = [item['duration_weeks'] for item in graph1_data]
            catalog_percentages = [item['catalog_percentage'] for item in graph1_data]
            print(f"- Duration range: {min(durations):.1f} - {max(durations):.1f} weeks")
            print(f"- Catalog percentage range: {min(catalog_percentages):.1f}% - {max(catalog_percentages):.1f}%")
            print(f"- Average duration: {sum(durations)/len(durations):.1f} weeks")
            print(f"- Average catalog percentage: {sum(catalog_percentages)/len(catalog_percentages):.1f}%")
            
            # Show top 5 longest and shortest epics
            sorted_by_duration = sorted(graph1_data, key=lambda x: x['duration_weeks'], reverse=True)
            print(f"\nTop 5 longest epics:")
            for epic in sorted_by_duration[:5]:
                print(f"  {epic['epic_key']}: {epic['duration_weeks']:.1f} weeks, {epic['catalog_percentage']:.1f}% Catalog")
            
            print(f"\nTop 5 shortest epics:")
            for epic in sorted_by_duration[-5:]:
                print(f"  {epic['epic_key']}: {epic['duration_weeks']:.1f} weeks, {epic['catalog_percentage']:.1f}% Catalog")
        
        print(f"\nGraph 2: Catalog Change Histogram")
        print(f"- Total epics analyzed: {len(graph2_data)}")
        if graph2_data:
            # Count by bucket
            bucket_counts = {}
            for item in graph2_data:
                bucket = item['bucket']
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            
            for bucket in ['0-10%', '10-30%', '30-50%', '50-100%']:
                count = bucket_counts.get(bucket, 0)
                percentage = (count / len(graph2_data)) * 100 if len(graph2_data) > 0 else 0
                print(f"- {bucket}: {count} epics ({percentage:.1f}%)")
        
        logger.info("Data export completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main())