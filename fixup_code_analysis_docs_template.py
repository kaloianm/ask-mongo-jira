import argparse
import logging
import os
from dotenv import load_dotenv

from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Setup the global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Analyze code changes using OpenAI API for commits in a JIRA epic")
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    mongodb_url = os.getenv('MONGODB_URL')
    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return

    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_url)
        db = client['ask-mongo-jira']
        collection = db['code_analysis']

        # Query documents where classification is null
        query = {'classification': None}
        documents = collection.find(query)

        # Counter for tracking modified documents
        modified_count = 0

        # Iterate over each document
        for doc in documents:
            print(f"\nProcessing document with _id: {doc['_id']}")

            raw_response_lines = doc['raw_response']
            if not isinstance(raw_response_lines, list):
                raise ValueError("Expected 'raw_response' to be a list of strings")

            classification = None
            reasoning = None

            for i, line in enumerate(raw_response_lines):
                line = line.strip().replace('**', '')
                line_lower = line.lower()

                # Look for classification line
                if line_lower.startswith('classification:'):
                    classification = line.split(':', 1)[1].strip()

                # Look for reasoning line
                if line_lower.startswith('reasoning:'):
                    # Get the reasoning part after the colon
                    reasoning_start = line.split(':', 1)[1].strip()

                    # Collect remaining lines as part of reasoning
                    reasoning_parts = [reasoning_start] if reasoning_start else []
                    for j in range(i + 1, len(raw_response_lines)):
                        remaining_line = raw_response_lines[j].strip()
                        if remaining_line:
                            reasoning_parts.append(remaining_line)

                    reasoning = ' '.join(reasoning_parts)

                if (classification and reasoning):
                    break

            if (not classification) or (not reasoning):
                continue

            doc['classification'] = classification
            doc['reasoning'] = reasoning

            # Write the updated document back to the database
            result = collection.replace_one({'_id': doc['_id']}, doc)

            if result.modified_count > 0:
                print(f"Document with _id {doc['_id']} updated successfully.")
                modified_count += 1
            else:
                print(f"No changes made to document with _id {doc['_id']}.")

        logging.info("\nCompleted. Total documents modified: %d", modified_count)

    finally:
        # Close the MongoDB connection
        client.close()
        logging.info("MongoDB connection closed.")


if __name__ == "__main__":
    main()
