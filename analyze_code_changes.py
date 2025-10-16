#!/usr/bin/env python3
"""
analyze_code_changes - A tool for analyzing code changes using OpenAI API based on commits and file
changes stored in MongoDB by fetch_jira_epic.py and fetch_code_changes.py.

Usage:
    python analyze_code_changes.py EPIC-123

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
    OPENAI_API_KEY - OpenAI API key (required)
    OPENAI_BASE_URL - OpenAI API base URL (optional, defaults to https://api.openai.com/v1)
    OPENAI_MODEL - OpenAI model to use (optional, defaults to gpt-4)
    MCP_SERVER_URL - MCP Server URL (optional, for enhanced code analysis with additional context)

The script will:
1. Query the "ask-mongo-jira" database using aggregation to join
   jira_issues -> commits -> file_changes
2. For each JIRA issue, generate comprehensive analysis questions covering all commits and file
   changes
3. Store the analysis results in the "code_analysis" collection

The aggregation approach reduces database round-trips and provides better performance by joining:
- JIRA issues from the epic
- Related commits referenced in the development.commits field of each issue
- File changes for each commit
All in a single MongoDB aggregation pipeline.

The code_analysis collection will contain documents with:
- epic_key: The epic ticket ID
- issue_key: The JIRA issue key
- analysis_type: Type of analysis performed (e.g., "issue_summary", "issue_potential_issues",
  "issue_impact_assessment")
- question: The question sent to OpenAI
- response: OpenAI's analysis response
- model_used: The OpenAI model used for analysis
- timestamp: When the analysis was performed
- last_updated: When this record was last updated
- issue_stats: Statistics about the issue (commit count, repositories, file changes count)
"""

import os
import logging
import argparse
import asyncio
import datetime
from typing import Dict, Any, List
from pathlib import Path

# Third-party imports
from openai import AsyncOpenAI
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import tomli

# Load environment variables from .env file
load_dotenv()

# Setup the global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file"""
    if not Path(config_path).exists():
        config_path = Path(__file__).parent / "config.toml"

    try:
        with open(config_path, 'rb') as f:
            return tomli.load(f)
    except (FileNotFoundError, tomli.TOMLDecodeError) as e:
        logger.error("Error loading configuration file %s: %s", config_path, e)
        raise ValueError(f"Could not load configuration from {config_path}: {e}") from e


class CodeAnalyzer:
    """Main class for analyzing code changes using OpenAI API"""

    def __init__(self, mongodb_url: str, openai_api_key: str, openai_base_url: str,
                 openai_model: str, mcp_server_url: str, config: Dict[str, Any]):
        """Initialize with configuration parameters"""
        # Store configuration
        self.mongodb_url = mongodb_url
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.mcp_server_url = mcp_server_url
        self.config = config

        # Log configuration
        logger.info("Initializing CodeAnalyzer with configuration:")
        logger.info("  mongodb_url: %s", "***" if mongodb_url else None)
        logger.info("  openai_api_key: %s", "***" if openai_api_key else None)
        logger.info("  openai_base_url: %s", self.openai_base_url)
        logger.info("  openai_model: %s", self.openai_model)
        logger.info("  mcp_server_url: %s", self.mcp_server_url)

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)

        # Initialize MongoDB client
        self.mongodb_client = AsyncIOMotorClient(self.mongodb_url)
        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]
        self.commits_collection = self.db["commits"]
        self.file_changes_collection = self.db["file_changes"]
        self.code_analysis_collection = self.db["code_analysis"]

    async def _get_file_changes_for_commit(self, commit_id: str) -> List[Dict[str, Any]]:
        """
        Get all file changes for a specific commit
        
        Args:
            commit_id: The git commit SHA
            
        Returns:
            List of file change documents from MongoDB
        """
        logger.debug("Fetching file changes for commit %s", commit_id[:8])

        cursor = self.file_changes_collection.find({"commit_id": commit_id})
        file_changes = await cursor.to_list(length=None)

        logger.debug("Found %d file changes for commit %s", len(file_changes), commit_id[:8])
        return file_changes

    async def _get_epic_data_by_issue(self, epic_key: str) -> List[Dict[str, Any]]:
        """
        Get all epic data grouped by JIRA issue using a single aggregation to get issues and commits,
        then fetch file changes separately for better performance.
        
        Args:
            epic_key: The epic ticket ID (e.g., "SERVER-12345")
            
        Returns:
            List of documents with issue data and all related commits with file changes
        """
        logger.info("Fetching aggregated data by issue for epic %s", epic_key)

        # MongoDB aggregation pipeline to get issues and their commits (without file changes)
        pipeline = [
            # Stage 1: Match issues in the epic
            {
                "$match": {
                    "epic": epic_key
                }
            },

            # Stage 2: Unwind development.commits array to process each commit separately
            {
                "$unwind": {
                    "path": "$development.commits",
                    "preserveNullAndEmptyArrays": True
                }
            },

            # Stage 3: Lookup commits using development.commits.id as foreign key
            {
                "$lookup": {
                    "from": "commits",
                    "localField": "development.commits.id",
                    "foreignField": "commit_id",
                    "as": "commit_details"
                }
            },

            # Stage 4: Unwind commit details
            {
                "$unwind": {
                    "path": "$commit_details",
                    "preserveNullAndEmptyArrays": True
                }
            },

            # Stage 5: Group back by issue to collect all commits (without file changes)
            {
                "$group": {
                    "_id": "$key",
                    "epic_key": {
                        "$first": "$epic"
                    },
                    "issue_key": {
                        "$first": "$key"
                    },
                    "issue_summary": {
                        "$first": "$summary"
                    },
                    "issue_description": {
                        "$first": "$description"
                    },
                    "commits": {
                        "$push": {
                            "$cond": {
                                "if": {
                                    "$ne": ["$commit_details", None]
                                },
                                "then": {
                                    "commit_id": "$commit_details.commit_id",
                                    "message": "$commit_details.message",
                                    "author": "$commit_details.author",
                                    "repository": "$commit_details.repository"
                                },
                                "else": "$$REMOVE"
                            }
                        }
                    }
                }
            },

            # Stage 6: Filter out issues without commits
            {
                "$match": {
                    "commits": {
                        "$ne": []
                    }
                }
            }
        ]

        cursor = self.jira_issues_collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)

        # Now fetch file changes separately for each commit
        for issue_data in results:
            commits = issue_data.get('commits', [])
            for commit in commits:
                commit_id = commit.get('commit_id')
                if commit_id:
                    file_changes = await self._get_file_changes_for_commit(commit_id)
                    commit['file_changes'] = file_changes
                else:
                    commit['file_changes'] = []

        logger.info("Found %d issues with commits", len(results))
        return results

    def _generate_issue_analysis_questions(self, issue_data: Dict[str,
                                                                  Any]) -> List[Dict[str, str]]:
        """
        Generate analysis questions for an entire JIRA issue (all commits and file changes)
        
        Args:
            issue_data: Issue document with all commits and file changes
            
        Returns:
            List of dictionaries with 'type' and 'question' keys
        """
        issue_key = issue_data.get('issue_key')
        issue_summary = issue_data.get('issue_summary')
        issue_description = issue_data.get('issue_description')
        issue_commits = issue_data.get('commits', [])

        # Collect all changes across all commits
        commit_ids = []
        commit_diffs = []

        for commit in issue_commits:
            commit_id = commit.get('commit_id')
            commit_author = commit.get('author')
            commit_message = commit.get('message')
            file_changes = commit.get('file_changes')

            all_file_changes = []
            for file_change in file_changes:
                filename = file_change.get('filename')
                status = file_change.get('status')
                patch = file_change.get('patch')

                all_file_changes.append(f"File: {filename}\nStatus: {status}\nPatch:\n{patch}\n")

            single_commit = f"""
                Commit ID: {commit_id}\n
                Author: {commit_author}\n
                Message: {commit_message}\n
                Changes: {'\n'.join(all_file_changes)}\n{'-'*40}\n
            """

            commit_ids.append(commit_id)
            commit_diffs.append(single_commit)

        # Template variables for string formatting
        template_vars = {
            'issue_key':
            issue_key,
            'issue_summary':
            issue_summary,
            'issue_description':
            issue_description,
            'commit_diffs':
            '\n'.join(commit_diffs) if len(commit_diffs) > 0 else 'No commit details available',
        }

        questions = []

        analysis_questions_config = self.config['analysis_questions']

        for question_key, question_config in analysis_questions_config.items():
            questions.append({
                'analysis_type': question_key,
                'analysis_version': question_config['version'],
                'question': question_config['template'].format(**template_vars),
            })

        return {'questions': questions, 'commit_ids': commit_ids}

    async def _analyze_with_openai(self, question: str) -> str:
        """
        Send a question to OpenAI and get the analysis response
        
        Args:
            question: The analysis question to send to OpenAI
            
        Returns:
            OpenAI's response text
        """
        logger.debug("Sending question to OpenAI (model: %s)", self.openai_model)

        # Get OpenAI configuration
        openai_config = self.config['openai']
        system_prompt = openai_config['system_prompt']

        # Use the new OpenAI v2+ async client
        response = await self.openai_client.chat.completions.create(model=self.openai_model,
                                                                    messages=[{
                                                                        "role":
                                                                        "system",
                                                                        "content":
                                                                        system_prompt
                                                                    }, {
                                                                        "role": "user",
                                                                        "content": question
                                                                    }])

        usage = response.usage
        usage_stats = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }

        lines = response.choices[0].message.content.strip().split('\n')
        classification = None
        reasoning = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for classification line
            if line.lower().startswith('classification:'):
                classification = line.split(':', 1)[1].strip()

            # Look for reasoning line
            elif line.lower().startswith('reasoning:'):
                # Get the reasoning part after the colon
                reasoning_start = line.split(':', 1)[1].strip()

                # Collect remaining lines as part of reasoning
                reasoning_parts = [reasoning_start] if reasoning_start else []
                for j in range(i + 1, len(lines)):
                    remaining_line = lines[j].strip()
                    if remaining_line:
                        reasoning_parts.append(remaining_line)

                reasoning = ' '.join(reasoning_parts)
                break

        # Return parsed result if both parts found, otherwise return raw response
        if classification and reasoning:
            return {
                'classification': classification,
                'reasoning': reasoning,
                'raw_response': lines,
                'usage_stats': usage_stats
            }
        else:
            # If parsing failed, return the raw response in a structured format
            logger.warning("Could not parse classification response, storing raw response only")
            return {
                'classification': None,
                'reasoning': None,
                'raw_response': lines,
                'usage_stats': usage_stats
            }

    async def _store_issue_analysis_in_mongodb(self, analysis_data: Dict[str, Any]):
        """
        Store issue-level code analysis results in MongoDB
        
        Args:
            analysis_data: Dictionary containing analysis results
        """
        analysis_data['last_updated'] = datetime.datetime.now(datetime.timezone.utc)

        # Create a unique filter based on epic, issue, and analysis type
        filter_query = {
            "epic_key": analysis_data["epic_key"],
            "issue_key": analysis_data["issue_key"],
            "analysis_type": analysis_data["analysis_type"],
            "analysis_version": analysis_data["analysis_version"],
        }

        # Upsert the analysis data
        result = await self.code_analysis_collection.replace_one(filter_query,
                                                                 analysis_data,
                                                                 upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Analysis %s in code_analysis collection for issue %s", action,
                     analysis_data["issue_key"])

    async def process_epic(self, epic_key: str) -> None:
        """
        Main processing function: analyze all code changes in an epic by JIRA issue
        
        Args:
            epic_key: The epic ticket ID to analyze
        """
        logger.info("Starting issue-level analysis for epic %s", epic_key)

        # Get all data grouped by issue
        epic_issues = await self._get_epic_data_by_issue(epic_key)

        total_analyses = 0
        processed_issues = 0
        errors = 0

        for issue_data in epic_issues:
            issue_epic = issue_data['epic_key']
            issue_key = issue_data['issue_key']

            # Generate analysis questions for the entire issue
            questions = self._generate_issue_analysis_questions(issue_data)

            for question_data in questions['questions']:
                try:
                    # Check if we already have this analysis
                    existing = await self.code_analysis_collection.find_one({
                        "epic_key":
                        issue_epic,
                        "issue_key":
                        issue_key,
                        "analysis_type":
                        question_data['analysis_type'],
                        "analysis_version":
                        question_data['analysis_version'],
                    })

                    if existing:
                        logger.info("Skipping analysis for %s (%s/%s)", issue_key,
                                    question_data['analysis_type'],
                                    question_data['analysis_version'])
                        continue

                    logger.info("Analyzing %s (%s/%s)", issue_key, question_data['analysis_type'],
                                question_data['analysis_version'])

                    # TODO: Enhance question with MCP context if available
                    enhanced_question = question_data['question']

                    # Get OpenAI analysis
                    response = await self._analyze_with_openai(enhanced_question)

                    # Prepare analysis document
                    analysis_doc = {
                        'epic_key': issue_epic,
                        'issue_key': issue_key,
                        'commit_ids': questions['commit_ids'],
                        'analysis_type': question_data['analysis_type'],
                        'analysis_version': question_data['analysis_version'],
                        'question': enhanced_question,
                        'classification': response['classification'],
                        'reasoning': response['reasoning'],
                        'raw_response': response['raw_response'],
                        'model_used': self.openai_model,
                        'usage_stats': response.get('usage_stats'),
                        'timestamp': datetime.datetime.now(datetime.timezone.utc),
                    }

                    # Store in MongoDB
                    await self._store_issue_analysis_in_mongodb(analysis_doc)
                    total_analyses += 1

                    logger.info("Completed %s analysis for issue %s",
                                question_data['analysis_type'], issue_key)

                except Exception as e:
                    logger.error("Error analyzing issue %s (%s): %s", issue_key,
                                 question_data['analysis_type'], e)
                    errors += 1

            processed_issues += 1

        logger.info("Epic analysis complete: %d issues processed, %d total analyses, %d errors",
                    processed_issues, total_analyses, errors)

    async def setup_database_indexes(self) -> None:
        """Set up database indexes for efficient querying on the code_analysis collection"""
        # Create compound index for epic_key and issue_key for efficient ticket-level queries
        await self.code_analysis_collection.create_index([("epic_key", 1), ("issue_key", 1)],
                                                         name="epic_issue_index")

        logger.info("Database indexes created successfully for code_analysis collection")

    async def close_mongodb_connection(self) -> None:
        """Close MongoDB connection"""
        if self.mongodb_client:
            self.mongodb_client.close()
            logger.info("MongoDB connection closed")


async def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Analyze code changes using OpenAI API for commits in a JIRA epic")
    parser.add_argument("epic", help="Epic ticket ID to analyze (e.g., SPM-1234)")
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Extract configuration from environment variables and arguments
    mongodb_url = os.getenv('MONGODB_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_base_url = os.getenv('OPENAI_BASE_URL')
    openai_model = os.getenv('OPENAI_MODEL')
    mcp_server_url = os.getenv('MCP_SERVER_URL')

    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return

    if not openai_api_key:
        logger.error("OpenAI API key is required. Set the OPENAI_API_KEY environment variable")
        return

    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except ValueError as e:
        logger.error("Failed to load configuration: %s", e)
        return

    # Initialize the analyzer
    analyzer = CodeAnalyzer(mongodb_url=mongodb_url,
                            openai_api_key=openai_api_key,
                            openai_base_url=openai_base_url,
                            openai_model=openai_model,
                            mcp_server_url=mcp_server_url,
                            config=config)

    try:
        # Set up database indexes
        await analyzer.setup_database_indexes()

        # Process the epic analysis
        logger.info("Starting code analysis for epic %s...", args.epic)
        await analyzer.process_epic(args.epic)
        logger.info("Code analysis completed successfully")

    finally:
        # Clean up MongoDB connection
        await analyzer.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
