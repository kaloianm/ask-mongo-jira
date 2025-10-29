#!/usr/bin/env python3
"""
analyze_code_changes - Tool for analyzing code changes using OpenAI API based on commits and file
                       changes stored in MongoDB by fetch_jira_epic.py and fetch_code_changes.py.

Usage:
    python3 analyze_code_changes.py --help

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)

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
"""

import argparse
import asyncio
import datetime
import logging
import os
import subprocess
from typing import Dict, Any, List
from pathlib import Path

# Third-party imports
from openai import AsyncOpenAI
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import tomli
import httpx

# Local imports
from aggregations import load_aggregation_pipeline

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


class TokenRefreshHTTPClient(httpx.AsyncClient):
    """
    Custom HTTP client that handles automatic token refresh on 401 Unauthorized errors
    """

    def __init__(self, openai_api_key_refresh_command, **kwargs):
        """
        Initialize the HTTP client with token refresh capability
        """
        super().__init__(**kwargs)
        self.openai_api_key_refresh_command = openai_api_key_refresh_command

        self._current_token = None
        self._refresh_lock = asyncio.Lock()

    async def refresh_token(self):
        """
        Refresh the API token
        """
        async with self._refresh_lock:
            try:
                logger.info("Refreshing OpenAI API token")
                process = subprocess.run(
                    self.openai_api_key_refresh_command,
                    shell=True,
                    check=False,
                    capture_output=True,
                )
                self._current_token = process.stdout.decode().strip()
                logger.info("OpenAI API token refreshed successfully")
            except Exception as e:
                logger.error("Failed to refresh OpenAI API token: %s", e)
                raise

    async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        """
        Override send method to handle token refresh on 401 errors

        Args:
            request: HTTP request to send
            **kwargs: Additional arguments

        Returns:
            HTTP response
        """
        # Update Authorization header with current token
        if "Authorization" in request.headers:
            if not self._current_token:
                await self.refresh_token()
            request.headers["Authorization"] = f"Bearer {self._current_token}"

        # First attempt
        response = await super().send(request, **kwargs)

        # If we get a 401, try to refresh token
        if response.status_code == 401:
            logger.warning("Received 401 Unauthorized, attempting token refresh")

            try:
                # Refresh the token
                await self.refresh_token()

                # Update the request with new token
                request.headers["Authorization"] = f"Bearer {self._current_token}"

                # Retry the request
                logger.info("Retrying request with refreshed token")
                response = await super().send(request, **kwargs)

                if response.status_code == 401:
                    logger.error("Still received 401 after token refresh")
                else:
                    logger.info("Request succeeded after token refresh")

            except Exception as e:
                logger.error("Token refresh failed: %s", e)
                # Pass-through to return the original 401 response if refresh fails

        return response


class CodeAnalyzer:
    """
    Main class for analyzing code changes using OpenAI API
    """

    def __init__(self, mongodb_url: str, openai_api_key: str, openai_api_key_refresh_command: str,
                 openai_base_url: str, openai_model: str, config: Dict[str, Any]):
        """
        Initialize with configuration parameters

        Args:
            mongodb_url: MongoDB connection URL
            openai_api_key: OpenAI API key
            openai_api_key_refresh_command: Command to refresh OpenAI API key
            openai_base_url: OpenAI API base URL
            openai_model: OpenAI model to use
            config: Configuration dictionary
        """
        # Store configuration
        self.mongodb_url = mongodb_url
        self.openai_api_key = openai_api_key
        self.openai_api_key_refresh_command = openai_api_key_refresh_command
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.config = config

        # Log configuration
        logger.info("Initializing CodeAnalyzer with configuration:")
        logger.info("  mongodb_url: %s", "***" if mongodb_url else None)
        logger.info("  openai_api_key: %s", "***" if openai_api_key else None)
        logger.info("  openai_api_key_refresh_command: %s", openai_api_key_refresh_command)
        logger.info("  openai_base_url: %s", self.openai_base_url)
        logger.info("  openai_model: %s", self.openai_model)

        # Initialize MongoDB client
        self.mongodb_client = AsyncIOMotorClient(self.mongodb_url)
        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]
        self.commits_collection = self.db["commits"]
        self.file_changes_collection = self.db["file_changes"]
        self.code_analysis_collection = self.db["code_analysis"]

        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = AsyncOpenAI(
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
            )
        elif openai_api_key_refresh_command:
            self.openai_client = AsyncOpenAI(
                base_url=self.openai_base_url,
                api_key="TO BE FILLED LATER",
                http_client=TokenRefreshHTTPClient(openai_api_key_refresh_command),
            )

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

    async def _get_epic_data_by_issue(self, epic_key: str):
        """
        Get all epic data grouped by JIRA issue using a single aggregation to get issues and commits,
        then fetch file changes separately for better performance.

        Args:
            epic_key: The epic ticket ID (e.g., "SPM-12345")

        Returns:
            List of documents with issue data and all related commits with file changes
        """
        logger.info("Fetching aggregated data by issue for %s",
                    epic_key if epic_key else "all epics")

        # Load MongoDB aggregation pipeline from JSON file
        pipeline = load_aggregation_pipeline("analyze_code_changes_aggregation.json")

        # Replace the placeholder with the actual epic key
        if epic_key:
            pipeline[0]["$match"]["epic"] = epic_key
        else:
            pipeline[0]["$match"].pop("epic")

        async for issue_data in self.jira_issues_collection.aggregate(pipeline):
            commits = issue_data['commits']
            for commit in commits:
                commit['file_changes'] = await self._get_file_changes_for_commit(commit['id'])

            yield issue_data

    def _generate_issue_analysis_question(self, issue_data: Dict[str, Any], model_to_use: str,
                                          question_key: str) -> List[Dict[str, str]]:
        """
        Generate analysis questions for an entire JIRA issue (all commits and file changes)

        Args:
            issue_data: Issue document with all commits and file changes

        Returns:
            List of dictionaries with 'type' and 'question' keys
        """
        issue_epic = issue_data['epic_key']
        issue_key = issue_data['issue_key']
        issue_summary = issue_data['issue_summary']
        issue_description = issue_data['issue_description']
        issue_commits = issue_data['commits']

        # Collect all changes across all commits
        commit_ids = []
        commit_diffs = []

        for commit in issue_commits:
            commit_id = commit['id']
            commit_detail = commit['detail']
            if not commit_detail:
                logging.debug("Skipping commit %s from (%s, %s) with missing details", commit_id,
                              issue_epic, issue_key)
                continue

            commit_author = commit_detail['author']
            commit_message = commit_detail['message']
            commit_stats = commit_detail['stats']

            commit_all_file_changes = []

            if commit_stats['total'] < 10000:
                for file_change in commit['file_changes']:
                    filename = file_change['filename']
                    status = file_change['status']

                    if file_change['changes'] > 5000:
                        patch = f"[Patch too large to display ({file_change['changes']} changes)]"
                    elif file_change['changes'] > 2500:
                        patch = file_change['short_patch']
                    else:
                        patch = file_change['patch']

                    commit_all_file_changes.append(
                        f"File: {filename}\nStatus: {status}\nPatch:\n{patch}\n")
            else:
                commit_all_file_changes.append(
                    f"[Skipped file changes for commit {commit_id} due to large size ({commit_stats['total']} changes)]"
                )

            changes_text = '\n'.join(commit_all_file_changes)
            separator = '-' * 40
            single_commit = f"""
                Commit ID: {commit_id}\n
                Author: {commit_author}\n
                Message: {commit_message}\n
                Changes: {changes_text}\n{separator}\n
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

        analysis_questions_config = self.config['analysis_questions']
        question_config = analysis_questions_config[question_key]
        return {
            'analysis_type':
            question_key,
            'analysis_version':
            question_config['version'],
            'model_used':
            model_to_use,
            'question':
            question_config['prompt'] + '\n' + question_config['template'].format(**template_vars),
            'commit_ids':
            commit_ids
        }

    def _extract_classification_and_reasoning(self, response_text: str):
        """
        Extract classification and reasoning from OpenAI response text
        """
        lines = response_text.split('\n')
        classification = None
        reasoning = None

        for i, line in enumerate(lines):
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
                for j in range(i + 1, len(lines)):
                    remaining_line = lines[j].strip()
                    if remaining_line:
                        reasoning_parts.append(remaining_line)

                reasoning = ' '.join(reasoning_parts)

            if (classification and reasoning):
                break

        return classification, reasoning

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

        response_text = response.choices[0].message.content.strip()
        classification, reasoning = self._extract_classification_and_reasoning(response_text)

        # Return parsed result if both parts found, otherwise return raw response
        if classification and reasoning:
            return {
                'classification': classification,
                'reasoning': reasoning,
                'raw_response': response_text,
                'usage_stats': usage_stats
            }
        else:
            # If parsing failed, return the raw response in a structured format
            logger.warning("Could not parse classification response, storing raw response only")
            return {
                'classification': None,
                'reasoning': None,
                'raw_response': response_text,
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
            "model_used": analysis_data["model_used"],
        }

        # Upsert the analysis data
        result = await self.code_analysis_collection.replace_one(filter_query,
                                                                 analysis_data,
                                                                 upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Analysis %s in code_analysis collection for issue %s", action,
                     analysis_data["issue_key"])

    async def process_epic(self, epic_key: str, question_key: str):
        """
        Main processing function: analyze all code changes in an epic by JIRA issue
        
        Args:
            epic_key: The epic ticket ID to analyze
        """
        processed_issues = 0  # == skipped_issues + analyzed_issues + errors
        skipped_issues = 0
        analyzed_issues = 0
        errors = 0

        async for issue_data in self._get_epic_data_by_issue(epic_key):
            issue_epic = issue_data['epic_key']
            issue_key = issue_data['issue_key']
            processed_issues += 1

            # Generate analysis questions for the entire issue
            question_data = self._generate_issue_analysis_question(issue_data, self.openai_model,
                                                                   question_key)
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
                    "model_used":
                    question_data['model_used'],
                })

                if existing:
                    logger.debug("Skipping analysis for %s/%s (%s/%s/%s)", issue_epic, issue_key,
                                 question_data['analysis_type'], question_data['analysis_version'],
                                 question_data['model_used'])
                    skipped_issues += 1
                    continue

                logger.info("Analyzing %s/%s (%s/%s/%s)", issue_epic, issue_key,
                            question_data['analysis_type'], question_data['analysis_version'],
                            question_data['model_used'])

                # Get OpenAI analysis
                response = await self._analyze_with_openai(question_data['question'])

                # Prepare analysis document
                analysis_doc = {
                    'epic_key': issue_epic,
                    'issue_key': issue_key,
                    'analysis_type': question_data['analysis_type'],
                    'analysis_version': question_data['analysis_version'],
                    'model_used': self.openai_model,
                    'commit_ids': question_data['commit_ids'],
                    'classification': response['classification'],
                    'reasoning': response['reasoning'],
                    'question': question_data['question'],
                    'raw_response': response['raw_response'],
                    'usage_stats': response['usage_stats'],
                    'timestamp': datetime.datetime.now(datetime.timezone.utc),
                }

                # Store in MongoDB
                await self._store_issue_analysis_in_mongodb(analysis_doc)

                logger.info("Completed analysis for issue %s (%s/%s/%s)", analysis_doc['issue_key'],
                            analysis_doc['analysis_type'], analysis_doc['analysis_version'],
                            analysis_doc['model_used'])
                analyzed_issues += 1

            except Exception as e:
                logger.error("Error analyzing issue %s/%s (%s/%s): %s", issue_epic, issue_key,
                             question_data['analysis_type'], question_data['analysis_version'], e)
                errors += 1

        logger.info("Analysis complete: %d processed, %d skipped, %d analyzed, %d errors",
                    processed_issues, skipped_issues, analyzed_issues, errors)
        assert processed_issues == (skipped_issues + analyzed_issues + errors)

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
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")

    parser.add_argument("--epic", help="Epic ticket ID to analyze (e.g., SPM-1234)")

    # OpenAI configuration parameters
    parser.add_argument("--openai-base-url",
                        required=True,
                        help="OpenAI API base URL (e.g., https://api.openai.com/v1)")
    parser.add_argument("--openai-api-key", help="OpenAI API key for authentication")
    parser.add_argument("--openai-api-key-refresh-command",
                        help="Command to refresh OpenAI API key (alternative to --openai-api-key)")
    parser.add_argument("--openai-model", required=True, help="OpenAI model to use (e.g., gpt-4)")

    args = parser.parse_args()

    # Validate that exactly one of the API key options is provided
    if not args.openai_api_key and not args.openai_api_key_refresh_command:
        parser.error("Either --openai-api-key or --openai-api-key-refresh-command must be provided")

    if args.openai_api_key and args.openai_api_key_refresh_command:
        parser.error(
            "Only one of --openai-api-key or --openai-api-key-refresh-command can be provided, not both"
        )

    # Setup logging
    setup_logging(args.log_level)

    # Extract configuration from environment variables and command line arguments
    mongodb_url = os.getenv('MONGODB_URL')
    openai_base_url = args.openai_base_url
    openai_api_key = args.openai_api_key
    openai_api_key_refresh_command = args.openai_api_key_refresh_command
    openai_model = args.openai_model

    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
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
                            openai_api_key_refresh_command=openai_api_key_refresh_command,
                            openai_base_url=openai_base_url,
                            openai_model=openai_model,
                            config=config)

    try:
        # Set up database indexes
        await analyzer.setup_database_indexes()

        # Process the epic analysis
        analysis_questions_config = analyzer.config['analysis_questions']

        for question_key in analysis_questions_config.keys():
            logger.info("Starting code analysis for %s ...",
                        args.epic if args.epic else "all epics")
            await analyzer.process_epic(args.epic, question_key)
            logger.info("Code analysis completed successfully")

    finally:
        await analyzer.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
