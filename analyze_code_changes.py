#!/usr/bin/env python3
"""
analyze_code_changes - A tool for analyzing code changes using OpenAI API based on commits and file
changes stored in MongoDB by fetch_jira_epic.py and fetch_code_changes.py.

Usage:
    python analyze_code_changes.py EPIC-123 --openai-api-key your-api-key

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
    OPENAI_API_KEY - OpenAI API key (required)
    OPENAI_BASE_URL - OpenAI API base URL (optional, defaults to https://api.openai.com/v1)
    OPENAI_MODEL - OpenAI model to use (optional, defaults to gpt-4)
    MCP_SERVER_URL - MCP Server URL (optional, for enhanced code analysis with additional context)

The script will:
1. Query the "ask-mongo-jira" database using aggregation to join jira_issues -> commits -> file_changes
2. For each JIRA issue, generate comprehensive analysis questions covering all commits and file changes
3. Store the analysis results in the "code_analysis" collection

The aggregation approach reduces database round-trips and provides better performance by joining:
- JIRA issues from the epic
- Related commits referenced in the development.commits field of each issue
- File changes for each commit
All in a single MongoDB aggregation pipeline.

The code_analysis collection will contain documents with:
- epic_key: The epic ticket ID
- issue_key: The JIRA issue key
- analysis_type: Type of analysis performed (e.g., "issue_summary", "issue_potential_issues", "issue_impact_assessment")
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
from openai import APIError, APIConnectionError, APITimeoutError
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

        # TODO: Add MCP server integration for enhanced code context
        # When mcp_server_url is provided, this could be used to:
        # - Fetch additional code context from repositories
        # - Get related code dependencies and relationships
        # - Enhance analysis with broader codebase understanding

        # Initialize MongoDB client
        self.mongodb_client = AsyncIOMotorClient(self.mongodb_url)
        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]
        self.commits_collection = self.db["commits"]
        self.file_changes_collection = self.db["file_changes"]
        self.code_analysis_collection = self.db["code_analysis"]

    async def get_epic_issues(self, epic_key: str) -> List[Dict[str, Any]]:
        """
        Get all SERVER tickets from the epic
        
        Args:
            epic_key: The epic ticket ID (e.g., "SERVER-12345")
            
        Returns:
            List of issue documents from MongoDB
        """
        logger.info("Fetching all issues for epic %s", epic_key)

        # Find all issues in the epic
        cursor = self.jira_issues_collection.find({"epic": epic_key})
        issues = await cursor.to_list(length=None)

        logger.info("Found %d issues in epic %s", len(issues), epic_key)
        return issues

    async def get_commits_for_issue(self, issue_key: str) -> List[Dict[str, Any]]:
        """
        Get all commits related to a specific JIRA issue
        
        Args:
            issue_key: The JIRA issue key (e.g., "SERVER-67890")
            
        Returns:
            List of commit documents from MongoDB
        """
        logger.debug("Fetching commits for issue %s", issue_key)

        # Find commits that reference this JIRA issue
        cursor = self.commits_collection.find({"jira_issues": issue_key})
        commits = await cursor.to_list(length=None)

        logger.debug("Found %d commits for issue %s", len(commits), issue_key)
        return commits

    async def get_file_changes_for_commit(self, commit_id: str) -> List[Dict[str, Any]]:
        """
        Get all file changes for a specific commit
        
        Args:
            commit_id: The git commit SHA
            
        Returns:
            List of file change documents from MongoDB
        """
        logger.debug("Fetching file changes for commit %s", commit_id[:8])

        # Find all file changes for this commit
        cursor = self.file_changes_collection.find({"commit_id": commit_id})
        file_changes = await cursor.to_list(length=None)

        logger.debug("Found %d file changes for commit %s", len(file_changes), commit_id[:8])
        return file_changes

    def _generate_analysis_questions(self, file_change: Dict[str, Any],
                                     commit_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate different types of analysis questions for a file change
        
        Args:
            file_change: File change document from MongoDB
            commit_data: Commit document from MongoDB
            
        Returns:
            List of dictionaries with 'type' and 'question' keys
        """
        filename = file_change.get('filename', '')
        status = file_change.get('status', '')
        patch = file_change.get('patch', '')
        commit_message = commit_data.get('message', '')
        author = commit_data.get('author', {}).get('name', 'Unknown')

        # Template variables for string formatting
        template_vars = {
            'filename': filename,
            'status': status,
            'patch': patch,
            'commit_message': commit_message,
            'author': author
        }

        questions = []
        questions_config = self.config.get('analysis_questions', {})

        # Change summary question
        change_summary_config = questions_config.get('change_summary', {})
        questions.append({
            'type':
            change_summary_config.get('type', 'change_summary'),
            'question':
            change_summary_config.get('template', '').format(**template_vars)
        })

        # Potential issues question (only for substantial changes)
        min_patch_length = questions_config.get('min_patch_length', 50)
        if patch and len(patch) > min_patch_length:
            potential_issues_config = questions_config.get('potential_issues', {})
            questions.append({
                'type':
                potential_issues_config.get('type', 'potential_issues'),
                'question':
                potential_issues_config.get('template', '').format(**template_vars)
            })

        # Impact assessment question
        impact_assessment_config = questions_config.get('impact_assessment', {})
        questions.append({
            'type':
            impact_assessment_config.get('type', 'impact_assessment'),
            'question':
            impact_assessment_config.get('template', '').format(**template_vars)
        })

        return questions

    def _generate_issue_analysis_questions(self, issue_data: Dict[str,
                                                                  Any]) -> List[Dict[str, str]]:
        """
        Generate analysis questions for an entire JIRA issue (all commits and file changes)
        
        Args:
            issue_data: Issue document with all commits and file changes
            
        Returns:
            List of dictionaries with 'type' and 'question' keys
        """
        issue_key = issue_data.get('issue_key', '')
        issue_summary = issue_data.get('issue_summary', '')
        issue_description = issue_data.get('issue_description', '')
        commits = issue_data.get('commits', [])

        # Collect all file changes across all commits
        all_file_changes = []
        commit_summaries = []
        repositories = set()

        for commit in commits:
            commit_id = commit.get('commit_id', '')
            commit_message = commit.get('message', '')
            commit_author = commit.get('author', {})
            repository = commit.get('repository', '')
            file_changes = commit.get('file_changes', [])

            if repository:
                repositories.add(repository)

            if commit_message:
                author_name = commit_author.get('name', 'Unknown') if isinstance(
                    commit_author, dict) else 'Unknown'
                commit_summaries.append(
                    f"- {commit_id[:8]}: {commit_message.strip()} (by {author_name})")

            all_file_changes.extend(file_changes)

        # Group file changes by filename to show consolidated changes
        files_summary = {}
        for file_change in all_file_changes:
            filename = file_change.get('filename', '')
            if filename:
                if filename not in files_summary:
                    files_summary[filename] = {
                        'status': file_change.get('status', ''),
                        'total_additions': 0,
                        'total_deletions': 0,
                        'patches': []
                    }
                files_summary[filename]['total_additions'] += file_change.get('additions', 0)
                files_summary[filename]['total_deletions'] += file_change.get('deletions', 0)
                if file_change.get('patch'):
                    files_summary[filename]['patches'].append(file_change.get('patch', ''))

        # Create file changes summary text
        files_text = []
        for filename, summary in files_summary.items():
            files_text.append(
                f"- {filename} ({summary['status']}, +{summary['total_additions']}/-{summary['total_deletions']} lines)"
            )

        # Template variables for string formatting
        template_vars = {
            'issue_key': issue_key,
            'issue_summary': issue_summary,
            'issue_description': issue_description or 'No description available',
            'repositories': ', '.join(repositories) if repositories else 'Unknown',
            'commit_count': len(commits),
            'file_count': len(files_summary),
            'commits_summary':
            '\n'.join(commit_summaries) if commit_summaries else 'No commit details available',
            'files_summary': '\n'.join(files_text) if files_text else 'No file changes available'
        }

        questions = []
        questions_config = self.config.get('analysis_questions', {})

        # Issue summary analysis
        change_summary_config = questions_config.get('change_summary', {})
        questions.append({
            'type':
            change_summary_config.get('type', 'issue_summary'),
            'question':
            change_summary_config.get('template', '').format(**template_vars)
        })

        # Only generate detailed analysis if there are substantial changes
        total_changes = sum(summary['total_additions'] + summary['total_deletions']
                            for summary in files_summary.values())
        min_patch_length = questions_config.get('min_patch_length', 50)

        if total_changes > min_patch_length:
            # Potential issues analysis
            potential_issues_config = questions_config.get('potential_issues', {})
            questions.append({
                'type':
                potential_issues_config.get('type', 'issue_potential_issues'),
                'question':
                potential_issues_config.get('template', '').format(**template_vars)
            })

        # Impact assessment analysis
        impact_assessment_config = questions_config.get('impact_assessment', {})
        questions.append({
            'type':
            impact_assessment_config.get('type', 'issue_impact_assessment'),
            'question':
            impact_assessment_config.get('template', '').format(**template_vars)
        })

        return questions

    async def get_mcp_context(self, filename: str, repository: str) -> str:
        """
        Get additional context from MCP server for enhanced code analysis
        
        Args:
            filename: The file being analyzed
            repository: The repository name
            
        Returns:
            Additional context string or empty string if MCP not configured
        """
        if not self.mcp_server_url:
            return ""

        # TODO: Implement MCP server integration
        # This could include:
        # - Fetching related files and dependencies
        # - Getting code structure and architecture context
        # - Retrieving documentation and comments
        logger.debug("MCP server integration not yet implemented for %s in %s", filename,
                     repository)
        return ""

    async def analyze_with_openai(self, question: str) -> str:
        """
        Send a question to OpenAI and get the analysis response
        
        Args:
            question: The analysis question to send to OpenAI
            
        Returns:
            OpenAI's response text
        """
        try:
            logger.debug("Sending question to OpenAI (model: %s)", self.openai_model)

            # Get OpenAI configuration
            openai_config = self.config.get('openai', {})
            system_prompt = openai_config.get(
                'system_prompt', 'You are an expert software engineer analyzing code changes.')

            # Use the new OpenAI v2+ async client
            response = await self.openai_client.chat.completions.create(model=self.openai_model,
                                                                        messages=[{
                                                                            "role":
                                                                            "system",
                                                                            "content":
                                                                            system_prompt
                                                                        }, {
                                                                            "role":
                                                                            "user",
                                                                            "content":
                                                                            question
                                                                        }])

            return response.choices[0].message.content.strip()

        except (APIConnectionError, APITimeoutError) as e:
            logger.error("Connection error calling OpenAI API: %s", e)
            return f"Connection error analyzing code: {str(e)}"
        except APIError as e:
            logger.error("OpenAI API error: %s", e)
            return f"Error analyzing code: {str(e)}"
        except (KeyError, AttributeError) as e:
            logger.error("API response format error: %s", e)
            return f"API response format error: {str(e)}"

    async def store_analysis_in_mongodb(self, analysis_data: Dict[str, Any]):
        """
        Store code analysis results in MongoDB
        
        Args:
            analysis_data: Dictionary containing analysis results
        """
        analysis_data['last_updated'] = datetime.datetime.now(datetime.timezone.utc)

        # Create a unique filter based on epic, issue, commit, file, and analysis type
        filter_query = {
            "epic_key": analysis_data["epic_key"],
            "issue_key": analysis_data["issue_key"],
            "commit_id": analysis_data["commit_id"],
            "filename": analysis_data["filename"],
            "analysis_type": analysis_data["analysis_type"]
        }

        # Upsert the analysis data
        result = await self.code_analysis_collection.replace_one(filter_query,
                                                                 analysis_data,
                                                                 upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Analysis %s in code_analysis collection for %s:%s", action,
                     analysis_data["issue_key"], analysis_data["filename"])

    async def get_epic_data_by_issue(self, epic_key: str) -> List[Dict[str, Any]]:
        """
        Get all epic data grouped by JIRA issue (all commits and file changes per issue)
        
        Args:
            epic_key: The epic ticket ID (e.g., "SERVER-12345")
            
        Returns:
            List of documents with issue data and all related commits/file changes
        """
        logger.info("Fetching aggregated data by issue for epic %s", epic_key)

        # MongoDB aggregation pipeline to group all changes by JIRA issue
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

            # Stage 5: Lookup file changes for each commit
            {
                "$lookup": {
                    "from": "file_changes",
                    "localField": "commit_details.commit_id",
                    "foreignField": "commit_id",
                    "as": "file_changes"
                }
            },

            # Stage 6: Group back by issue to collect all commits and file changes
            {
                "$group": {
                    "_id": "$key",
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
                                    "repository": "$commit_details.repository",
                                    "file_changes": "$file_changes"
                                },
                                "else": "$$REMOVE"
                            }
                        }
                    }
                }
            },

            # Stage 7: Filter out issues without commits
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

        logger.info("Found %d issues with commits for epic %s", len(results), epic_key)
        return results

    async def process_epic_analysis(self, epic_key: str) -> None:
        """
        Main processing function: analyze all code changes in an epic by JIRA issue
        
        Args:
            epic_key: The epic ticket ID to analyze
        """
        logger.info("Starting issue-level analysis for epic %s", epic_key)

        # Get all data grouped by issue
        epic_issues = await self.get_epic_data_by_issue(epic_key)

        total_analyses = 0
        processed_issues = 0
        errors = 0

        for issue_data in epic_issues:
            issue_key = issue_data.get('issue_key', 'unknown')

            if not issue_key:
                logger.warning("Skipping issue with no key")
                continue

            logger.info("Analyzing issue %s with %d commits", issue_key,
                        len(issue_data.get('commits', [])))

            # Generate analysis questions for the entire issue
            questions = self._generate_issue_analysis_questions(issue_data)

            for question_data in questions:
                try:
                    # Check if we already have this analysis
                    existing = await self.code_analysis_collection.find_one({
                        "epic_key":
                        epic_key,
                        "issue_key":
                        issue_key,
                        "analysis_type":
                        question_data['type']
                    })

                    if existing:
                        logger.debug("Analysis already exists for %s (%s)", issue_key,
                                     question_data['type'])
                        continue

                    # Get additional context from MCP server if configured
                    # Use the first repository found in the commits
                    repositories = set()
                    for commit in issue_data.get('commits', []):
                        repo = commit.get('repository', '')
                        if repo:
                            repositories.add(repo)

                    mcp_context = ""
                    if repositories:
                        first_repo = next(iter(repositories))
                        mcp_context = await self.get_mcp_context(issue_key, first_repo)

                    # Enhance question with MCP context if available
                    enhanced_question = question_data['question']
                    if mcp_context:
                        enhanced_question += f"\n\nAdditional Context from Codebase:\n{mcp_context}"

                    # Get OpenAI analysis
                    response = await self.analyze_with_openai(enhanced_question)

                    # Prepare analysis document
                    analysis_doc = {
                        'epic_key': epic_key,
                        'issue_key': issue_key,
                        'analysis_type': question_data['type'],
                        'question': question_data['question'],
                        'response': response,
                        'model_used': self.openai_model,
                        'timestamp': datetime.datetime.now(datetime.timezone.utc),
                        'issue_stats': {
                            'commit_count':
                            len(issue_data.get('commits', [])),
                            'repositories':
                            list(repositories),
                            'total_file_changes':
                            sum(
                                len(commit.get('file_changes', []))
                                for commit in issue_data.get('commits', []))
                        }
                    }

                    # Store in MongoDB
                    await self.store_issue_analysis_in_mongodb(analysis_doc)
                    total_analyses += 1

                    logger.info("Completed %s analysis for issue %s", question_data['type'],
                                issue_key)

                except Exception as e:
                    logger.error("Error analyzing issue %s (%s): %s", issue_key,
                                 question_data['type'], e)
                    errors += 1

            processed_issues += 1

        logger.info("Epic analysis complete: %d issues processed, %d total analyses, %d errors",
                    processed_issues, total_analyses, errors)

    async def store_issue_analysis_in_mongodb(self, analysis_data: Dict[str, Any]):
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
            "analysis_type": analysis_data["analysis_type"]
        }

        # Upsert the analysis data
        result = await self.code_analysis_collection.replace_one(filter_query,
                                                                 analysis_data,
                                                                 upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Analysis %s in code_analysis collection for issue %s", action,
                     analysis_data["issue_key"])

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
    """Main function"""
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
        await analyzer.process_epic_analysis(args.epic)
        logger.info("Code analysis completed successfully")

    finally:
        # Clean up MongoDB connection
        await analyzer.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
