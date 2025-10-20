#!/usr/bin/env python3
"""
fetch_jira_epic - Tool for fetching all issues in a Jira epic and their development information,
                  and storing them in MongoDB.

Usage:
    python fetch_jira_epic.py --help

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
    JIRA_URL - Jira server URL (required)
    JIRA_API_TOKEN - Jira API token (required)

The script will store all fetched issues as documents in the "ask-mongo-jira" database, in the
"jira_issues" collection.
"""

import os
import json
import datetime
import logging
import argparse
import asyncio
from urllib.parse import urlparse

from typing import Dict, Any

# Third-party imports
from jira import JIRA
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReplaceOne

# Load environment variables from .env file
load_dotenv()

# Setup the global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration
    """
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


class JiraIssueFetcher:
    """
    Main class for querying Jira
    """

    def __init__(self, jira_url: str, jira_token: str, mongodb_url):
        """
        Initialize with configuration parameters
        """
        self.jira_url = jira_url
        self.jira_token = jira_token
        self.mongodb_url = mongodb_url

        # Log configuration (mask sensitive tokens)
        logger.info("Initializing JiraIssueFetcher with configuration:")
        logger.info("  jira_url: %s", jira_url)
        logger.info("  jira_token: %s", "***" if jira_token else None)
        logger.info("  mongodb_url: %s", "***" if mongodb_url else None)

        # Initialize clients
        self.jira_client = JIRA(server=self.jira_url, token_auth=self.jira_token)
        self.mongodb_client = AsyncIOMotorClient(self.mongodb_url)

        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]

    def _is_allowed_repository(self, url: str) -> bool:
        """
        Check if a repository URL belongs to allowed organizations
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        # We need at least 2 parts: owner and repo
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub commit URL: insufficient path components")

        owner = path_parts[0]
        repo = path_parts[1]

        # Validate owner and repo are non-empty
        if not owner or not repo:
            raise ValueError("Invalid GitHub commit URL: owner or repo is empty")

        return owner in ['mongodb', '10gen']

    def _get_development_info_via_api(self, issue) -> Dict[str, Any]:
        """
        Get development information using Jira's REST API directly
        """
        dev_info = {'commits': [], 'branches': [], 'pull_requests': []}

        # Get the internal issue ID (numeric) instead of the issue key
        issue_id = issue.id if hasattr(issue, 'id') else None
        issue_key = issue.key if hasattr(issue, 'key') else 'unknown'

        if not issue_id:
            logger.warning("Could not get internal issue ID for %s", issue_key)
            return dev_info

        # Use the development information REST endpoint
        dev_url = f"{self.jira_url}/rest/dev-status/1.0/issue/detail"
        params = {
            'issueId': issue_id,  # Use numeric ID instead of key
            'applicationType': 'github',  # or 'bitbucket', 'gitlab'
            'dataType': 'repository'
        }

        logger.info("Attempting to fetch development info via API for %s (ID: %s)", issue_key,
                    issue_id)

        # Use the Jira client's internal session which already has authentication
        jira_session = getattr(self.jira_client, '_session', None)
        if not jira_session:
            logger.warning("Cannot access Jira session for development API call")
            return dev_info

        # Make the actual API request
        response = jira_session.get(dev_url, params=params)

        if response.status_code == 200:
            data = response.json()
            logger.debug("Development API response: %s", json.dumps(data, indent=2))

            # Parse the response to extract commits, branches, and PRs
            if 'detail' in data:
                for detail in data['detail']:
                    if 'repositories' in detail:
                        for repo in detail['repositories']:
                            repo_url = repo.get('url')
                            if not self._is_allowed_repository(repo_url):
                                continue

                            # Extract commits
                            if 'commits' in repo:
                                for commit in repo['commits']:
                                    commit_info = {
                                        'id':
                                        commit.get('id', commit.get('displayId', None)),
                                        'message':
                                        commit.get('message', None),
                                        'author':
                                        commit.get('author', {}).get('name', None),
                                        'timestamp':
                                        datetime.datetime.fromisoformat(
                                            commit.get('authorTimestamp').replace('Z', '+00:00'))
                                        if commit.get('authorTimestamp') else None,
                                        'url':
                                        commit.get('url', None),
                                        'raw':
                                        commit,
                                    }
                                    dev_info['commits'].append(commit_info)

                            # Extract branches
                            if 'branches' in repo:
                                for branch in repo['branches']:
                                    branch_info = {
                                        'name': branch.get('name', None),
                                        'url': branch.get('url', None),
                                        'raw': branch,
                                    }
                                    dev_info['branches'].append(branch_info)

                            # Extract pull requests
                            if 'pullRequests' in repo:
                                for pr in repo['pullRequests']:
                                    pr_info = {
                                        'id': pr.get('id', None),
                                        'title': pr.get('title', None),
                                        'status': pr.get('status', None),
                                        'url': pr.get('url', None),
                                        'author': pr.get('author', {}).get('name', None),
                                        'raw': pr,
                                    }
                                    dev_info['pull_requests'].append(pr_info)

            logger.info("Found %d commits, %d branches, %d PRs for issue %s",
                        len(dev_info['commits']), len(dev_info['branches']),
                        len(dev_info['pull_requests']), issue_key)
        else:
            logger.warning("Development API returned status %d for issue %s", response.status_code,
                           issue_key)
            if response.status_code == 404:
                logger.info("Development info API not available or issue has no development data")

        return dev_info

    def _extract_development_info(self, issue) -> Dict[str, Any]:
        """
        Extract development information (commits, branches, PRs) from a Jira issue
        """
        dev_info = {'commits': [], 'branches': [], 'pull_requests': []}

        # Try to get development information using the REST API. This requires the issue to be
        # expanded with 'devinfo'.
        if hasattr(issue, 'fields') and hasattr(issue.fields, 'development'):
            development = issue.fields.development

            # Extract commits
            if hasattr(development, 'commits'):
                for commit in development.commits:
                    commit_url = getattr(commit, 'url', None)

                    # Only process commits from mongodb/* and 10gen/* repositories
                    if not self._is_allowed_repository(commit_url):
                        continue

                    commit_info = {
                        'id':
                        getattr(commit, 'id', None),
                        'message':
                        getattr(commit, 'message', None),
                        'author':
                        getattr(commit, 'author', {}).get('name', None) if hasattr(
                            commit, 'author') else None,
                        'timestamp':
                        datetime.datetime.fromisoformat(
                            str(getattr(commit, 'timestamp', None)).replace('Z', '+00:00'))
                        if getattr(commit, 'timestamp', None) else None,
                        'url':
                        commit_url,
                        'raw':
                        commit,
                    }
                    dev_info['commits'].append(commit_info)

            # Extract branches
            if hasattr(development, 'branches'):
                for branch in development.branches:
                    branch_info = {
                        'name': getattr(branch, 'name', None),
                        'url': getattr(branch, 'url', None),
                        'raw': branch,
                    }
                    dev_info['branches'].append(branch_info)

            # Extract pull requests
            if hasattr(development, 'pullRequests'):
                for pr in development.pullRequests:
                    pr_info = {
                        'id':
                        getattr(pr, 'id', None),
                        'title':
                        getattr(pr, 'title', None),
                        'status':
                        getattr(pr, 'status', None),
                        'url':
                        getattr(pr, 'url', None),
                        'author':
                        getattr(pr, 'author', {}).get('name', None)
                        if hasattr(pr, 'author') else None,
                        'raw':
                        pr,
                    }
                    dev_info['pull_requests'].append(pr_info)

        # Alternative approach: Use Jira's REST API directly
        if not dev_info['commits'] and not dev_info['branches'] and not dev_info['pull_requests']:
            dev_info = self._get_development_info_via_api(issue)

        return dev_info if any(dev_info.values()) else None

    async def get_epic_issues(self, epic_key: str):
        """
        Get all issues from Core Server project that belong to a specific epic - returns an iterator
        """
        # JQL to find all issues in Core Server project that belong to the epic
        jql = f'project = "Core Server" AND "Epic Link" = {epic_key}'
        logger.info("Searching for issues in epic %s using JQL: %s", epic_key, jql)

        # Search for issues with expanded fields
        issues = self.jira_client.search_issues(jql,
                                                maxResults=False,
                                                expand='changelog,comment,devinfo')

        logger.info("Found %d issues in epic %s", len(issues), epic_key)

        # Yield each issue with full details (similar to get_jira_ticket)
        for issue in issues:
            existing = await self.jira_issues_collection.find_one({
                "epic": epic_key,
                "issue": issue.key
            })

            if existing:
                logger.info("Issue %s (%s) already exists in MongoDB, skipping", epic_key,
                            issue.key)
                continue

            # Get detailed issue information (reusing logic from get_jira_ticket)
            issue_data = {
                'epic':
                epic_key,
                'issue':
                issue.key,
                'summary':
                issue.fields.summary,
                'description':
                issue.fields.description,
                'status':
                issue.fields.status.name,
                'created':
                datetime.datetime.fromisoformat(issue.fields.created.replace('Z', '+00:00'))
                if issue.fields.created else None,
                'updated':
                datetime.datetime.fromisoformat(issue.fields.updated.replace('Z', '+00:00'))
                if issue.fields.updated else None,
                'issue_type':
                issue.fields.issuetype.name,
                'priority':
                issue.fields.priority.name if issue.fields.priority else None,
                'components':
                [comp.name for comp in issue.fields.components] if issue.fields.components else [],
            }

            # Extract development information if available
            issue_data['development'] = self._extract_development_info(issue)

            yield issue_data

    async def store_issues_in_mongodb(self, issues: list) -> None:
        """
        Store issues in MongoDB database using batch upsert operations
        """
        logger.info("Batch upserting %d issues in MongoDB", len(issues))

        if not issues:
            logger.info("No issues to store in MongoDB")
            return

        # Add timestamp for when records were last updated
        current_time = datetime.datetime.now(datetime.timezone.utc)

        # Prepare bulk operations for batch processing
        bulk_operations = []

        for issue in issues:
            # Add last_updated timestamp
            issue["last_updated"] = current_time

            # Create filter for unique identification (epic + issue)
            filter_query = {"epic": issue["epic"], "issue": issue["issue"]}

            # Create ReplaceOne operation with upsert=True
            operation = ReplaceOne(filter_query, issue, upsert=True)
            bulk_operations.append(operation)

        # Execute all operations in a single batch
        result = await self.jira_issues_collection.bulk_write(bulk_operations, ordered=False)

        # Log the results
        inserted_count = result.upserted_count
        updated_count = result.modified_count

        logger.info("Batch operation completed: %d inserted, %d updated, %d total processed",
                    inserted_count, updated_count, len(issues))

    async def setup_database_indexes(self) -> None:
        """
        Set up database indexes for efficient querying
        """
        # Create a compound index on epic and issue for uniqueness and efficient queries
        await self.jira_issues_collection.create_index([("epic", 1), ("issue", 1)],
                                                       unique=True,
                                                       name="epic_issue_unique")

        # Create a unique index on the issue field
        await self.jira_issues_collection.create_index([("issue", 1)],
                                                       unique=True,
                                                       name="issue_unique")

        logger.info("Database indexes created successfully")

    async def close_mongodb_connection(self) -> None:
        """
        Close MongoDB connection
        """
        if self.mongodb_client:
            self.mongodb_client.close()
            logger.info("MongoDB connection closed")


async def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Query Jira tickets and store in MongoDB")
    parser.add_argument("epic", help="Epic ticket ID to get all Core Server issues in that epic")
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")

    args = parser.parse_args()

    # Setup logging with specified level
    setup_logging(args.log_level)

    # Extract configuration from environment variables
    jira_url = os.getenv('JIRA_URL')
    jira_token = os.getenv('JIRA_API_TOKEN')
    mongodb_url = os.getenv('MONGODB_URL')

    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return

    # Initialize the tool with configuration
    tool = JiraIssueFetcher(jira_url=jira_url, jira_token=jira_token, mongodb_url=mongodb_url)

    try:
        # Set up database indexes
        await tool.setup_database_indexes()

        # Collect data
        logger.info("Fetching issues for epic: %s", args.epic)

        # Collect all issues from the iterator
        issues = []
        async for issue in tool.get_epic_issues(args.epic):
            issues.append(issue)

        # Store in MongoDB instead of printing
        await tool.store_issues_in_mongodb(issues)

        logger.info("Successfully processed %d issues for epic %s", len(issues), args.epic)

    finally:
        await tool.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
