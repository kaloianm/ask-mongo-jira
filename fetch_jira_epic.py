#!/usr/bin/env python3
"""
fetch_jira_epic - A simple tool for fetching all issues in a Jira epic from the "Core Server"
project, including development information.

Usage:
"""

import os
import json
import logging
import argparse
from typing import Dict, Any

# Third-party imports
from jira import JIRA
from github import Github
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup the global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


class JiraIssueIterator:
    """Main class for querying Jira, GitHub, and OpenAI"""

    def __init__(self,
                 jira_url: str = None,
                 jira_token: str = None,
                 github_token: str = None,
                 openai_api_key: str = None):
        """Initialize with configuration parameters"""
        # Store configuration
        self.jira_url = jira_url
        self.jira_token = jira_token
        self.github_token = github_token
        self.openai_api_key = openai_api_key

        # Log configuration (mask sensitive tokens)
        logger.info("Initializing JiraIssueIterator with configuration:")
        logger.info("  jira_url: %s", jira_url)
        logger.info("  jira_token: %s", "***" if jira_token else None)
        logger.info("  github_token: %s", "***" if github_token else None)
        logger.info("  openai_api_key: %s", "***" if openai_api_key else None)

        # Initialize clients
        self.jira_client = None
        if self.jira_url and self.jira_token:
            self.jira_client = JIRA(server=self.jira_url, token_auth=self.jira_token)

        self.github_client = None
        if self.github_token:
            self.github_client = Github(self.github_token)

        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def _get_development_info_via_api(self, issue) -> Dict[str, Any]:
        """Get development information using Jira's REST API directly"""
        dev_info = {'commits': [], 'branches': [], 'pull_requests': [], 'repositories': []}

        try:
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
                                repo_name = repo.get('name', 'unknown')
                                dev_info['repositories'].append(repo_name)

                                # Extract commits
                                if 'commits' in repo:
                                    for commit in repo['commits']:
                                        commit_info = {
                                            'id': commit.get('id', commit.get('displayId', None)),
                                            'message': commit.get('message', None),
                                            'author': commit.get('author', {}).get('name', None),
                                            'timestamp': commit.get('authorTimestamp', None),
                                            'url': commit.get('url', None),
                                            'repository': repo_name,
                                            'raw': commit,
                                        }
                                        dev_info['commits'].append(commit_info)

                                # Extract branches
                                if 'branches' in repo:
                                    for branch in repo['branches']:
                                        branch_info = {
                                            'name': branch.get('name', None),
                                            'url': branch.get('url', None),
                                            'repository': repo_name
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
                                            'repository': repo_name
                                        }
                                        dev_info['pull_requests'].append(pr_info)

                logger.info("Found %d commits, %d branches, %d PRs for issue %s",
                            len(dev_info['commits']), len(dev_info['branches']),
                            len(dev_info['pull_requests']), issue_key)
            else:
                logger.warning("Development API returned status %d for issue %s",
                               response.status_code, issue_key)
                if response.status_code == 404:
                    logger.info(
                        "Development info API not available or issue has no development data")

        except Exception as e:
            logger.warning("Could not fetch development info via API for %s: %s", issue_key, e)

        return dev_info

    def _extract_development_info(self, issue) -> Dict[str, Any]:
        """Extract development information (commits, branches, PRs) from a Jira issue"""
        dev_info = {'commits': [], 'branches': [], 'pull_requests': [], 'repositories': []}

        try:
            # Try to get development information using the REST API
            # This requires the issue to be expanded with 'devinfo'
            if hasattr(issue, 'fields') and hasattr(issue.fields, 'development'):
                development = issue.fields.development

                # Extract commits
                if hasattr(development, 'commits'):
                    for commit in development.commits:
                        commit_info = {
                            'id':
                            getattr(commit, 'id', None),
                            'message':
                            getattr(commit, 'message', None),
                            'author':
                            getattr(commit, 'author', {}).get('name', None) if hasattr(
                                commit, 'author') else None,
                            'timestamp':
                            str(getattr(commit, 'timestamp', None)) if getattr(
                                commit, 'timestamp', None) else None,
                            'url':
                            getattr(commit, 'url', None),
                            'repository':
                            getattr(commit, 'repository', {}).get('name', None) if hasattr(
                                commit, 'repository') else None,
                            'raw':
                            commit,
                        }
                        dev_info['commits'].append(commit_info)

                # Extract branches
                if hasattr(development, 'branches'):
                    for branch in development.branches:
                        branch_info = {
                            'name':
                            getattr(branch, 'name', None),
                            'url':
                            getattr(branch, 'url', None),
                            'repository':
                            getattr(branch, 'repository', {}).get('name', None) if hasattr(
                                branch, 'repository') else None
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
                            'repository':
                            getattr(pr, 'repository', {}).get('name', None) if hasattr(
                                pr, 'repository') else None,
                        }
                        dev_info['pull_requests'].append(pr_info)

            # Alternative approach: Use Jira's REST API directly
            if not dev_info['commits'] and not dev_info['branches'] and not dev_info[
                    'pull_requests']:
                dev_info = self._get_development_info_via_api(issue)

        except Exception as e:
            logger.warning("Could not extract development info for %s: %s", issue.key, e)

        return dev_info if any(dev_info.values()) else None

    def get_epic_issues(self, epic_key: str):
        """Get all issues from Core Server project that belong to a specific epic - returns an iterator"""
        if not self.jira_client:
            raise ValueError("Jira not configured. Set JIRA_URL and JIRA_API_TOKEN")

        try:
            # JQL to find all issues in Core Server project that belong to the epic
            jql = f'project = "Core Server" AND "Epic Link" = {epic_key}'

            logger.info("Searching for issues in epic %s using JQL: %s", epic_key, jql)

            # Search for issues with expanded fields
            issues = self.jira_client.search_issues(jql,
                                                    maxResults=5,
                                                    expand='changelog,comment,devinfo')

            logger.info("Found %d issues in epic %s", len(issues), epic_key)

            # Yield each issue with full details (similar to get_jira_ticket)
            for issue in issues:
                try:
                    # Get detailed issue information (reusing logic from get_jira_ticket)
                    issue_data = {
                        'epic':
                        epic_key,
                        'key':
                        issue.key,
                        'summary':
                        issue.fields.summary,
                        'description':
                        issue.fields.description,
                        'status':
                        issue.fields.status.name,
                        'assignee':
                        issue.fields.assignee.displayName if issue.fields.assignee else None,
                        'reporter':
                        issue.fields.reporter.displayName if issue.fields.reporter else None,
                        'created':
                        str(issue.fields.created),
                        'updated':
                        str(issue.fields.updated),
                        'issue_type':
                        issue.fields.issuetype.name,
                        'priority':
                        issue.fields.priority.name if issue.fields.priority else None,
                        'components':
                        [comp.name
                         for comp in issue.fields.components] if issue.fields.components else [],
                    }

                    # Extract development information if available
                    dev_info = self._extract_development_info(issue)
                    if dev_info:
                        issue_data['development'] = dev_info

                    yield issue_data

                except Exception as e:
                    logger.error("Error processing issue %s: %s",
                                 issue.key if hasattr(issue, 'key') else 'unknown', e)
                    # Continue with next issue instead of stopping entirely
                    continue

        except Exception as e:
            logger.error("Error fetching epic issues for %s: %s", epic_key, e)
            # Return empty iterator on error
            return iter([])


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Query Jira tickets")
    parser.add_argument("epic", help="Epic ticket ID to get all Core Server issues in that epic")

    args = parser.parse_args()

    # Setup logging with default INFO level
    setup_logging("INFO")

    # Extract configuration from environment variables
    jira_url = os.getenv('JIRA_URL')
    jira_token = os.getenv('JIRA_API_TOKEN')
    github_token = os.getenv('GITHUB_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Initialize the tool with configuration
    tool = JiraIssueIterator(jira_url=jira_url,
                             jira_token=jira_token,
                             github_token=github_token,
                             openai_api_key=openai_api_key)

    # Collect data
    logger.info("Fetching issues for epic: %s", args.epic)

    # Collect all issues from the iterator
    issues = []
    for issue in tool.get_epic_issues(args.epic):
        issues.append(issue)

    # Output results
    print(json.dumps(issues, indent=2))


if __name__ == "__main__":
    main()
