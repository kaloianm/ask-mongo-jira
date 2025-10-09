#!/usr/bin/env python3
"""
fetch_code_changes - A tool for fetching code changes from local Git repositories based on git commits
stored in MongoDB by fetch_jira_epic.py.

Usage:
    python fetch_code_changes.py

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
    GIT_REPOS_PATH - Base path where Git repositories are located (optional, defaults to current directory)

The script will:
1. Query the "ask-mongo-jira" database's "jira_issues" collection for git commits
2. For each unique commit, fetch the code changes from local Git repositories using GitPython
3. Store the results in the "commits" collection

The commits collection will contain documents with:
- commit_id: The git commit SHA
- repository: The repository name
- author: Commit author information
- message: Commit message
- timestamp: Commit timestamp
- files_changed: List of files changed with their diffs
- stats: Commit statistics (additions, deletions, total changes)
- jira_issues: List of JIRA issue keys that reference this commit
- last_updated: When this record was last updated
"""

import os
import logging
import argparse
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Third-party imports
import git
from git import Repo, InvalidGitRepositoryError, GitCommandError
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables from .env file
load_dotenv()

# Setup the global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


class GitCodeFetcher:
    """Main class for fetching code changes from local Git repositories based on MongoDB commit data"""

    def __init__(self, mongodb_url: str = None, git_repos_path: str = None):
        """Initialize with configuration parameters"""
        # Store configuration
        self.mongodb_url = mongodb_url
        self.git_repos_path = git_repos_path

        # Log configuration
        logger.info("Initializing GitCodeFetcher with configuration:")
        logger.info("  mongodb_url: %s", "***" if mongodb_url else None)
        logger.info("  git_repos_path: %s", self.git_repos_path)

        # Initialize MongoDB client
        self.mongodb_client = AsyncIOMotorClient(self.mongodb_url)
        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]
        self.commits_collection = self.db["commits"]

        # Cache for opened repositories
        self.repo_cache = {}

    def _extract_owner_repo(self, commit_url):
        # Remove trailing slash and split by '/'
        parts = commit_url.rstrip('/').split('/')

        # Find the index of 'commit'
        try:
            commit_index = parts.index('commit')
        except ValueError as exc:
            raise ValueError("Invalid GitHub commit URL: 'commit' segment not found") from exc

        # Ensure there are enough parts before 'commit' for owner and repo
        if commit_index < 2:
            raise ValueError("Invalid GitHub commit URL: not enough segments before 'commit'")

        # Extract owner and repo (second and third parts before 'commit')
        owner = parts[commit_index - 2]
        repo = parts[commit_index - 1]

        # Validate owner and repo are non-empty
        if not owner or not repo:
            raise ValueError("Invalid GitHub commit URL: owner or repo is empty")

        return owner, repo

    def _get_repo_path(self, owner: str, repo: str) -> Optional[Path]:
        """
        Get the local path to a Git repository.
        """
        path = Path(self.git_repos_path) / Path(owner) / Path(repo)
        if path.exists() and (path / '.git').exists():
            logger.debug("Found repository at: %s", path)
            return path

        logger.warning("Could not find local repository for: %s/%s", owner, repo)
        return None

    def _get_repo(self, owner: str, repo: str) -> Optional[Repo]:
        """
        Get a Git repository object, using cache if available.
        
        Args:
            repository_name: Repository name (e.g., "owner/repo")
            
        Returns:
            Git Repo object or None if not found
        """
        repository_name = f"{owner}/{repo}"
        if repository_name in self.repo_cache:
            return self.repo_cache[repository_name]

        repo_path = self._get_repo_path(owner, repo)
        if not repo_path:
            return None

        try:
            repo_obj = Repo(str(repo_path))
            self.repo_cache[repository_name] = repo_obj
            logger.debug("Opened repository: %s", repo_path)
            return repo_obj
        except InvalidGitRepositoryError:
            logger.error("Invalid Git repository at: %s", repo_path)
            return None

    async def fetch_commit_details(self, owner: str, repo: str,
                                   commit_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed commit information from local Git repository.
        
        Args:
            owner/repo: Repository name
            commit_id: The git commit SHA
            
        Returns:
            Dictionary with commit details or None if not found
        """
        try:
            repo_obj = self._get_repo(owner, repo)
            if not repo_obj:
                logger.warning("Could not find local repository for: %s/%s", owner, repo)
                return None

            logger.debug("Fetching commit %s from repository %s/%s", commit_id, owner, repo)

            try:
                commit = repo_obj.commit(commit_id)
            except GitCommandError as e:
                logger.warning("Commit %s not found in repository %s/%s: %s", commit_id, owner,
                               repo, e)
                return None

            # Get the diff for this commit
            if commit.parents:
                # Compare with first parent
                diff = commit.parents[0].diff(commit, create_patch=True)
            else:
                # Initial commit - compare with empty tree
                diff = commit.diff(git.NULL_TREE, create_patch=True)

            # Extract file changes
            files_changed = []

            for diff_item in diff:
                # Determine change type
                if diff_item.new_file:
                    status = 'added'
                elif diff_item.deleted_file:
                    status = 'removed'
                elif diff_item.renamed_file:
                    status = 'renamed'
                else:
                    status = 'modified'

                # Get file paths
                filename = diff_item.b_path or diff_item.a_path
                previous_filename = diff_item.a_path if diff_item.renamed_file else None

                # Count line changes from diff
                patch_text = diff_item.diff.decode('utf-8',
                                                   errors='ignore') if diff_item.diff else ""
                additions = patch_text.count('\n+') - patch_text.count('\n+++')
                deletions = patch_text.count('\n-') - patch_text.count('\n---')

                # Ensure non-negative counts
                additions = max(0, additions)
                deletions = max(0, deletions)

                file_info = {
                    'filename': filename,
                    'status': status,
                    'additions': additions,
                    'deletions': deletions,
                    'changes': additions + deletions,
                    'patch': patch_text if patch_text else None,
                }

                if previous_filename:
                    file_info['previous_filename'] = previous_filename

                files_changed.append(file_info)

            # Calculate total stats
            total_additions = sum(f['additions'] for f in files_changed)
            total_deletions = sum(f['deletions'] for f in files_changed)

            # Prepare commit data
            commit_data = {
                'commit_id': commit_id,
                'repository': f"{owner}/{repo}",
                'author': {
                    'name': commit.author.name,
                    'email': commit.author.email,
                    'date': commit.authored_datetime.isoformat(),
                },
                'committer': {
                    'name': commit.committer.name,
                    'email': commit.committer.email,
                    'date': commit.committed_datetime.isoformat(),
                },
                'message': commit.message,
                'sha': commit.hexsha,
                'files_changed': files_changed,
                'stats': {
                    'total': total_additions + total_deletions,
                    'additions': total_additions,
                    'deletions': total_deletions,
                },
                'parents': [parent.hexsha for parent in commit.parents],
            }

            logger.info("Successfully fetched commit %s from %s/%s (%d files changed)",
                        commit_id[:8], owner, repo, len(files_changed))
            return commit_data

        except GitCommandError as e:
            logger.error("Git command error fetching commit %s: %s", commit_id, e)
            return None

    async def store_commit_in_mongodb(self, commit_data: Dict[str, Any], jira_issues: List[str]):
        """
        Store or update a commit in the commits collection.
        Args:
            commit_data: Detailed commit information from local Git
            jira_issues: List of JIRA issue keys that reference this commit
        """
        commit_data['jira_issues'] = jira_issues
        commit_data['last_updated'] = datetime.datetime.now(datetime.timezone.utc)

        # Use commit_id as the unique identifier
        filter_query = {"commit_id": commit_data["commit_id"]}

        # Upsert the commit data
        result = await self.commits_collection.replace_one(filter_query, commit_data, upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Commit %s %s in commits collection", commit_data["commit_id"][:8], action)

    async def process_jira_issues(self) -> None:
        """
        Main processing function: iterate through JIRA issues and fetch commit details from local Git repositories.
        """
        # Query all JIRA issues that have development info with commits
        cursor = self.jira_issues_collection.find(
            {"development.commits": {
                "$exists": True,
                "$ne": []
            }})

        processed_commits = 0
        skipped_commits = 0
        errors = 0
        issue_count = 0

        async for issue in cursor:
            issue_count += 1
            issue_key = issue.get('key', 'unknown')
            dev_info = issue.get('development', {})
            commits = dev_info.get('commits', [])

            logger.info("Processing issue %s with %d commits", issue_key, len(commits))

            for commit in commits:
                commit_id = commit.get('id')
                if not commit_id:
                    logger.warning("Skipping commit with no ID in issue %s", issue_key)
                    continue

                owner, repo = self._extract_owner_repo(commit.get('url'))
                logger.debug("Processing commit %s from %s/%s (issue: %s)", commit_id[:8], owner,
                             repo, issue_key)

                # Check if we already have this commit in our collection
                existing = await self.commits_collection.find_one({"commit_id": commit_id})
                if existing:
                    # Update the existing commit to include this JIRA issue if not already present
                    if issue_key not in existing.get('jira_issues', []):
                        await self.commits_collection.update_one({"commit_id": commit_id}, {
                            "$addToSet": {
                                "jira_issues": issue_key
                            },
                            "$set": {
                                "last_updated": datetime.datetime.utcnow()
                            }
                        })
                        logger.debug("Added issue %s to existing commit %s", issue_key,
                                     commit_id[:8])
                    else:
                        logger.debug("Commit %s already has issue %s, skipping", commit_id[:8],
                                     issue_key)
                    skipped_commits += 1
                    continue

                # Fetch detailed commit information from local Git repository
                commit_data = await self.fetch_commit_details(owner, repo, commit_id)

                if commit_data:
                    # Store in MongoDB with this JIRA issue
                    await self.store_commit_in_mongodb(commit_data, [issue_key])
                    processed_commits += 1
                    logger.info("Successfully processed commit %s for issue %s", commit_id[:8],
                                issue_key)
                else:
                    logger.warning("Could not fetch commit %s from local Git for issue %s",
                                   commit_id, issue_key)
                errors += 1

        logger.info(
            "Processing complete: %d issues processed, %d commits processed, %d skipped, %d errors",
            issue_count, processed_commits, skipped_commits, errors)

    async def setup_database_indexes(self) -> None:
        """Set up database indexes for efficient querying on the commits collection"""
        # Create index on commit_id for uniqueness and efficient queries
        await self.commits_collection.create_index("commit_id",
                                                   unique=True,
                                                   name="commit_id_unique")

        # Create index on jira_issues for finding commits by JIRA issue
        await self.commits_collection.create_index("jira_issues", name="jira_issues_index")

        logger.info("Database indexes created successfully for commits collection")

    async def close_mongodb_connection(self) -> None:
        """Close MongoDB connection"""
        if self.mongodb_client:
            self.mongodb_client.close()
            logger.info("MongoDB connection closed")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fetch code changes from local Git repositories for commits stored in MongoDB")
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Extract configuration from environment variables
    mongodb_url = os.getenv('MONGODB_URL')
    git_repos_path = os.getenv('GIT_REPOS_PATH')

    if not mongodb_url:
        logger.error("MongoDB URL is required. Set the MONGODB_URL environment variable")
        return

    if not git_repos_path:
        logger.error(
            "Git repositories path is required. Set the GIT_REPOS_PATH environment variable")
        return

    # Initialize the fetcher
    fetcher = GitCodeFetcher(mongodb_url=mongodb_url, git_repos_path=git_repos_path)

    try:
        # Set up database indexes
        await fetcher.setup_database_indexes()

        # Process JIRA issues and their commits
        logger.info("Starting to fetch code changes from local Git repositories...")
        await fetcher.process_jira_issues()
        logger.info("Code change fetching completed successfully")

    finally:
        # Clean up MongoDB connection
        await fetcher.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
