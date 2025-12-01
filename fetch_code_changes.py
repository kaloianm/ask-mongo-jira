#!/usr/bin/env python3
"""
fetch_code_changes - Tool for fetching code changes from local Git repositories based on git
                     commits stored in MongoDB by fetch_jira_epic.py.

Usage:
    python3 fetch_code_changes.py --help
    python3 fetch_code_changes.py --epic SPM-1234

Environment Variables:
    MONGODB_URL - MongoDB connection URL (required)
    GIT_REPOS_PATH - Base path where Git repositories are located (required)

The script will:
1. Query the "ask-mongo-jira" database's "jira_issues" collection for git commits
2. Optionally filter by a specific epic using the --epic argument
3. For each unique commit, fetch the code changes from local Git repositories using GitPython
4. Store the results in two collections: "commits" and "file_changes"
"""

import os
import logging
import argparse
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from urllib.parse import urlparse

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
    """
    Setup logging configuration
    """
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


class GitCodeFetcher:
    """
    Main class for fetching code changes from local Git repositories based on MongoDB commit data
    """

    def __init__(self, mongodb_url: str, git_repos_path: str):
        """
        Initialize with configuration parameters
        """
        # Store configuration
        self.git_repos_path = git_repos_path
        self.git_repos_cache = {}

        # Log configuration
        logger.info("Initializing GitCodeFetcher with configuration:")
        logger.info("  mongodb_url: %s", "***" if mongodb_url else None)
        logger.info("  git_repos_path: %s", self.git_repos_path)

        # Initialize MongoDB client
        self.mongodb_client = AsyncIOMotorClient(mongodb_url)
        self.db = self.mongodb_client["ask-mongo-jira"]
        self.jira_issues_collection = self.db["jira_issues"]
        self.commits_collection = self.db["commits"]
        self.file_changes_collection = self.db["file_changes"]

    def _extract_owner_repo(self, commit_url):
        """
        Extract owner and repository name from a GitHub commit URL using urlparse
        """
        parsed = urlparse(commit_url)
        path_parts = parsed.path.strip('/').split('/')

        # We need at least 2 parts: owner and repo
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub commit URL: insufficient path components")

        owner = path_parts[0]
        repo = path_parts[1]

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
        if repository_name in self.git_repos_cache:
            return self.git_repos_cache[repository_name]

        repo_path = self._get_repo_path(owner, repo)
        if not repo_path:
            return None

        try:
            repo_obj = Repo(str(repo_path))
            self.git_repos_cache[repository_name] = repo_obj
            logger.debug("Opened repository: %s", repo_path)
            return repo_obj
        except InvalidGitRepositoryError:
            logger.error("Invalid Git repository at: %s", repo_path)
            return None

    def _is_commit_on_master_branch(self, repo_obj: Repo, commit_id: str) -> bool:
        """
        Check if a commit is on the master/main branch and not a cherry-pick.

        Args:
            repo_obj: Git repository object
            commit_id: The commit ID to check

        Returns:
            True if commit is on master/main branch and not a cherry-pick
        """
        # Get the commit object
        commit = repo_obj.commit(commit_id)

        # Check if commit is reachable from master branch
        master_branches = ['master']
        is_on_master = False

        for branch_name in master_branches:
            try:
                # Check if the branch exists
                branch = repo_obj.heads[branch_name]
                # Check if commit is an ancestor of this branch
                if repo_obj.is_ancestor(commit, branch.commit):
                    is_on_master = True
                    break
            except (IndexError, git.exc.GitCommandError):
                # Branch doesn't exist or other git error
                continue

        if not is_on_master:
            logger.debug("Commit %s is not on master/main branch", commit_id[:8])
            return False

        # Check if this commit is a cherry-pick by examining the commit message
        # Cherry-picks typically have "(cherry picked from commit <sha>)" in the message
        if "(cherry picked from commit" in commit.message.lower():
            logger.debug("Commit %s appears to be a cherry-pick based on commit message",
                         commit_id[:8])
            return False

        # Additional check: if commit has multiple parents with the same tree (indicating cherry-pick)
        # This is more complex and may have false positives, so we'll skip it for now

        return True

    async def _fetch_commit_details(self, owner: str, repo: str, commit_id: str) -> Optional[tuple]:
        """
        Fetch detailed commit information from local Git repository.
        
        Args:
            owner/repo: Repository name
            commit_id: The git commit SHA
            
        Returns:
            Tuple of (commit_data, files_changed) or None if not found
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
            except ValueError as e:
                logger.warning("Invalid commit ID %s in repository %s/%s: %s", commit_id, owner,
                               repo, e)
                return None

            # Check if commit is on master branch and not a cherry-pick
            if not self._is_commit_on_master_branch(repo_obj, commit_id):
                logger.info(
                    "Skipping commit %s: %s ... as it's not on master branch or is a cherry-pick",
                    commit_id[:8],
                    commit.message.splitlines()[0][:24])
                return None

            # Get the diff for this commit with extended context using Git's unified context option
            if commit.parents:
                # Compare with first parent, including 50 lines of context before and after changes
                diff = commit.parents[0].diff(commit, create_patch=True, unified=50)
                short_diff = commit.parents[0].diff(commit, create_patch=True)
            else:
                # Initial commit - compare with empty tree, including 50 lines of context
                diff = commit.diff(git.NULL_TREE, create_patch=True, unified=50)
                short_diff = commit.diff(git.NULL_TREE, create_patch=True)

            # Extract file changes
            files_changed = []

            for diff_item, short_diff_item in zip(diff, short_diff):
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
                short_patch_text = short_diff_item.diff.decode(
                    'utf-8', errors='ignore') if short_diff_item.diff else ""
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
                    'short_patch': short_patch_text if short_patch_text else None
                }

                if previous_filename:
                    file_info['previous_filename'] = previous_filename

                files_changed.append(file_info)

            # Calculate total stats
            total_additions = sum(f['additions'] for f in files_changed)
            total_deletions = sum(f['deletions'] for f in files_changed)

            # Prepare commit data (without files_changed)
            commit_data = {
                'commit_id': commit_id,
                'repository': f"{owner}/{repo}",
                'author': {
                    'name': commit.author.name,
                    'email': commit.author.email,
                    'date': commit.authored_datetime,
                },
                'committer': {
                    'name': commit.committer.name,
                    'email': commit.committer.email,
                    'date': commit.committed_datetime,
                },
                'message': commit.message,
                'sha': commit.hexsha,
                'stats': {
                    'total': total_additions + total_deletions,
                    'additions': total_additions,
                    'deletions': total_deletions,
                },
                'parents': [parent.hexsha for parent in commit.parents],
            }

            logger.info("Successfully fetched commit %s from %s/%s (%d files changed)",
                        commit_id[:8], owner, repo, len(files_changed))
            return commit_data, files_changed

        except GitCommandError as e:
            logger.error("Git command error fetching commit %s: %s", commit_id, e)
            return None

    async def _store_commit_in_mongodb(self, commit_data: Dict[str, Any], jira_issues: List[str],
                                       scan_version: int):
        """
        Store or update a commit in the commits collection.
        Args:
            commit_data: Detailed commit information from local Git
            jira_issues: List of JIRA issue keys that reference this commit
        """
        commit_data['jira_issues'] = jira_issues
        commit_data['version'] = scan_version
        commit_data['last_updated'] = datetime.datetime.now(datetime.timezone.utc)

        # Use commit_id as the unique identifier
        filter_query = {"commit_id": commit_data["commit_id"]}

        # Upsert the commit data
        result = await self.commits_collection.replace_one(filter_query, commit_data, upsert=True)

        action = "inserted" if result.upserted_id else "updated"
        logger.debug("Commit %s %s in commits collection", commit_data["commit_id"][:8], action)

    async def _store_file_changes_in_mongodb(self, commit_id: str, repository: str,
                                             files_changed: List[Dict[str,
                                                                      Any]], scan_version: int):
        """
        Store file changes in the file_changes collection.
        Args:
            commit_id: The git commit SHA
            repository: Repository name
            files_changed: List of file change information
        """
        if not files_changed:
            return

        current_time = datetime.datetime.now(datetime.timezone.utc)

        # First, remove any existing file changes for this commit
        await self.file_changes_collection.delete_many({"commit_id": commit_id})

        # Prepare file change documents
        file_change_docs = []
        for file_change in files_changed:
            doc = {
                'commit_id': commit_id,
                'repository': repository,
                'version': scan_version,
                'last_updated': current_time,
                **file_change  # Include all file change fields
            }
            file_change_docs.append(doc)

        # Insert all file changes
        if file_change_docs:
            result = await self.file_changes_collection.insert_many(file_change_docs)
            logger.debug("Inserted %d file changes for commit %s", len(result.inserted_ids),
                         commit_id[:8])

    async def _mark_jira_issue_fetched_in_mongodb(self, epic_key: str, issue_key: str,
                                                  scan_version: str):
        """
        Mark a JIRA issue as fetched by updating it with a fetch_version field.

        Args:
            epic_key: The epic ticket ID
            issue_key: The JIRA issue key
            scan_version: The scan version to mark this issue with
        """
        filter_query = {"epic": epic_key, "issue": issue_key}
        update_query = {
            "$set": {
                "fetch_version": scan_version,
                "fetch_timestamp": datetime.datetime.now(datetime.timezone.utc)
            }
        }

        result = await self.jira_issues_collection.update_one(filter_query, update_query)

        if result.matched_count > 0:
            logger.debug("Marked issue %s in epic %s as fetched with version %s", issue_key,
                         epic_key, scan_version)
        else:
            logger.warning("Could not find issue %s in epic %s to mark as fetched", issue_key,
                           epic_key)

    async def process_jira_issues(self, scan_version: int, epic_key: str = None) -> None:
        """
        Main processing function: iterate through JIRA issues and fetch commit details from local
        Git repositories.
        
        Args:
            scan_version: Version number for this scan
            epic_key: Optional epic key to filter issues (e.g., "SPM-1234")
        """
        # Build query to find JIRA issues that have development info with commits
        query = {"development.commits": {"$exists": True, "$ne": []}}

        # Add epic filter if specified
        if epic_key:
            query["epic"] = epic_key
            logger.info("Filtering issues by epic: %s", epic_key)

        cursor = self.jira_issues_collection.find(query)

        processed_commits = 0
        skipped_commits = 0
        errors = 0
        issue_count = 0

        async for issue in cursor:
            issue_count += 1
            epic_key = issue['epic']
            issue_key = issue['issue']
            dev_info = issue['development']
            commits = dev_info['commits']

            logger.info("Processing issue %s/%s with %d commits", epic_key, issue_key, len(commits))

            for commit in commits:
                commit_id = commit['id']
                if not commit_id:
                    logger.warning("Skipping commit with no ID in issue %s", issue_key)
                    continue

                owner, repo = self._extract_owner_repo(commit['url'])
                logger.debug("Processing commit %s from %s/%s (issue: %s)", commit_id[:8], owner,
                             repo, issue_key)

                # Check if we already have this commit in our collection
                existing = await self.commits_collection.find_one({"commit_id": commit_id})
                if existing:
                    # Update the existing commit to include this JIRA issue if not already present
                    if issue_key not in existing['jira_issues']:
                        await self.commits_collection.update_one({"commit_id": commit_id}, {
                            "$addToSet": {
                                "jira_issues": issue_key
                            },
                            "$set": {
                                "last_updated": datetime.datetime.now(datetime.timezone.utc)
                            }
                        })
                        logger.debug("Added issue %s to existing commit %s", issue_key,
                                     commit_id[:8])
                    else:
                        logger.debug("Commit %s already includes issue %s", commit_id[:8],
                                     issue_key)

                    # Skip processing if the existing commit entry is already at the current scan
                    # version
                    if ('version' in existing) and existing['version'] == scan_version:
                        skipped_commits += 1
                        continue

                # Fetch detailed commit information from local Git repository
                commit_result = await self._fetch_commit_details(owner, repo, commit_id)

                if commit_result:
                    commit_data, files_changed = commit_result
                    # Store commit data and file changes in separate collections
                    await self._store_commit_in_mongodb(commit_data, [issue_key], scan_version)
                    await self._store_file_changes_in_mongodb(commit_id, f"{owner}/{repo}",
                                                              files_changed, scan_version)
                    processed_commits += 1
                    logger.info("Successfully processed commit %s for issue %s", commit_id[:8],
                                issue_key)
                else:
                    errors += 1

                # Mark the JIRA issue as fetched in the jira_issues collection
                await self._mark_jira_issue_fetched_in_mongodb(epic_key, issue_key, scan_version)

        logger.info(
            "Processing complete: %d issues processed, %d commits processed, %d skipped, %d errors",
            issue_count, processed_commits, skipped_commits, errors)

    async def setup_database_indexes(self) -> None:
        """
        Set up database indexes for efficient querying on both commits and file_changes collections
        """
        # Indexes for commits collection
        await self.commits_collection.create_index("commit_id",
                                                   unique=True,
                                                   name="commit_id_unique")

        # Create index on jira_issues for finding commits by JIRA issue
        await self.commits_collection.create_index("jira_issues", name="jira_issues_index")

        # Indexes for file_changes collection
        await self.file_changes_collection.create_index("commit_id", name="commit_id_index")

        # Compound index for efficient querying by commit and filename
        await self.file_changes_collection.create_index([("commit_id", 1), ("filename", 1)],
                                                        name="commit_filename_index")

        # Index on repository for filtering by repository
        await self.file_changes_collection.create_index("repository", name="repository_index")

        logger.info(
            "Database indexes created successfully for commits and file_changes collections")

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
    parser = argparse.ArgumentParser(
        description="Fetch code changes from local Git repositories for commits stored in MongoDB")
    parser.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Set the logging level")
    parser.add_argument(
        "--epic",
        help="Epic ticket ID to filter issues (e.g., SPM-1234). If not provided, processes all epics"
    )

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
        await fetcher.process_jira_issues(scan_version=1, epic_key=args.epic)
        logger.info("Code change fetching completed successfully")

    finally:
        await fetcher.close_mongodb_connection()


if __name__ == "__main__":
    asyncio.run(main())
