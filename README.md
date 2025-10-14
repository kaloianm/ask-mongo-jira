# AskMongoJIRA

A set of Python scripts to fetch the code changes for a given Jira Epic and analyze them with AI.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv python3-venv
source python3-venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

4. Copy `config.toml.example` to `config.toml` and customize analysis questions:
```bash
cp config.toml.example config.toml
# Edit config.toml to customize AI prompts and analysis questions
```

## Usage

### Fetch Jira Epic and store in MongoDB:
```bash
python fetch_jira_epic.py EPIC-123
```

### Sync code changes with fetched Jira Epics stored in MongoDB:
```bash
python fetch_code_changes.py
```

### Analyze code changes with OpenAI:
```bash
python analyze_code_changes.py EPIC-123 --openai-api-key your-api-key --openai-model gpt-4
```

## Scripts Overview

1. **`fetch_jira_epic.py`** - Fetches all SERVER tickets in a JIRA epic and stores them in MongoDB
2. **`fetch_code_changes.py`** - Extracts code changes from local Git repositories for commits referenced in JIRA issues
3. **`analyze_code_changes.py`** - Uses OpenAI API to analyze code changes at the JIRA issue level with MongoDB aggregation for optimal performance

## Database Performance

- **Optimized Indexing**: Single compound index on `(epic_key, issue_key)` for ticket-level queries
- **Aggregation Pipeline**: Joins issues → commits → file_changes in one query instead of multiple round-trips
- **Efficient Storage**: Normalized collections with minimal data duplication

## Collections Structure

- **`jira_issues`** - JIRA issue details with development information
- **`commits`** - Git commit metadata (author, message, stats, etc.)
- **`file_changes`** - Individual file changes per commit with diffs
- **`code_analysis`** - OpenAI analysis results for complete JIRA issues (not individual files)

## OpenAI Integration

The `analyze_code_changes.py` script uses OpenAI API v2.3.0 for code analysis. It supports:
- **Async API calls** for better performance
- **Multiple models** (gpt-4, gpt-3.5-turbo, etc.)
- **Custom base URLs** for OpenAI-compatible APIs
- **Robust error handling** with specific exception types
- **Configurable prompts** via TOML configuration file

## Configuration

The analysis behavior can be customized via the `config.toml` file:

- **Analysis Questions**: Customize the prompts sent to OpenAI for different analysis types
- **OpenAI Parameters**: Customize the system prompt

Example configuration sections:
```toml
[openai]
system_prompt = "Your custom system prompt..."

[analysis_questions.change_summary]
template = "Your custom question template with {issue_key}, {issue_summary}, {commits_summary}, etc."
```

## MCP Server Integration

The `analyze_code_changes.py` script supports integration with Model Context Protocol (MCP) servers for enhanced code analysis with additional context. Set the `MCP_SERVER_URL` environment variable to enable this feature.

### MCP Server Capabilities

Any MCP-compatible server can be used to provide:
- **Code Dependencies**: Related files and modules that interact with the changed code
- **Architecture Context**: Understanding of how the changed code fits into the larger system
- **Documentation**: Inline comments, README files, and API documentation
- **Historical Context**: Previous changes and evolution of the code
- **Test Coverage**: Related test files and coverage information

### Implementation

To implement MCP server integration:
1. Set `MCP_SERVER_URL` to your MCP server endpoint
2. The `get_mcp_context()` method will be called for each file analysis
3. Additional context will be automatically included in OpenAI prompts

### Example MCP Servers

- Custom codebase indexing servers
- GitHub-based context providers  
- IDE language servers with MCP support
- Documentation and wiki servers

## Environment Variables

See [.env.example](.env.example) for all configuration options.