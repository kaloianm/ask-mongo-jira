# AskMongoJIRA
A set of Python scripts to fetch all the tickets for a given Jira Epic and their code changes and analyze them through OpenAI.

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

3. Copy `.env.example` to `.env` and fill the respective API credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

4. Edit `config.toml` if needed to customize the AI prompts and analysis questions.

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
python analyze_code_changes.py EPIC-123 \
  --openai-base-url https://api.openai.com/v1 \
  --openai-api-key <your-api-key> \
  --openai-model gpt-5
```

## Scripts Overview
1. **`fetch_jira_epic.py`** - Fetches all SERVER tickets in a JIRA epic and their commit ids (only) and stores them in MongoDB
2. **`fetch_code_changes.py`** - Extracts code changes from local Git repositories for commits referenced in JIRA issues fetched by the previous script
3. **`analyze_code_changes.py`** - Uses OpenAI API to analyze code changes at the JIRA issue level with MongoDB aggregation for optimal performance

## Collections Structure
- **`jira_epics`** - JIRA epic details
- **`jira_issues`** - JIRA issue details with development information
- **`commits`** - Git commit metadata (author, message, stats, etc.)
- **`file_changes`** - Individual file changes per commit with diffs
- **`code_analysis`** - OpenAI analysis results for complete JIRA issues (not individual files)

## Environment Variables
See [.env.example](.env.example) for all configuration options.
