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
- **`jira_epics`** - Contains important Jira Epic details for each of the analyzed projects, most notably its start_date, end_date and name (summary). Entirely copied from Jira, without any transformation.
- **`jira_issues`** - Contains the Jira issue details along with development information, representing the commit ids that went into solving this issue. Entirely copied from Jira, without any transformation. Foreign keys are `epic` that links back into `jira_epics` and `development.commits.id` that links into `commits.commit_id`.
- **`commits`** - For each issue commit from `jira_issues.development.commits` contains detailed commit information obtained from Git itself, such as the commit metadata (author, message, stats, etc.). Foreign keys are `jira_issues` that link back into `jira_issues.issue` and `commit_id` that links into `file_changes.commit_id`.
- **`file_changes`** - For each individual file changed by a commit from `commits.commit_id` contains the actual diff (in long and short form). Foreign key is `commit_id` that links back into `commits.commit_id`.
- **`code_analysis`** - For all `file_changes` belonging to the same `commit_id` contains OpenAI analysis results. Foreign keys are `commit_ids` that link back into `commits` and `epic_key`/`issue_key` that link back into `jira_epics`/`jira_issues`.

## Environment Variables
See [.env.example](.env.example) for all configuration options.
