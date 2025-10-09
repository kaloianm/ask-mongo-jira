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

## Usage

### Fetch Jira Epic and store in MongoDB:
```bash
python fetch_jira_epic.py EPIC-123
```

### Sync code changes with fetched Jira Epics stored in MongoDB:
```bash
python fetch_code_changes.py
```

## Environment Variables

See [.env](.env).