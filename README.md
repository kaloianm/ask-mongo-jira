# AskMongoJIRA

A simple Python tool for querying Jira tickets, GitHub pull requests, and analyzing them with AI.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

### Query a Jira ticket:
```bash
python main.py --jira-ticket PROJ-123
```

### Query a GitHub PR:
```bash
python main.py --github-pr 45 --repo owner/repo-name
```

### Ask AI about the data:
```bash
python main.py --jira-ticket PROJ-123 --question "What is this ticket about?"
python main.py --github-pr 45 --repo owner/repo --question "What changes were made?"
```

### Output as JSON:
```bash
python main.py --jira-ticket PROJ-123 --output json
```

## Environment Variables

- `JIRA_URL`: Your Jira instance URL
- `JIRA_USERNAME`: Your Jira username/email
- `JIRA_API_TOKEN`: Your Jira API token
- `GITHUB_TOKEN`: Your GitHub personal access token
- `OPENAI_API_KEY`: Your OpenAI API key