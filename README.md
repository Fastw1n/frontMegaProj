## Bring Your Own Key (BYOK)

The system supports **Bring Your Own LLM Key (BYOK)**.

This allows evaluators to plug in their own API key and test the SDLC pipeline
with different LLM providers and model capacities **without changing the codebase**.

### Required GitHub Actions secrets
- `OPENROUTER_API_KEY` — your OpenRouter API key

### Optional GitHub Actions variables
- `MODEL` — model id (e.g. `openai/gpt-4o-mini`)

If `MODEL` is not set, the system falls back to a default stable model.

Note: GitHub does not copy Secrets to forks. Evaluators should set `OPENROUTER_API_KEY` in the fork settings.

### How to run
1. Configure the required secret (and optional variables).
2. Create a GitHub Issue in the repository.
3. The pipeline will automatically execute:
   **Issue → Code Agent → Pull Request → CI → Reviewer Agent**

### Example Issue
**Title:** Change homepage heading  
**Body:**  
Change the main heading text to **"Front Mega — курс по фронтенду"**.  
Only change text, do not modify styles or layout.

## Running with Docker

The Code Agent can be run locally using Docker.

### Requirements
- Docker
- Docker Compose

### Run
```bash
export OPENROUTER_API_KEY=your_key
export GITHUB_TOKEN=your_github_token
export GITHUB_REPOSITORY=owner/repo
export ISSUE_NUMBER=1

docker-compose up --build

