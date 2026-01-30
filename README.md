## Bring Your Own Key (BYOK)

The system supports Bring Your Own LLM Key.

To run the Coding Agent, configure the following GitHub Actions secrets:

- OPENROUTER_API_KEY – your OpenRouter API key

Optionally configure variables:
- MODEL – model id (e.g. openai/gpt-4o-mini)

This allows evaluators to test the pipeline with different LLM providers
and model capacities without changing the codebase.
