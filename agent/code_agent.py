import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

from github import Auth, Github
from openai import OpenAI


# --- Models allowlist (free) ---
ALLOWLIST_DEFAULT = [
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "tngtech/deepseek-r1t2-chimera:free",
]


# --- Repo scanning settings ---
EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
    "__pycache__",
    "coverage",
    ".idea",
    ".vscode",
}
ALLOWED_EXTS = {
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".html",
    ".json",
    ".md",
    ".yml",
    ".yaml",
}
MAX_FILES_IN_CONTEXT = 8
MAX_CHARS_PER_FILE = 7000  # to keep prompts bounded


def sh(cmd: str) -> str:
    print(f"$ {cmd}")
    out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    print(out)
    return out


def try_sh(cmd: str) -> Tuple[bool, str]:
    try:
        return True, sh(cmd)
    except subprocess.CalledProcessError as e:
        print(e.output)
        return False, e.output


def iter_models() -> List[str]:
    env_model = (os.getenv("MODEL") or "").strip()
    models = []
    if env_model:
        models.append(env_model)
    for m in ALLOWLIST_DEFAULT:
        if m not in models:
            models.append(m)
    return models


def openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    # OpenRouter is OpenAI-compatible; base_url is required
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def looks_retryable_error(msg: str) -> bool:
    s = msg.lower()
    markers = [
        "rate limit",
        "rate-limited",
        "temporarily rate-limited",
        "provider returned error",
        "no endpoints found",
        "insufficient credits",
        "no models provided",
        "error code: 429",
        "error code: 402",
        "error code: 404",
    ]
    return any(m in s for m in markers)


def gather_repo_files() -> List[Path]:
    root = Path(".").resolve()
    result: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        # exclude dirs
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        # skip huge files
        try:
            if p.stat().st_size > 300_000:
                continue
        except OSError:
            continue
        result.append(rel)
    return result


def score_file(rel: Path, issue_text: str) -> int:
    """Simple heuristic scoring for relevance."""
    text = issue_text.lower()
    name = str(rel).lower()
    score = 0

    keywords = re.findall(r"[a-zA-Z0-9_/-]{3,}", text)
    for k in keywords[:80]:
        if k in name:
            score += 5

    # prefer typical frontend entrypoints
    boosts = [
        "src/app",
        "src/main",
        "src/index",
        "src/pages",
        "src/components",
        "app.tsx",
        "app.jsx",
        "main.tsx",
        "main.jsx",
        "index.tsx",
        "index.jsx",
        "package.json",
        "vite.config",
        "next.config",
    ]
    for b in boosts:
        if b in name:
            score += 10

    return score


def read_file(rel: Path) -> str:
    p = Path(rel)
    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_CHARS_PER_FILE:
        content = content[:MAX_CHARS_PER_FILE] + "\n\n/* ...truncated... */\n"
    return content


def build_context(issue_title: str, issue_body: str) -> str:
    issue_text = f"{issue_title}\n{issue_body}".strip()
    all_files = gather_repo_files()

    ranked = sorted(all_files, key=lambda r: score_file(r, issue_text), reverse=True)
    chosen = ranked[:MAX_FILES_IN_CONTEXT]

    parts = []
    parts.append("=== REPOSITORY FILE CONTEXT (selected) ===\n")
    for rel in chosen:
        parts.append(f"\n--- FILE: {rel} ---\n")
        parts.append(read_file(rel))
        parts.append("\n")
    return "".join(parts)


def extract_diff(text: str) -> str:
    """
    Expect the model to output a unified diff in a fenced code block:
    ```diff
    ...
    ```
    """
    m = re.search(r"```diff\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # try non-fenced diff
        m2 = re.search(r"(^diff --git .*?$.*)", text, re.DOTALL | re.MULTILINE)
        if m2:
            return m2.group(1).strip()
        raise ValueError("No diff found in model output. Expected ```diff ...```.")
    return m.group(1).strip()


def call_model_for_diff(model: str, issue_title: str, issue_body: str, context: str) -> str:
    client = openrouter_client()
    prompt = f"""
You are a senior software engineer. Produce a minimal, correct change for the task.

Task (GitHub Issue):
Title: {issue_title}
Body:
{issue_body}

Rules:
- Output ONLY a unified git diff in a fenced block: ```diff ... ```
- Make minimal changes.
- Do not touch node_modules, dist, build, .git.
- If unsure, change the most obvious frontend entrypoint (App / main page).
- Keep existing style; avoid large reformatting.

{context}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write code changes as unified diffs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    issue_number_str = os.getenv("ISSUE_NUMBER")
    base_branch = (os.getenv("BASE_BRANCH") or "main").strip()

    if not token or not repo_name or not issue_number_str:
        raise ValueError("Missing env vars: GITHUB_TOKEN, GITHUB_REPOSITORY, ISSUE_NUMBER")

    issue_number = int(issue_number_str)

    # GitHub API
    auth = Auth.Token(token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(number=issue_number)

    issue_title = issue.title or ""
    issue_body = issue.body or ""

    branch = f"issue-{issue_number}"

    # 1) Create branch from base
    base = repo.get_branch(base_branch)
    try:
        repo.create_git_ref(ref=f"refs/heads/{branch}", sha=base.commit.sha)
        print(f"Created branch: {branch}")
    except Exception:
        print("Branch already exists (ok)")

    # 2) Prepare git
    sh("git config user.email 'actions@github.com'")
    sh("git config user.name 'github-actions[bot]'")
    sh(f"git fetch origin {branch}:{branch} || true")
    sh(f"git checkout {branch}")
    sh(f"git remote set-url origin https://x-access-token:{token}@github.com/{repo_name}.git")

    # Clean caches
    sh("find . -type d -name __pycache__ -prune -exec rm -rf {} + || true")
    sh("find . -type f -name '*.pyc' -delete || true")

    # 3) Build lightweight context
    context = build_context(issue_title, issue_body)

    # 4) Ask model for diff with fallback models
    last_err = None
    last_raw = None
    diff_text = None

    for model in iter_models():
        try:
            print(f"[LLM] Trying model: {model}")
            raw = call_model_for_diff(model, issue_title, issue_body, context)
            last_raw = raw
            diff_text = extract_diff(raw)
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"[LLM] Model failed: {model} | error: {msg}")
            if looks_retryable_error(msg) or looks_retryable_error(last_raw or ""):
                continue
            # non-retryable parsing errors etc — try next model anyway, but keep going
            continue

    if not diff_text:
        raise RuntimeError(f"All models failed. Last error: {last_err}\nLast output: {last_raw}")

    print("=== DIFF (from model) ===")
    print(diff_text)

    # 5) Apply patch
    # Write patch to file and apply
    patch_file = Path("agent_patch.diff")
    patch_file.write_text(diff_text, encoding="utf-8")

    ok, out = try_sh("git apply --whitespace=fix agent_patch.diff")
    if not ok:
        # try 3-way apply
        ok2, out2 = try_sh("git apply --3way --whitespace=fix agent_patch.diff")
        if not ok2:
            raise RuntimeError("Failed to apply patch. Aborting.")

    # 6) Run basic frontend checks (best effort)
    # Note: If your repo uses yarn/pnpm, adjust later.
    try_sh("npm ci")
    try_sh("npm run lint --if-present")
    try_sh("npm test --if-present")
    try_sh("npm run build --if-present")

    # Clean caches again
    sh("find . -type d -name __pycache__ -prune -exec rm -rf {} + || true")
    sh("find . -type f -name '*.pyc' -delete || true")

    # 7) Commit only if there are changes
    status = subprocess.check_output("git status --porcelain", shell=True, text=True).strip()
    print("$ git status --porcelain")
    print(status)

    if not status:
        print("No changes detected. Not creating PR.")
        return

    sh("git add -A")
    sh(f"git commit -m 'Auto-fix issue #{issue_number}'")
    sh(f"git push -u origin {branch}")

    # 8) Create PR if not exists
    pr_title = f"Auto-fix for issue #{issue_number}: {issue_title}".strip()
    pr_body = (
        "This PR was automatically generated by Code Agent (fallback mode, no tools).\n\n"
        "It was produced as a unified diff and applied via git apply.\n"
    )

    try:
        repo.create_pull(
            title=pr_title,
            body=pr_body[:65000],
            head=branch,
            base=base_branch,
        )
        print("Pull Request created")
    except Exception as e:
        print(f"PR already exists or could not be created: {e}")


if __name__ == "__main__":
    main()
