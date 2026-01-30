import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from github import Auth, Github
from openai import OpenAI


# --- Default model (stable, BYOK-friendly) ---
DEFAULT_MODEL = "openai/gpt-4o-mini"

# --- Fallback models (free) ---
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
MAX_CHARS_PER_FILE = 7000
MAX_TOTAL_CONTEXT_CHARS = 45_000  # общий лимит, чтобы не улетать в токены


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
    """
    BYOK logic:
    - If MODEL env is set -> try it first
    - Else -> try DEFAULT_MODEL (stable)
    - Then fallback allowlist (free)
    """
    env_model = (os.getenv("MODEL") or "").strip()
    models: List[str] = []

    if env_model:
        models.append(env_model)
    else:
        models.append(DEFAULT_MODEL)

    for m in ALLOWLIST_DEFAULT:
        if m not in models:
            models.append(m)

    return models


def openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Configure it in GitHub Actions Secrets to use BYOK."
        )

    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def looks_retryable_error(msg: str) -> bool:
    s = (msg or "").lower()
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
        "tool use",
        "support tool use",
    ]
    return any(m in s for m in markers)


def gather_repo_files() -> List[Path]:
    root = Path(".").resolve()
    result: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        try:
            if p.stat().st_size > 300_000:
                continue
        except OSError:
            continue
        result.append(rel)
    return result


def score_file(rel: Path, issue_text: str) -> int:
    text = issue_text.lower()
    name = str(rel).lower()
    score = 0

    keywords = re.findall(r"[a-zA-Z0-9_/-]{3,}", text)
    for k in keywords[:80]:
        if k in name:
            score += 5

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

    parts: List[str] = ["=== REPOSITORY FILE CONTEXT (selected) ===\n"]
    total = 0

    for rel in chosen:
        block = f"\n--- FILE: {rel} ---\n{read_file(rel)}\n"
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "".join(parts)


def extract_diff(text: str) -> str:
    m = re.search(r"```diff\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
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
- Output ONLY a valid git-style unified diff produced by `git diff`
- The diff MUST start with: diff --git a/... b/...
- Include --- a/... and +++ b/... lines and @@ hunks
- Do not include any commentary outside the ```diff``` block


{context}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output ONLY unified diffs in ```diff``` fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def comment_issue(repo, issue_number: int, body: str) -> None:
    try:
        issue = repo.get_issue(number=issue_number)
        issue.create_comment(body[:65000])
    except Exception as e:
        print(f"Failed to comment issue: {e}")


def main() -> None:
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    issue_number_str = os.getenv("ISSUE_NUMBER")
    base_branch = (os.getenv("BASE_BRANCH") or "main").strip()

    if not token or not repo_name or not issue_number_str:
        raise ValueError("Missing env vars: GITHUB_TOKEN, GITHUB_REPOSITORY, ISSUE_NUMBER")

    issue_number = int(issue_number_str)

    auth = Auth.Token(token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)

    issue = repo.get_issue(number=issue_number)
    issue_title = issue.title or ""
    issue_body = issue.body or ""

    branch = f"issue-{issue_number}"

    # 1) Create branch
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

    # 3) Context
    context = build_context(issue_title, issue_body)

    # 4) Ask model for diff (BYOK + fallback)
    last_err: Optional[Exception] = None
    last_raw: Optional[str] = None
    diff_text: Optional[str] = None
    used_model: Optional[str] = None

    for model in iter_models():
        try:
            print(f"[LLM] Trying model: {model}")
            raw = call_model_for_diff(model, issue_title, issue_body, context)
            last_raw = raw
            diff_text = extract_diff(raw)
            diff_text = diff_text.replace("\r\n", "\n").strip() + "\n"
            used_model = model
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"[LLM] Model failed: {model} | error: {msg}")

            # keep trying next models for common provider errors
            if looks_retryable_error(msg) or looks_retryable_error(last_raw or ""):
                continue
            continue

    if not diff_text:
        # Helpful feedback in the Issue (BYOK friendliness)
        comment_issue(
            repo,
            issue_number,
            "❌ **Code Agent failed to generate a patch.**\n\n"
            f"Last error: `{last_err}`\n\n"
            "BYOK tip: configure `OPENROUTER_API_KEY` (Secrets) and optionally `MODEL` (Variables), "
            "then reopen/create a new Issue to retry.",
        )
        raise RuntimeError(f"All models failed. Last error: {last_err}\nLast output: {last_raw}")

    print(f"=== DIFF (from model: {used_model}) ===")
    print(diff_text)

    # 5) Apply patch
    patch_file = Path("agent_patch.diff")
    patch_file.write_text(diff_text, encoding="utf-8")

    ok, _ = try_sh("git apply --whitespace=fix agent_patch.diff")
    if not ok:
        ok2, _ = try_sh("git apply --3way --whitespace=fix agent_patch.diff")
        if not ok2:
            comment_issue(
                repo,
                issue_number,
                "❌ **Generated patch could not be applied.**\n\n"
                f"Model: `{used_model}`\n\n"
                "Tip: make the Issue more specific (file names, exact text changes) and retry.",
            )
            raise RuntimeError("Failed to apply patch. Aborting.")

    # Remove patch file to avoid committing it
    try:
        patch_file.unlink()
    except Exception:
        pass

    def find_node_project_dir() -> str | None:
        """
        Finds the first directory containing package.json.
        Skips node_modules and .git.
        """
        for p in Path(".").rglob("package.json"):
            if "node_modules" in p.parts or ".git" in p.parts:
                continue
            return str(p.parent)
        return None
    # 6) Best-effort checks
    node_dir = find_node_project_dir()

    if node_dir:
        print(f"Found Node project in: {node_dir}")
        try_sh(f"cd {node_dir} && npm install")
        try_sh(f"cd {node_dir} && npm run lint --if-present")
        try_sh(f"cd {node_dir} && npm test --if-present")
        try_sh(f"cd {node_dir} && npm run build --if-present")
    else:
        print("No package.json found. Skipping npm checks.")

    # Clean caches again
    sh("find . -type d -name __pycache__ -prune -exec rm -rf {} + || true")
    sh("find . -type f -name '*.pyc' -delete || true")

    # 7) Commit if changed
    status = subprocess.check_output("git status --porcelain", shell=True, text=True).strip()
    print("$ git status --porcelain")
    print(status)

    if not status:
        comment_issue(
            repo,
            issue_number,
            "ℹ️ **No changes detected after applying the patch.**\n\n"
            f"Model tried: `{used_model}`\n\n"
            "Tip: rephrase the Issue with more specific requirements.",
        )
        print("No changes detected. Not creating PR.")
        return

    sh("git add -A")
    sh(f"git commit -m 'Auto-fix issue #{issue_number}'")
    sh(f"git push -u origin {branch}")

    # 8) Create PR
    pr_title = f"Auto-fix for issue #{issue_number}: {issue_title}".strip()
    pr_body = (
        "This PR was automatically generated by Code Agent (BYOK, no-tools fallback).\n\n"
        f"Model used: `{used_model}`\n\n"
        "Patch was generated as unified diff and applied via `git apply`.\n"
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
