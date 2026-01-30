import os
from typing import List, Tuple, Optional

from github import Auth, Github


def get_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise ValueError(f"Missing env var: {name}")
    return v


def summarize_files(pr) -> Tuple[List[str], int, int]:
    files = pr.get_files()
    names: List[str] = []
    additions = 0
    deletions = 0
    for f in files:
        names.append(f.filename)
        additions += getattr(f, "additions", 0) or 0
        deletions += getattr(f, "deletions", 0) or 0
    return names, additions, deletions


def get_checks_summary(repo, pr) -> Tuple[str, List[str]]:
    """
    Returns (overall_status, details)
    overall_status: "success" | "failure" | "pending" | "unknown"
    """
    try:
        # Use combined status (works for many CI setups)
        combined = repo.get_commit(pr.head.sha).get_combined_status()
        states = [s.state for s in combined.statuses] 
        details = [f"{s.context}: {s.state}" for s in combined.statuses]
        if any(s in ("failure", "error") for s in states):
            return "failure", details
        if any(s == "pending" for s in states):
            return "pending", details
        if states and all(s == "success" for s in states):
            return "success", details
        return "unknown", details
    except Exception as e:
        return "unknown", [f"Could not fetch checks: {e}"]


def detect_issue_number_from_branch(branch_name: str) -> Optional[int]:
    if branch_name.startswith("issue-"):
        tail = branch_name.split("issue-", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def write_step_summary(markdown_text: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(markdown_text + "\n")


def main() -> None:
    token = get_env("GITHUB_TOKEN")
    repo_name = get_env("GITHUB_REPOSITORY")
    pr_number = int(get_env("PR_NUMBER"))

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    file_names, additions, deletions = summarize_files(pr)
    checks_overall, checks_details = get_checks_summary(repo, pr)

    issue_number = detect_issue_number_from_branch(pr.head.ref)
    issue_title = None
    issue_body = None
    if issue_number:
        try:
            issue = repo.get_issue(number=issue_number)
            issue_title = issue.title or ""
            issue_body = issue.body or ""
        except Exception:
            pass

    # Basic “review policy” MVP:
    # - If checks failed -> request changes
    # - If pending -> comment only
    # - If success -> approve
    suspicious = any(name.endswith(("package-lock.json", "pnpm-lock.yaml", "yarn.lock")) for name in file_names)

    if checks_overall == "failure":
        verdict = "request_changes"
        verdict_text = "❌ CI/checks failed — требуется исправление."
    elif checks_overall == "pending":
        verdict = "comment"
        verdict_text = "⏳ CI/checks ещё выполняются — финальный вердикт позже."
    elif checks_overall == "success":
        if suspicious:
            verdict = "comment"
            verdict_text = "✅ CI/checks успешны, но есть изменения lock-файлов — проверь, что это осознанно."
        else:
            verdict = "approve"
            verdict_text = "✅ CI/checks успешны — выглядит готовым."
    else:
        verdict = "comment"
        verdict_text = "ℹ️ Не удалось однозначно определить статус CI/checks."


    issue_block = ""
    if issue_number and issue_title is not None:
        issue_block = f"\n**Linked Issue:** #{issue_number} — {issue_title}\n"

    files_block = "\n".join([f"- {n}" for n in file_names[:30]])
    if len(file_names) > 30:
        files_block += f"\n- ... (+{len(file_names)-30} more)"

    checks_block = "\n".join([f"- {d}" for d in checks_details[:30]])
    if len(checks_details) > 30:
        checks_block += f"\n- ... (+{len(checks_details)-30} more)"

    comment_body = f"""## 🤖 AI Reviewer Report

{verdict_text}
{issue_block}
**PR:** #{pr_number}  
**Branch:** `{pr.head.ref}` → `{pr.base.ref}`  
**Changes:** +{additions} / -{deletions}  
**Checks status:** `{checks_overall}`

### Changed files
{files_block if files_block else "- (no files?)"}

### Checks details
{checks_block if checks_block else "- (no checks found)"}

### Notes
- This is an automated review based on GitHub PR metadata + checks.
- If CI is red, please fix and push updates — review will rerun automatically.
"""

    pr.create_issue_comment(comment_body)

    write_step_summary(comment_body)

    if verdict == "approve":
        pr.create_review(event="APPROVE", body="✅ CI green. Auto-approval by Reviewer Agent.")
    elif verdict == "request_changes":
        pr.create_review(event="REQUEST_CHANGES", body="❌ CI failed. Please fix and push updates.")
    else:
        pr.create_review(event="COMMENT", body="⏳ Neutral automated review. Waiting for CI or more signal.")

    print("Reviewer Agent: review posted.")


if __name__ == "__main__":
    main()
