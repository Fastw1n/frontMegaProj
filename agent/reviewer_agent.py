import os
from github import Github, Auth


def main():
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")

    if not token or not repo_name or not pr_number:
        raise ValueError("Missing environment variables")

    pr_number = int(pr_number)

    auth = Auth.Token(token)
    g = Github(auth=auth)
    repo = g.get_repo(repo_name)

    pr = repo.get_pull(pr_number)

    files = pr.get_files()

    changed_files = []
    for file in files:
        changed_files.append(file.filename)

    comment_body = f"""
## 🤖 AI Reviewer Report

Changed files:
{chr(10).join(['- ' + f for f in changed_files])}

Basic review complete.
    """

    pr.create_issue_comment(comment_body)

    print("Review comment posted")


if __name__ == "__main__":
    main()
