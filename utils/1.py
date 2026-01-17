import subprocess
import sys


def run(cmd: str) -> None:
    # Run git commands and stop on first failure.
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


repo_url = "https://github.com/drggk-mlan/my_code_snippets.git"

run("git init")
run("git add .")
run('git commit -m "Initial commit"')
run("git branch -M main")
run(f"git remote add origin {repo_url}")
run("git push -u origin main")