"""
Automated submission script for the TUM.ai x Unite Hackathon evaluator.

Usage:
    uv run python submit.py --challenge 2 --file submission.csv
    uv run python submit.py --challenge 1 --file submission.parquet
    uv run python submit.py --challenge 2 --file submission.csv --level 1

Credentials are read from .env:
    TEAM=<your team name>
    PASSWORD=<your password>
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

PORTAL_URL = "https://unite-evaluator.vercel.app"
LOGIN_URL = f"{PORTAL_URL}/login"


def login(page, team: str, password: str) -> None:
    """Authenticate and establish a browser session."""
    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    page.locator("#teamName").fill(team)
    page.locator("#password").fill(password)
    page.locator("button[type=submit]").click()

    # Auth token is set via Supabase; wait for redirect or session cookie.
    try:
        page.wait_for_url(lambda url: "/login" not in url, timeout=10_000)
    except PlaywrightTimeout:
        pass
    page.goto(PORTAL_URL, wait_until="domcontentloaded")

    if "/login" in page.url:
        raise RuntimeError("Login failed — check TEAM / PASSWORD in your .env file")

    print(f"✓ Logged in as {team}")


def submit(
    page,
    challenge_id: int,
    file_path: Path,
    level: int = 2,
) -> None:
    """Navigate to a challenge page, set granularity, upload file, and submit."""
    challenge_url = f"{PORTAL_URL}/challenges/{challenge_id}"
    page.goto(challenge_url, wait_until="domcontentloaded")

    print(f"✓ Opened challenge {challenge_id}")

    # Select matching granularity (only relevant for challenge 2 which has Level buttons).
    if challenge_id == 2:
        level_labels = {1: "Level 1", 2: "Level 2", 3: "Level 3"}
        target_label = level_labels.get(level, "Level 2")
        matched = False
        for btn in page.locator("button[type=button]").all():
            if target_label in btn.inner_text():
                if btn.is_disabled():
                    raise RuntimeError(
                        f"Level {level} is not yet available on the portal (button is disabled)."
                    )
                btn.click()
                print(f"✓ Selected {target_label}")
                matched = True
                break
        if not matched and level in level_labels:
            raise RuntimeError(
                f"Level {level} button not found on this challenge page — "
                "it may not be available yet."
            )

    # Upload file.
    file_input = page.locator("input[type=file]")
    file_input.set_input_files(str(file_path.resolve()))
    print(f"✓ Attached file: {file_path.name}")

    submission_result: dict = {}

    # Submit and wait for the API response using expect_response.
    print("⏳ Submitting…")
    with page.expect_response(
        lambda r: r.url.endswith("/api/submit") and r.request.method == "POST",
        timeout=30_000,
    ) as response_info:
        page.locator("button[type=submit]").click()

    response = response_info.value
    try:
        submission_result = response.json()
    except (ValueError, TypeError, AttributeError) as exc:
        print("✗ Failed to parse submission API response as JSON:", exc)
        try:
            status = getattr(response, "status", None)
            if status is not None:
                print("  HTTP status:", status)
            body_preview = response.text()
            if len(body_preview) > 300:
                body_preview = body_preview[:300] + "…"
            print("  Response text (truncated):", body_preview)
        except Exception as inner_exc:
            print("  Additionally failed to read raw response text:", inner_exc)
        return

    if not submission_result.get("success"):
        print("✗ Submission API did not confirm success:", submission_result)
        return

    submission = submission_result.get("submission")
    submission_id = submission.get("id") if isinstance(submission, dict) else None
    if submission_id is not None:
        print(f"✓ Submission accepted (id: {submission_id})")
    else:
        print(
            "⚠ Submission API reported success but did not include a submission id:",
            submission_result,
        )
    print("⏳ Waiting for scoring results…")

    # Wait for "Your Submissions" section to appear with a score (not just "evaluating").
    try:
        page.wait_for_function(
            """() => {
                const text = document.body.innerText;
                return text.includes('Your Submissions') && text.includes('Total Score:');
            }""",
            timeout=60_000,
        )
    except PlaywrightTimeout:
        print("⚠  Scoring timed out — results may appear on the portal later.")

    page.wait_for_timeout(1000)

    # Extract the submissions section from the page text.
    result_text = page.locator("body").inner_text()
    print("\n─── Submission Result ───")
    in_submissions = False
    for line in result_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "Your Submissions" in line:
            in_submissions = True
        if in_submissions:
            print(line)
            # Stop after the first submission block.
            if line.startswith("Spend Captured:") or line.startswith("0."):
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a solution to the Unite Hackathon portal")
    parser.add_argument("--challenge", type=int, required=True, choices=[1, 2], help="Challenge number")
    parser.add_argument("--file", type=Path, required=True, help="Path to the submission file")
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Matching granularity level for challenge 2 (default: 2; level 3 may not be available yet)",
    )
    parser.add_argument("--headed", action="store_true", help="Run browser in headed (visible) mode")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    team = os.getenv("TEAM")
    password = os.getenv("PASSWORD")
    if not team or not password:
        print("Error: TEAM and PASSWORD must be set in .env", file=sys.stderr)
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context()
        page = context.new_page()
        try:
            login(page, team, password)
            submit(page, args.challenge, args.file, level=args.level)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
