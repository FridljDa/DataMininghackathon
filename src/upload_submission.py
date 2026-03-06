"""Upload a challenge submission CSV through the evaluator web UI.

This script automates:
1) optional cookie injection / login
2) opening the challenge page
3) attaching the submission CSV
4) triggering upload
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


DEFAULT_BASE_URL = "https://unite-evaluator.vercel.app"
DEFAULT_CHALLENGE_ID = "2"
DEFAULT_TIMEOUT_MS = 20_000


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _parse_cookie_header(cookie_header: str, domain: str) -> list[dict[str, Any]]:
    cookies: list[dict[str, Any]] = []
    parts = [chunk.strip() for chunk in cookie_header.split(";") if chunk.strip()]
    for part in parts:
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookies.append(
            {
                "name": name.strip(),
                "value": value.strip(),
                "domain": domain,
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        )
    return cookies


def _load_cookies_from_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Common formats:
    # 1) Playwright storage state: {"cookies": [...], "origins": [...]}
    # 2) Cookie export: [...]
    if isinstance(raw, dict) and "cookies" in raw and isinstance(raw["cookies"], list):
        return raw["cookies"]
    if isinstance(raw, list):
        return raw
    raise ValueError(
        f"Unsupported cookie JSON format in {path}. "
        "Expected a list of cookies or a Playwright storage_state object."
    )


def _is_login_page(page) -> bool:
    return (
        page.get_by_role("heading", name="Sign In").count() > 0
        or page.get_by_label("Team Name").count() > 0
        or "/login" in page.url
    )


def _extract_login_error(page) -> str | None:
    # Try typical error containers first.
    selectors = (
        "[role='alert']",
        ".text-red-500",
        ".text-destructive",
        ".error",
        "[data-error]",
    )
    for selector in selectors:
        loc = page.locator(selector)
        if loc.count() > 0:
            text = loc.first.inner_text().strip()
            if text:
                return text

    # Fallback: include form text to aid debugging.
    form = page.locator("form")
    if form.count() > 0:
        text = form.first.inner_text().strip()
        if text:
            return text
    return None


def _do_login(page, team_name: str, password: str) -> None:
    team_field = page.get_by_placeholder("Your team name")
    if team_field.count() == 0:
        team_field = page.get_by_label("Team Name")
    password_field = page.get_by_placeholder("Your password")
    if password_field.count() == 0:
        password_field = page.get_by_label("Password")

    team_field.first.fill(team_name)
    password_field.first.fill(password)

    team_value = team_field.first.input_value()
    password_value = password_field.first.input_value()
    if not team_value or not password_value:
        raise RuntimeError("Login fields were not filled correctly before submit.")

    submit_button = page.get_by_role("button", name="Sign In").first
    if not submit_button.is_enabled():
        raise RuntimeError("Sign In button is disabled after filling credentials.")
    submit_button.click()
    try:
        page.wait_for_timeout(1500)
        page.wait_for_url("**/challenges/**", timeout=5000)
    except PlaywrightTimeoutError:
        # It may redirect somewhere else first or remain on login if auth failed.
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        # Some apps keep long-lived requests open; this is fine.
        pass


def _upload_csv(page, csv_path: Path) -> None:
    all_file_inputs = page.locator("input[type='file']")
    input_count = all_file_inputs.count()
    if input_count == 0:
        raise RuntimeError("No file input found on the challenge page.")

    print(f"Found {input_count} file input(s).")
    upload_bound = False
    for idx in range(input_count):
        candidate = all_file_inputs.nth(idx)
        candidate.wait_for(state="attached", timeout=DEFAULT_TIMEOUT_MS)
        try:
            accept_attr = candidate.get_attribute("accept") or ""
            if accept_attr and ".csv" not in accept_attr.lower():
                continue

            candidate.set_input_files(str(csv_path))
            upload_bound = True
            files_len = candidate.evaluate("el => (el.files ? el.files.length : 0)")
            print(f"Input #{idx} files length after set: {files_len}")
            # Some components clear the hidden input after internal processing.
            break
        except PlaywrightError as exc:
            print(f"Input #{idx} set_input_files failed: {exc}")
            continue

    if not upload_bound:
        raise RuntimeError("Failed to bind CSV file to any file input.")

    page.screenshot(path="outputs/upload_after_file_select.png", full_page=True)

    # Some UIs auto-submit on file selection. Try explicit buttons if present.
    button_patterns = ("Submit", "Upload", "Send", "Evaluate")
    clicked_button = False
    for label in button_patterns:
        button = page.get_by_role("button", name=label)
        if button.count() > 0:
            first = button.first
            if first.is_enabled():
                first.click()
                clicked_button = True
                break
            print(f"Found '{label}' button but it is disabled.")

    if not clicked_button:
        print("No enabled submit-like button found. Continuing in case auto-upload is used.")
        # Emit page body text snippet for diagnostics when submit remains disabled.
        body_text = page.locator("body").inner_text()
        compact = " ".join(body_text.split())
        print(f"Page text snippet after upload: {compact[:400]}")

    # Give backend a brief moment to process and render a result.
    page.wait_for_timeout(3000)


def parse_args() -> argparse.Namespace:
    # Load .env values before building defaults for CLI flags.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file for credentials/cookies (default: .env).",
    )
    pre_args, _ = pre_parser.parse_known_args()
    _load_env_file(Path(pre_args.env_file).expanduser())

    parser = argparse.ArgumentParser(
        description="Upload data/10_submission/submission.csv to Unite evaluator."
    )
    parser.add_argument(
        "--env-file",
        default=pre_args.env_file,
        help="Path to .env file for credentials/cookies (default: .env).",
    )
    parser.add_argument(
        "--csv",
        default="data/10_submission/submission.csv",
        help="Path to submission CSV (default: data/10_submission/submission.csv).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Evaluator base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--challenge-id",
        default=DEFAULT_CHALLENGE_ID,
        help=f"Challenge ID (default: {DEFAULT_CHALLENGE_ID}).",
    )
    parser.add_argument(
        "--team-name",
        default=os.getenv("UNITE_TEAM_NAME"),
        help="Team name (or set UNITE_TEAM_NAME).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("UNITE_PASSWORD"),
        help="Password (or set UNITE_PASSWORD).",
    )
    parser.add_argument(
        "--cookie-header",
        default=os.getenv("UNITE_COOKIE_HEADER"),
        help="Cookie header string, e.g. 'a=b; c=d' (or UNITE_COOKIE_HEADER).",
    )
    parser.add_argument(
        "--cookie-json",
        help="Path to JSON file with cookies (list or Playwright storage_state).",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run browser in visible mode (headless is default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")
    Path("outputs").mkdir(parents=True, exist_ok=True)

    challenge_url = f"{args.base_url.rstrip('/')}/challenges/{args.challenge_id}"
    login_url = f"{args.base_url.rstrip('/')}/login"
    domain = args.base_url.replace("https://", "").replace("http://", "").strip("/")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headful)
        context = browser.new_context()
        page = context.new_page()

        # Optional cookie-based auth before first navigation.
        cookies_to_add: list[dict[str, Any]] = []
        if args.cookie_header:
            cookies_to_add.extend(_parse_cookie_header(args.cookie_header, domain=domain))
        if args.cookie_json:
            cookies_to_add.extend(_load_cookies_from_json(Path(args.cookie_json)))
        if cookies_to_add:
            context.add_cookies(cookies_to_add)

        page.goto(challenge_url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)

        # Login only if still unauthenticated.
        if _is_login_page(page):
            if not args.team_name or not args.password:
                raise RuntimeError(
                    "Authentication required. Provide either cookies "
                    "(--cookie-header/--cookie-json) or credentials "
                    "(--team-name/--password or env vars)."
                )
            if "/login" not in page.url:
                page.goto(login_url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
            _do_login(page, team_name=args.team_name, password=args.password)
            page.goto(challenge_url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)

        if _is_login_page(page):
            page.screenshot(path="outputs/login_failed.png", full_page=True)
            login_error = _extract_login_error(page)
            if login_error:
                raise RuntimeError(
                    "Still on login page after authentication attempt. "
                    f"Page feedback: {login_error}"
                )
            raise RuntimeError("Still on login page after authentication attempt.")

        _upload_csv(page, csv_path)

        print(f"Upload attempted for: {csv_path}")
        print(f"Current page: {page.url}")
        page.screenshot(path="outputs/upload_result.png", full_page=True)
        print("Saved screenshot: outputs/upload_result.png")

        browser.close()


if __name__ == "__main__":
    main()
