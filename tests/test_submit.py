from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from submit import login, submit, main, PORTAL_URL, LOGIN_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(url: str = PORTAL_URL) -> MagicMock:
    """Return a minimal mock of a Playwright page."""
    page = MagicMock()
    type(page).url = property(lambda self: url)
    return page


def _make_button(text: str, disabled: bool = False) -> MagicMock:
    btn = MagicMock()
    btn.inner_text.return_value = text
    btn.is_disabled.return_value = disabled
    return btn


def _make_response_ctx(json_body: dict) -> MagicMock:
    """Return a mock that behaves like page.expect_response() context manager."""
    response = MagicMock()
    response.json.return_value = json_body

    info = MagicMock()
    info.value = response

    @contextmanager
    def _ctx(*_args, **_kwargs):
        yield info

    ctx_mgr = MagicMock(side_effect=_ctx)
    return ctx_mgr


# ---------------------------------------------------------------------------
# login()
# ---------------------------------------------------------------------------

class TestLogin:
    def test_successful_login_fills_form_and_navigates(self) -> None:
        page = _make_page(url=PORTAL_URL)  # not /login → success
        login(page, "myteam", "secret")

        page.goto.assert_any_call(LOGIN_URL)
        # Both fill calls go through page.locator().fill; verify both values
        # were passed (order matches the login implementation).
        fill_calls = page.locator.return_value.fill.call_args_list
        assert call("myteam") in fill_calls
        assert call("secret") in fill_calls
        page.locator("button[type=submit]").click.assert_called_once()
        # must navigate to portal root after Supabase cookie is set
        page.goto.assert_called_with(PORTAL_URL)

    def test_failed_login_raises(self) -> None:
        page = _make_page(url=f"{PORTAL_URL}/login")
        with pytest.raises(RuntimeError, match="Login failed"):
            login(page, "bad", "creds")

    def test_login_prints_team_name(self, capsys) -> None:
        page = _make_page(url=PORTAL_URL)
        login(page, "AAD", "pw")
        assert "AAD" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------

class TestSubmit:
    def _page_with_buttons(
        self,
        buttons: list[MagicMock],
        response_body: dict | None = None,
        body_text: str = "Your Submissions\nTotal Score:\n€0.00\nSpend Captured: 0.00%",
    ) -> MagicMock:
        page = MagicMock()
        page.locator.return_value.all.return_value = buttons
        page.locator.return_value.inner_text.return_value = body_text
        page.expect_response = _make_response_ctx(
            response_body or {"success": True, "submission": {"id": "abc-123"}}
        )
        page.locator("body").inner_text.return_value = body_text
        return page

    # --- level selection ---

    def test_level2_button_selected_and_clicked(self, tmp_path) -> None:
        csv = tmp_path / "sub.csv"
        csv.write_text("legal_entity_id,cluster\n1,30020903|Bissell\n")
        l1 = _make_button("Level 1 — E-Class only")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons([l1, l2])

        submit(page, challenge_id=2, file_path=csv, level=2)

        l2.click.assert_called_once()
        l1.click.assert_not_called()

    def test_level1_button_selected_and_clicked(self, tmp_path) -> None:
        csv = tmp_path / "sub.csv"
        csv.write_text("legal_entity_id,cluster\n1,30020903\n")
        l1 = _make_button("Level 1 — E-Class only")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons([l1, l2])

        submit(page, challenge_id=2, file_path=csv, level=1)

        l1.click.assert_called_once()
        l2.click.assert_not_called()

    def test_level3_button_not_found_raises(self, tmp_path) -> None:
        csv = tmp_path / "sub.csv"
        csv.write_text("legal_entity_id,cluster\n1,30020903\n")
        # Only L1/L2 present — L3 not in DOM yet
        page = self._page_with_buttons(
            [_make_button("Level 1 — E-Class only"),
             _make_button("Level 2 — E-Class + Manufacturer")]
        )

        with pytest.raises(RuntimeError, match="Level 3 button not found"):
            submit(page, challenge_id=2, file_path=csv, level=3)

    def test_level3_button_disabled_raises(self, tmp_path) -> None:
        csv = tmp_path / "sub.csv"
        csv.write_text("legal_entity_id,cluster\n1,30020903\n")
        l3 = _make_button("Level 3 — E-Class + Feature Combination", disabled=True)
        page = self._page_with_buttons(
            [_make_button("Level 1"), _make_button("Level 2"), l3]
        )

        with pytest.raises(RuntimeError, match="not yet available"):
            submit(page, challenge_id=2, file_path=csv, level=3)

    def test_no_level_buttons_on_challenge1_is_fine(self, tmp_path) -> None:
        """Challenge 1 has no level toggle; submit should not raise."""
        parquet2 = tmp_path / "sub2.parquet"
        parquet2.write_bytes(b"PAR1")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page2 = self._page_with_buttons([l2])
        submit(page2, challenge_id=1, file_path=parquet2, level=2)
        l2.click.assert_called_once()

    # --- file upload ---

    def test_file_is_attached(self, tmp_path) -> None:
        csv = tmp_path / "my_submission.csv"
        csv.write_text("legal_entity_id,cluster\n")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons([l2])

        submit(page, challenge_id=2, file_path=csv, level=2)

        page.locator("input[type=file]").set_input_files.assert_called_once_with(
            str(csv.resolve())
        )

    # --- API response handling ---

    def test_successful_submission_prints_id(self, tmp_path, capsys) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text("legal_entity_id,cluster\n")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons(
            [l2],
            response_body={"success": True, "submission": {"id": "dead-beef"}},
        )

        submit(page, challenge_id=2, file_path=csv, level=2)

        assert "dead-beef" in capsys.readouterr().out

    def test_failed_api_response_prints_warning_without_raising(self, tmp_path, capsys) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text("legal_entity_id,cluster\n")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons(
            [l2],
            response_body={"success": False, "error": "quota exceeded"},
        )

        # Should not raise
        submit(page, challenge_id=2, file_path=csv, level=2)

        out = capsys.readouterr().out
        assert "did not confirm success" in out

    def test_scoring_timeout_warns_without_raising(self, tmp_path, capsys) -> None:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        csv = tmp_path / "s.csv"
        csv.write_text("legal_entity_id,cluster\n")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons([l2])
        # Make wait_for_function raise a timeout
        page.wait_for_function.side_effect = PlaywrightTimeout("timeout")

        submit(page, challenge_id=2, file_path=csv, level=2)  # must not raise

        assert "timed out" in capsys.readouterr().out

    def test_navigates_to_correct_challenge_url(self, tmp_path) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text("legal_entity_id,cluster\n")
        l2 = _make_button("Level 2 — E-Class + Manufacturer")
        page = self._page_with_buttons([l2])

        submit(page, challenge_id=2, file_path=csv, level=2)

        page.goto.assert_called_once_with(f"{PORTAL_URL}/challenges/2")


# ---------------------------------------------------------------------------
# main() — CLI argument handling
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_file_exits_with_error(self, tmp_path, capsys) -> None:
        ghost = tmp_path / "nonexistent.csv"
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["submit", "--challenge", "2", "--file", str(ghost)]):
                main()
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_missing_env_vars_exits_with_error(self, tmp_path, capsys) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text("a,b\n")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["submit", "--challenge", "2", "--file", str(csv)]):
                with patch.dict("os.environ", {}, clear=True):
                    with patch("submit.load_dotenv"):
                        main()
        assert exc.value.code == 1
        assert "TEAM" in capsys.readouterr().err

    def test_invalid_level_rejected_by_argparse(self, tmp_path, capsys) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text("a,b\n")
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["submit", "--challenge", "2", "--file", str(csv), "--level", "4"]):
                main()
        assert exc.value.code == 2  # argparse error

    def test_level_3_accepted_by_argparse(self, tmp_path) -> None:
        """--level 3 must parse without argparse error (may fail later in submit())."""
        csv = tmp_path / "s.csv"
        csv.write_text("a,b\n")
        with patch("sys.argv", ["submit", "--challenge", "2", "--file", str(csv), "--level", "3"]):
            with patch("submit.load_dotenv"):
                with patch.dict("os.environ", {"TEAM": "t", "PASSWOR": "p"}):
                    with patch("submit.sync_playwright") as mock_pw:
                        # Make the browser/page chain raise RuntimeError (level 3 not found)
                        mock_page = MagicMock()
                        mock_page.locator.return_value.all.return_value = []
                        mock_pw.return_value.__enter__.return_value.chromium.launch.return_value \
                            .new_context.return_value.new_page.return_value = mock_page
                        # url property after login must not contain /login
                        type(mock_page).url = property(lambda self: PORTAL_URL)

                        with pytest.raises(RuntimeError, match="Level 3 button not found"):
                            main()
