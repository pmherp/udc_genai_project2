import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(dotenv_path=REPO_ROOT / ".env")

openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if openai_key:
    os.environ.setdefault("OPENAI_API_KEY", openai_key)
    os.environ.setdefault("OPENAI_KEY", openai_key)
    os.environ.setdefault("CHROMA_OPENAI_API_KEY", openai_key)

import llm_client  # noqa: E402
import rag_client  # noqa: E402
import ragas_evaluator  # noqa: E402


class TestComponentSmoke(unittest.TestCase):
    def test_llm_client_generate_response(self):
        response = llm_client.generate_response(
            openai_key="test-key",
            user_message="What was Apollo 11?",
            context="",
            conversation_history=[],
        )
        self.assertIsInstance(response, str)
        self.assertIn(
            "The available documents do not contain sufficient information",
            response,
        )

    def test_rag_client_discover_chroma_backends(self):
        backends = rag_client.discover_chroma_backends()
        self.assertIsInstance(backends, dict)

    def test_embedding_pipeline_stats_only_cli(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "embedding_pipeline.py"

        with tempfile.TemporaryDirectory() as temp_chroma_dir:
            cmd = [
                sys.executable,
                str(script_path),
                "--openai-key",
                "test-key",
                "--chroma-dir",
                temp_chroma_dir,
                "--collection-name",
                "test_collection",
                "--stats-only",
            ]
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            self.fail(
                "embedding_pipeline --stats-only failed with code "
                f"{result.returncode}. stderr: {result.stderr}"
            )

    def test_ragas_evaluator_evaluate_response_quality(self):
        scores = ragas_evaluator.evaluate_response_quality(
            "question",
            "answer",
            ["context"],
        )
        self.assertIsInstance(scores, dict)
        self.assertTrue(scores)


def _print_summary(result: unittest.result.TestResult) -> None:
    success = result.wasSuccessful()
    status_icon = "✅" if success else "❌"
    status_text = "PASS" if success else "FAIL"

    summary_rows = [
        ("Status", f"{status_icon} {status_text}"),
        ("Tests", str(result.testsRun)),
        ("Errors", str(len(result.errors))),
        ("Failures", str(len(result.failures))),
        ("Skipped", str(len(result.skipped))),
    ]

    print("\n" + "=" * 52)
    print("🧪 Component Smoke Test Summary")
    print("-" * 52)
    for label, value in summary_rows:
        print("{:<10}: {}".format(label, value))
    print("=" * 52)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestComponentSmoke
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    _print_summary(result)

    sys.exit(0 if result.wasSuccessful() else 1)
