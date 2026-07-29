import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "app" / "templates" / "index.html"


class UiTemplateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_original_gui_layout_is_preserved(self):
        self.assertIn("<title>LLM-Enhanced WBLCA</title>", self.html)
        self.assertIn('<p class="eyebrow">openLCA + LLM workflow</p>', self.html)
        self.assertIn("<h1>LLM-Enhanced WBLCA</h1>", self.html)
        self.assertIn('class="tool-grid"', self.html)
        self.assertIn("<h2>BOM Excel</h2>", self.html)
        self.assertIn("<h2>Material Query</h2>", self.html)
        self.assertIn("API ready", self.html)
        self.assertIn('id="materialResult">Result will appear here.</div>', self.html)

    def test_no_preflight_llama_or_capability_ui_was_added(self):
        self.assertNotIn("System capabilities", self.html)
        self.assertNotIn('fetch("/health"', self.html)
        self.assertNotIn("llamaStatusCard", self.html)
        self.assertNotIn("Llama loads on demand", self.html)

    def test_original_route_contracts_are_preserved(self):
        self.assertIn('fetch("/calculate"', self.html)
        self.assertIn(
            "`/calculate_bom_excel?filename=${encodeURIComponent(file.name)}`",
            self.html,
        )
        self.assertRegex(
            self.html,
            re.compile(
                r'fetch\("/calculate".*?body:\s*JSON\.stringify\(\{ input: material \}\)',
                re.DOTALL,
            ),
        )

    def test_llama_message_appears_only_in_post_run_result(self):
        self.assertIn("if (result.message)", self.html)
        self.assertIn('resultLines.push("", result.message);', self.html)
        self.assertIn(
            'response.headers.get("X-Status-Message") || ""',
            self.html,
        )
        self.assertGreater(
            self.html.index("if (result.message)"),
            self.html.index('fetch("/calculate"'),
        )

    def test_template_remains_self_contained(self):
        self.assertIn("<!doctype html>", self.html.lower())
        self.assertIn('name="viewport"', self.html)
        self.assertNotRegex(self.html, r'<script[^>]+src=')
        self.assertNotRegex(
            self.html,
            r'<link[^>]+rel=["\']stylesheet["\']',
        )


if __name__ == "__main__":
    unittest.main()
