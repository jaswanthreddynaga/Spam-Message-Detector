"""Run with python -m unittest -v test_app.py."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()

    def check_message(self):
        self.app.button[4].click().run()
        self.assertFalse(self.app.exception)

    def test_empty_and_whitespace(self):
        self.check_message()
        self.assertIn("Please type", self.app.warning[0].value)
        self.app.text_area[0].set_value("   ").run()
        self.check_message()
        self.assertEqual(len(self.app.warning), 1)

    def test_examples_and_edit(self):
        for name, expected_spam in [("Prize offer", True), ("Urgent promotion", True), ("Dinner plans", False), ("Everyday update", False)]:
            next(button for button in self.app.button if button.label == name).click().run()
            self.assertTrue(self.app.text_area[0].value)
            self.check_message()
            if expected_spam:
                self.assertIn("### Spam", self.app.error[0].value)
            else:
                self.assertIn("### Not Spam", self.app.success[0].value)
        self.app.text_area[0].set_value("Thanks, see you tomorrow!").run()
        self.assertEqual(len(self.app.success), 0)
        self.check_message()
        self.assertIn("### Not Spam", self.app.success[0].value)

    def test_unusual_inputs(self):
        for text in ["🎉", "123456", "你好", "x" * 10000]:
            self.app.text_area[0].set_value(text).run()
            self.check_message()
            self.assertEqual(len(self.app.error) + len(self.app.success), 1)


if __name__ == "__main__":
    unittest.main()
