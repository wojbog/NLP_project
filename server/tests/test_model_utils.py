import sys
import os
import numbers
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import get_model  # noqa: E402


class TestModelUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = get_model()

    def test_prediction_output(self):
        result = self.model.predict("Some example text.")
        self.assertIn("predicted_class", result)
        self.assertIsInstance(result["predicted_class"], str)

    def test_distribution_sum(self):
        probs = self.model.predict_distribution("Some example text.")
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_importance_alignment(self):
        result = self.model.compute_importance("Some example text.")
        self.assertIsInstance(result, list)
        tokens, importances = zip(*result)
        self.assertEqual(len(tokens), len(importances))
        self.assertTrue(all(isinstance(tok, str) for tok in tokens))
        self.assertTrue(all(isinstance(val, numbers.Number)
                            for val in importances))


if __name__ == '__main__':
    unittest.main()
