import unittest

from rmm.eval.tasks import iter_examples


class TaskNormalizationTest(unittest.TestCase):
    def test_copa(self):
        rows = [
            {
                "premise": "The ground was wet.",
                "question": "cause",
                "choice1": "It rained.",
                "choice2": "The sun shone.",
                "label": 0,
            }
        ]
        prompt, choices, answer = next(iter_examples("copa", rows))
        self.assertIn("CAUSE", prompt)
        self.assertEqual(choices, [" It rained.", " The sun shone."])
        self.assertEqual(answer, 0)

    def test_arc_letter_label(self):
        rows = [
            {
                "question": "Pick one.",
                "choices": {"label": ["A", "B"], "text": ["x", "y"]},
                "answerKey": "B",
            }
        ]
        _, choices, answer = next(iter_examples("arc_easy", rows))
        self.assertEqual(choices, [" x", " y"])
        self.assertEqual(answer, 1)


if __name__ == "__main__":
    unittest.main()

