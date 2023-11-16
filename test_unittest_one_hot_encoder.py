import unittest
from one_hot_encoder import fit_transform


class TestOneHotEncoder(unittest.TestCase):

    def test_empty_input(self):
        with self.assertRaises(TypeError):
            fit_transform()

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            fit_transform(0)

    def test_check_the_order(self):
        result = fit_transform('check', 'the', 'order')
        expected = [
            ('check', [0, 0, 1]),
            ('the', [0, 1, 0]),
            ('order', [1, 0, 0])
        ]
        self.assertEqual(result, expected)

    def test_list_check_the_order(self):
        result = fit_transform(['check', 'the', 'order'])
        self.assertIn(('check', [0, 0, 1]), result)

    def test_check_the_check(self):
        result = fit_transform('check', 'the', 'check')
        expected = [
            ('check', [0, 1]),
            ('the', [1, 0]),
            ('check', [0, 1])
        ]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
