#!/usr/bin/python

"""
Run all unittests tests of the project.
"""

import unittest

if __name__ == "__main__":
    test_cases = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(test_cases)
