import unittest
from autograder import runGrader


class TestQ1(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q1'])
        assert result['q1'] == 4


class TestQ2(unittest.TestCase):

    def test_q(self):
        result = runGrader(['-q', 'q2'])
        assert result['q2'] == 5


class TestQ3(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q3'])
        assert result['q3'] == 5


class TestQ4(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q4'])
        assert result['q4'] == 5


class TestQ5(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q5'])
        assert result['q5'] == 6
