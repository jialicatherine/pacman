import unittest
from autograder import runGrader

class TestQ1(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q1'])
        assert result['q1'] == 3

class TestQ2(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q2'])
        assert result['q2'] == 3

class TestQ3(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q3'])
        assert result['q3'] == 3

class TestQ4(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q4'])
        assert result['q4'] == 3

class TestQ5(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q5'])
        assert result['q5'] == 3

class TestQ6(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q6'])
        assert result['q6'] == 3

class TestQ7(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q7'])
        assert result['q7'] >= 4

class TestQ8(unittest.TestCase):
    
    def test_q(self):
        result = runGrader(['-q', 'q8'])
        assert result['q8'] == 3

