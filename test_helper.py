import pytest
from helper import (unique_characters, 
unique_characters2, balanced_check, Queue,
queue2stacks, LLcycleNode)

class TestLLCycke:
    def test_head(self):
        a = LLcycleNode(1)
        b = LLcycleNode(2)
        c = LLcycleNode (3)

        a.nextnode = b
        b.prevnode = a

        b.nextnode = c
        c.prevnode = b

        a.prevnode = c
        c.nextnode = a

        assert LLcycleNode.cycle_check(a) == True

class TestUniqueCharacters:
    """Test cases for the unique_characters function."""
    
    def test_empty_string(self):
        """Test that empty string returns True."""
        assert unique_characters("") == True
    
    def test_single_character(self):
        """Test that single character returns True."""
        assert unique_characters("a") == True
        assert unique_characters("1") == True
    
    def test_unique_characters(self):
        """Test that strings with unique characters return True."""
        assert unique_characters("abc") == True
        assert unique_characters("123") == True
        assert unique_characters("hello") == False
        assert unique_characters("world") == True
    
    def test_duplicate_characters(self):
        """Test that strings with duplicate characters return False."""
        assert unique_characters("hello") == False
        assert unique_characters("aa") == False
        assert unique_characters("abcabc") == False
    
    def test_case_sensitive(self):
        """Test that function is case sensitive."""
        assert unique_characters("aA") == True
        assert unique_characters("Aa") == True
        assert unique_characters("aa") == False
    
    def test_special_characters(self):
        """Test with special characters and spaces."""
        assert unique_characters("!@#") == True
        assert unique_characters("a b") == True
        assert unique_characters("  ") == False  # Two spaces


class TestUnique_characters2:
    """Test cases for the all_values_equal_to_one function."""
    
    def test_empty_sentense(self):
        """Test that empty string returns True."""
        assert unique_characters2('') == True
    
    def test_dup_characters(self):
        """Test that duplicate returns False."""
        assert unique_characters2('abab') == False
        assert unique_characters2('cc') == False
        assert unique_characters2('abcc') == False
    
    def test_mixed_values(self):
        """Test that unique returns True."""
        assert unique_characters2('claire') == True
        assert unique_characters2('nior') == True
        assert unique_characters2('abc') == True
    
    def test_single_values_one(self):
        """Test single returns False."""
        assert unique_characters2('a') == True
        assert unique_characters2('b') == True


class TestBalancedCheck:
    """Test cases for the balanced_check function."""
    
    def test_empty_string(self):
        """Test that empty string returns False."""
        assert balanced_check('') == False

    def test_balanced_check(self):
        """Test that balanced check returns True."""
        assert balanced_check('()()') == True
        assert balanced_check('(())') == True
        assert balanced_check('([])') == True
        assert balanced_check('([{}])') == True

    def test_unbalanced_check(self):
        """Test that unbalanced check returns False."""
        assert balanced_check('(()') == False
        assert balanced_check('())') == False
        assert balanced_check('([)]') == False
        assert balanced_check('([{]})') == False

        
class TestQueue2Stacks:
    """Test cases for the queue2stacks function."""
    
    def test_enqueue_dequeue(self):
        """Test that enqueue and dequeue work correctly."""
        q = queue2stacks()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        assert q.dequeue() == 1

class TestQueue:
    def test_denque(sef):
        q = Queue()
        q.enqueue("mine")
        q.enqueue("me")
        assert q.dequeue() == "mine"