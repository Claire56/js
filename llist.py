
class Node:
    def __init__(self, value):
        self.value = value
        self.nextnode = None


class nthNode:
    def __init__(self, value):
        self.value = value
        self.nextnode = None
    
    @staticmethod
    def get_nthnode(node, n):
        current = node
        count = 1
        while current is not None:
            if count == n:
                return current.value
            count += 1
            current = current.nextnode
        return None
        
    @staticmethod
    def get_nth_last_node(head, n):
        # One-pass solution using two pointers
        if head is None or n <= 0:
            return None
        
        # Create two pointers
        fast = head
        slow = head
        
        # Move fast pointer n steps ahead
        for _ in range(n):
            if fast is None:
                return None  # n is larger than list length
            fast = fast.nextnode
        
        # Move both pointers until fast reaches the end
        while fast is not None:
            fast = fast.nextnode
            slow = slow.nextnode
        
        # slow now points to the nth node from the end
        return slow.value