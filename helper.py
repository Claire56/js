def unique_characters(sentence):
    if not sentence:
        return True
    
    result = set(sentence)
    return len(result) == len(sentence)

def unique_characters2(sentence):
    if not sentence:
        return True
    char_count= []
    for i in sentence:
        if i in char_count:
            return False
        else : 
            char_count.append(i)
    return True

def balanced_check(sentence):
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    if not sentence:
        return False
    for ch in sentence:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:  # Check if stack is empty before popping
                return False
            pop_stack_item = stack.pop()
            if pairs[pop_stack_item] != ch:
                return False
    return not stack



class queue2stacks:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, item):
        self.stack1.append(item)

    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
    
class Queue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, item):
        self.in_stack.append(item)

    def dequeue(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

class LLcycleNode(object):
    '''takes a first node and returns boolean 
    indicating if the linked list is cycle
    '''
    def __init__(self, value):
        self.value = value
        self.nextnode = None
        self.prevnode = None

    def cycle_check(first_node):
        marker1 = first_node
        marker2 = first_node
        while marker2 != None and marker2.nextnode != None:
            marker1 = marker1.nextnode
            marker2 = marker2.nextnode.nextnode
            if marker2 == marker1:
                return True
        return False