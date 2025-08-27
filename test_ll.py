import pytest
from llist import(
    nthNode, Node
)

class TestnthNode:
    def test_2last_node(self):
        a = nthNode(1)
        b = nthNode(2)
        c = nthNode(3)
        d = nthNode(4)
        e = nthNode(5)

        a.nextnode = b
        b.nextnode = c
        c.nextnode = d
        d.nextnode = e

        assert nthNode.get_nthnode(a, 2) == 2
    def test_third_node(self):
        a = nthNode(1)
        b = nthNode(2)
        c = nthNode(3)
        d = nthNode(4)
        e = nthNode(5)

        a.nextnode = b
        b.nextnode = c
        c.nextnode = d
        d.nextnode = e

        assert nthNode.get_nthnode(a, 3) == 3

    def test_2ndlast_node(self):
        a = nthNode(1)
        b = nthNode(2)
        c = nthNode(3)
        d = nthNode(4)
        e = nthNode(5)

        a.nextnode = b
        b.nextnode = c
        c.nextnode = d
        d.nextnode = e

        assert nthNode.get_nth_last_node(a, 2) == 4


class Test_Node:
    def test_node(self):
        a = Node(2)
        b = Node(4)
        c = Node(6)

        a.nextnode = b
        b.nextnode = c