class Node:
    def __init__(self,key):
        self.left=None
        self.right=None
        self.data=key

def l_depth(root):
    height=1
    while(root is not None):
        root=root.left
        height=height+1
        return height
def r_depth(root):
    height=1
    while(root is not None):
        root=root.right
        height=height+1
        return height

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)