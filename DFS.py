class Node:
    def __init__(self,key):
        self.left=None
        self.right=None
        self.data=key
def PrintInorder(root):
    if root:
        PrintInorder(root.left)
        print(root.data),
        PrintInorder(root.right)
def PrintPreorder(root):
    if root:
        print(root.data),
        PrintPreorder(root.left)
        PrintPreorder(root.right)
def PrintPostorder(root):
    if root:
        PrintPostorder(root.left)
        PrintPostorder(root.right)
        print(root.data),

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

print("Preorder traversal of binary tree is")
PrintPreorder(root)
print("\nInorder traversal of binary tree is")
PrintInorder(root)
print("\nPostorder traversal of binary tree is")
PrintPostorder(root)