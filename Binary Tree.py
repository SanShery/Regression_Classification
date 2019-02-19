class Node:
    def __init__(self,key):
        self.left=0
        self.right=0
        self.data=key
def BFSFn(root):
    if root is None:
        return
    queue=[] #empty list
    queue.append(root)
    while(len(queue)>0):
        print(queue[0].data)
        node=queue.pop(0)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
root=Node(1)
root.left=Node(2)
root.right=Node(3)
root.left.right=Node(4)
BFSFn(root)