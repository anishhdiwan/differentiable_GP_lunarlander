from genepro.node import *
from genepro import node_impl
from genepro.multitree import Multitree
from node_vectors import node_vectors

class ExpressionMultiTree:
	'''
	The ExpressionMultiTree class is used to instantiate symbolic equation objects. These objects act as states in the first MDP. 


    - Considering an complete binary tree (one where all non-terminal nodes have exactly 2 children) of empty nodes as the initial state. 
    This avoids restricting the scope of the expressions that can be generated. 

    - The initial state is set to have a 'plus' root node and 'zero' left node to ensure consistency across episodes. These operators
    of course have no effect on the final expression generated by the policy, but act as a standardised starting state for all episodes

    - One of the agent's possible actions is to also add a terminal zero_node. The policy can hence also be directed towards 
    generating any binary tree (one where nodes can have any of {0,1,2} children)
    
    - Note: pre-order traversals are generally not unique. However, they can be if the arity of a node is known (which is the case here)
    
    - New nodes are added in the empty spots as per the pre-order traversal order
    
    - Subtracting nodes is not an option since, in the context of sequential decision making, it's existence has no effect on learning. The
    agent will simply learn to add a different operator (or add nothing) at a given spot if the current addition produces poor rewards
        '''

	def __init__(self, tree_depth: int, n_trees: int):

        self.tree_depth = tree_depth # depth of the tree including the root level

        self.multitree = init_multitree(n_trees)
        self.tree_full = [False, False, False, False]

        self.multitree_preorder_travs = get_multitree_preorder_travs(self.multitree)
		# self.preorder_trav = [None for _ in range(2**tree_depth - 1)] 


	def update(self, actions):
		# Update multitree and the pre-order traversal with the performed actions (addition of an operator to each individual tree)
        assert len(actions) == len(self.multitree.n_trees), "The number of actions must be the same as the number of trees in the multitree"

        for i in range(len(actions)):
            if not self.tree_full[i]:
                action = actions[i]
                child_added = update_tree(self.multitree.children[i], action)
                if not child_added:
                    self.tree_full[i] = True

        
        self.multitree_preorder_travs = get_multitree_preorder_travs(self.multitree)

        return self.tree_full

	def evaluate(self, main_env_state):
		# Evaluate the current multitree expression with the main_env's state to get action values for the main_env
        output = self.multitree.get_output_pt(main_env_state)
        return output

    def vectorise_preorder_trav(self):
        # Turn the preorder traversal of the tree (list of nodes that are operator tokens) into a vector representation
        vectorised_multitree_preorder_trav = []
        for trav in self.multitree_preorder_travs:
            vectorised_trav = np.zeros((2**self.tree_depth - 1, 2))
            for i in range(len(trav)):
                operator = trav[i]
                if operator.replace(".", "").isnumeric():
                    vectorised_trav[i] = np.array(node_vectors['const'])
                elif operator[:2] == "x_":
                    vectorised_trav[i] = np.array(node_vectors['x_'])
                else:
                    vectorised_trav[i] = np.array(node_vectors[operator])

            vectorised_multitree_preorder_trav.append(torch.tensor(vectorised_trav, dtype=torch.float32, requires_grad=True))

        return vectorised_multitree_preorder_trav


    def init_multitree(self, n_trees : int):
        '''
        init_multitree generates a multitree object with the plus node as a root node and a zero constant node as the first child of the root node
        The multitree object stores a list of children that in turn recursively store a list of their children.
        '''
        multitree = Multitree(n_trees)

        # Set all the children of the multitree to be empty single trees 
        for _ in range(n_trees):
            multitree.children.append(self.init_tree())
        
        return multitree

    def init_tree(self):
        '''
        init_tree initialises a tree as a cascading chain of node objects that each contain children. These trees form the expression "+ 0" and
        serve as the starting point for DQN
        '''
        root_node = node_impl.Plus() # plus node
        zero_node = node_impl.Constant()
        zero_node.set_value(0.0)

        # insert the zero node as a child for the plus node at index 0. 
        # By convention, index 0 is the left branch when generating a pre-order traversal
        root_node.inset_child(zero_node, 0) 

        return root_node

    def get_multitree_preorder_travs(self, multitree):
        '''
        Generates a list of n_trees pre-order traversals. Each pre-order traversal is a list of node names
        '''
        multitree_preorder_travs = []
        for i in range(multitree.n_trees):
            preorder_trav = []
            self.get_tree_preorder_travs(multitree.children[i], preorder_trav)
            multitree_preorder_travs.insert(preorder_trav, i)

        return multitree_preorder_travs

    def get_tree_preorder_travs(self, tree_root_node, preorder_trav=[]):
        '''
        Generates a tree's preorder traversal starting from its root node. By convention, the child at index 0 is considered to be the left child
        '''
        preorder_trav.append(tree_root_node.symb)
        for child in tree_root_node.children:
            if child.arity > 0:
                preorder_trav.append(self.get_tree_preorder_travs(child, preorder_trav))
            else:
                preorder_trav.append(child.symb)


    def update_tree(self, tree_root_node, action):
        '''
        update_tree adds a new node (action) to a currently existing tree as per the pre-order traversal order
        '''
        # a boolean var to check if a child was added. If after running this function, no child was added then the tree is saturated and the episode needs to end
        child_added = False 

        if (tree_root_node.arity > 0) and (len(tree_root_node.children) == 0):
            # if the current node has an arity > 0 and has no children then insert a child node at index 0 (left side)
            tree_root_node.inset_child(action, 0)
            child_added = True

        elif tree_root_node.arity - len(tree_root_node.children) == 1: 
            # if the current node can accomodate one more child 
            # this can happen for arity 1 nodes with no children or for arity 2 nodes with one child

            if len(tree_root_node.children) == 0:
                # if the node has no children then add a child node at index 0 (left side)
                tree_root_node.inset_child(action, 0)
                child_added = True
            else:
                # if the node has a child (which can only be a left child) then repeat for that child. 
                # if something was still not added (which can happen if the whole left branch is full) then add a right child
                child_added = self.update_tree(tree_root_node.children[0], action)
                if not child_added:
                    tree_root_node.inset_child(action, 1)

        elif tree_root_node.arity - len(tree_root_node.children) == 0:
            # if the current node already has its max possible children then repeat for both children
            for child in tree_root_node.children:
                child_added = self.update_tree(child, action)

        return child_added