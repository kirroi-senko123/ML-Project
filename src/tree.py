import cv2
import numpy as np

class Node:
	def __init__(self, label, parent = None, sibling = None, child = None, top = None, index = None):
		self.top = top
		self.parent = parent
		self.sibling = sibling
		self.child = child
		self.label = label
		self.index = index

class Tree:
	def __init__(self, coordinates):
		self.coordinates = coordinates
		self.root = None

	def locate_and_label(self, prev_node, curr_node):
		coordinates = self.coordinates
		# print("current node is \t" + curr_node.label + " prev node is \t" + prev_node.label)
		if(prev_node.parent):
			print(prev_node.parent.label + "\tis the parent of \t" + prev_node.label)


		l_mid = coordinates[prev_node.index][1]+coordinates[prev_node.index][3]/2
		l_bot = coordinates[prev_node.index][1]+coordinates[prev_node.index][3]
		l_top = coordinates[prev_node.index][1]
		r_bot = coordinates[curr_node.index][1]+coordinates[curr_node.index][3]
		r_top = coordinates[curr_node.index][1]
		prev_parent = prev_node.parent

		if(prev_node.label == '-'):
			l_bot = l_bot + coordinates[prev_node.index][2]/2
			l_top = l_top - coordinates[prev_node.index][2]/2


		if(curr_node.label == '-'):
			r_bot = r_bot + coordinates[curr_node.index][2]/2
			r_top = r_top + coordinates[curr_node.index][2]/2


		if prev_parent is None:
			if (l_mid > r_bot): 
				prev_node.top = curr_node
				curr_node.parent = prev_node
				prev_node = curr_node
			elif (l_mid < r_top):
				prev_node.child = curr_node
				curr_node.parent = prev_node
				prev_node = curr_node
			else:
				prev_node.sibling = curr_node
				curr_node.parent = prev_node.parent
				prev_node = curr_node
			return prev_node
		else:
			parent_mid = coordinates[prev_node.parent.index][1] + coordinates[prev_node.parent.index][3]/2
			if parent_mid > l_bot:
				if r_bot < parent_mid:
					if r_bot < l_mid and r_bot > l_top*1.1:
						prev_node.top = curr_node
						curr_node.parent = prev_node
						prev_node = curr_node
					elif r_top > l_mid:
						prev_node.child = curr_node
						curr_node.parent = prev_node
						prev_node = curr_node
					elif r_top < l_mid:
						prev_node.sibling = curr_node
						curr_node.parent = prev_node.parent
						prev_node = curr_node
				else:
					# print("HEREHERE")
					return self.locate_and_label(prev_node.parent, curr_node)
				return prev_node
			elif parent_mid < l_top:
				if r_top > parent_mid:
					if r_top < l_bot*0.9 and r_top > l_mid:
						prev_node.child = curr_node
						curr_node.parent = prev_node
						prev_node = curr_node
					elif r_bot < l_mid:
						prev_node.top = curr_node
						curr_node.parent = prev_node
						prev_node = curr_node
					elif r_top > l_mid:
						prev_node.sibling = curr_node
						curr_node.parent = prev_node.parent
						prev_node = curr_node
				else:
					# print("HOHOHO")
					return self.locate_and_label(prev_node.parent, curr_node)
				return prev_node
			else:
				pass
