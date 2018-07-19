
import networkx as nx 
import matplotlib.pyplot as plt 
from timeit import default_timer as timer 
import json
import numpy as np 
import random
import sys
import argparse

"""
Some functions to brute force the graph coloring problem (call back_track(graph))
or the search for the minimum of the hamiltonian (spin_search(graph))
"""


# features to add
# 2. visualize the colored graph

MAX_DEG = 5		# the maximum degree of any vertex
VERT = 32	# number of vertices of the graph
MAX_ITER = 5000		# the number of iterations the greedy algorithm solves for
FIXED_COLOR = 3		# the fixed (constant) number of colors we try to color the graph with
DROP = 0.1		# the fraction of edges randomly dropped in the optional sparsifying step

def generate_J(graph, fixed_color):
	num_vert = graph.number_of_nodes()
	N = num_vert * fixed_color
	J = np.zeros((N, N))

	##### add the first term (u = v, i != j)
	for v in range(num_vert):
		for i in range(fixed_color):
			for j in range(fixed_color):
				if j != i:
					J[v*fixed_color+i, v*fixed_color+j] += 1

	##### add the second term ((uv) in E, i == j)
	edge_list = list(graph.edges())
	for edge in edge_list:
		u, v = edge
		for i in range(fixed_color):
			J[u*fixed_color+i, v*fixed_color+i] += 1
	return J


def generate_h(graph, fixed_color):
	num_vert = graph.number_of_nodes()
	N = num_vert * fixed_color
	h = np.zeros(N)
	##### first term
	h += 2*(fixed_color-2)

	##### second term
	edge_list = list(graph.edges())
	for edge in edge_list:
		u, v = edge
		for i in range(fixed_color):
			h[u*fixed_color+i] += 1
			h[v*fixed_color+i] += 1
	return h

def build_adj_matrix(graph):
	n = graph.number_of_nodes()
	adj_matrix = np.zeros((n,n))

	##### diagonal
	for i in range(n):
		adj_matrix[i,i] = 1

	##### off-diagonal	
	edge_list = list(graph.edges())
	for edge in edge_list:
		u, v = edge    # u and v are 0 indexed
		adj_matrix[u, v] = 1
		adj_matrix[v, u] = 1
	return adj_matrix

def issafe(k, c, adj_matrix, x):   # check if node k can be colored with color c
	flag = True
	n = len(x)
	for i in range(n):
		if adj_matrix[k, i] == 1 and c == x[i]:
			flag = False

	return flag

def color_node(k, adj_matrix, x):
	n = len(x)
	# if x[n-1] != 0: # if the last node is colored 
	# 	print 'last node colored'
	# 	return 
	for c in range(1, FIXED_COLOR+1):
		if x[n-1] != 0: # if the last node is colored 
			# print 'last node colored'
			break 
		else: # otherwise clear the rest of the x[]
			for j in range(k, n):
				x[j] = 0

		# print 'trying to color ', k , ' with ', c, ', ', x
		if (issafe(k, c, adj_matrix, x)):
			x[k] = c
			# print k, ' is colored ', c
			if k+1 < n:
				color_node(k+1, adj_matrix, x)
			# else:
			#  	return

def back_track(graph):
	n = graph.number_of_nodes()
	x = np.zeros(n)

	adj_matrix = build_adj_matrix(graph)
	#x[0] = 1
#	print issafe(3, 1, adj_matrix, x) 
	start = timer()
	color_node(0, adj_matrix, x)
	end = timer()
	return x, end-start

def calc_obj(x, J, h):
	return x.T.dot(J).dot(x) + h.dot(x.T)

def search_spin(graph):
	N = graph.number_of_nodes()	# number of nodes
	x = np.zeros(N*FIXED_COLOR) * 2 - 1	# spin array
	n = len(x)	# number of spins

	J = generate_J(graph, FIXED_COLOR)
	h = generate_h(graph, FIXED_COLOR)

	# initialize min_loss and opt_config
	min_loss = sys.maxint
	opt_config = np.zeros(n)

	""" 
	convert numbers in the range [0, 2^n) into binary, and then that binary 
	number into an array of spins --> x
	"""
	start = timer()
	for num in range(np.power(2, n)-1, -1, -1):
		x = np.array(list(np.binary_repr(num).zfill(n))).astype(np.int8) * 2 -1
		# calculate the loss
		temp_loss = calc_obj(x, J, h)
		# if the loss is a new low, save as min_loss and save the spin config
		if temp_loss < min_loss:
			#print min_loss, '->', temp_loss, ', ', x
			min_loss = temp_loss
			opt_config = x
	end = timer()

	return min_loss, opt_config, end-start

def spin_to_color(x):
	x = np.reshape(x, (len(x)/FIXED_COLOR, FIXED_COLOR))
	color_list = -1 * np.ones(x.shape[0]) 
	for k in range(x.shape[0]):
		for i in range(x.shape[1]):
			if x[k,i] == 1:
				color_list[k] = i
	return color_list


def iter_search2(J, h):
	n = len(h)	# number of spins
	min_loss = sys.maxint
	opt_config = np.zeros(n)

	for num in range(np.power(2, n)-1, -1, -1):
		x = np.array(list(np.binary_repr(num).zfill(n))).astype(np.int8) * 2 -1
		temp_loss = calc_obj(x, J, h)
		if temp_loss < min_loss:
			print min_loss, '->', temp_loss, ', ', x
			min_loss = temp_loss
			opt_config = x

	return min_loss, opt_config


def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--export", help="optional export rudy file",action="store_true")
	# parser.add_argument("--sparsify", help="optional sparsify by dropping a fraction of the edges", action = "store_true")
	# parser.add_argument("--printh", help="optional print h",action="store_true")
	# parser.add_argument("--printJ", help="optional print J",action="store_true")
	# parser.add_argument("--draw", help="draw the vertices and edges of the graph",action="store_true")
	# args = parser.parse_args()
	
	##### generate a random graph with EACH vertex connected to MAX_DEG number of edges
	# g = nx.Graph()
	# g.add_nodes_from([0, 1, 2, 3])

	# g.add_edges_from([(0,1), (1,2), (2,3), (3,0), (1,3)])

	# nx.draw_circular(g, with_labels=True, font_weight='bold')
	# #plt.show()
	pass

	# adj_matrix = build_adj_matrix(g)
	# print(adj_matrix)



	# x = back_track(g)
	# if x[-1] == 0:
	# 	print 'There is no way to color the graph with', FIXED_COLOR, 'colors.'
	# else: 
	# 	print 'The first coloring found is', x, '.'

	#min_loss, opt_config = search_spin(g)
	#x = [1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1]
	#print spin_to_color(x)
	# deg_list = list(rando.degree())
	# max_deg = max(deg_list[i][1] for i in xrange(rando.number_of_nodes()))

	# print "============================================================"
	# print "Generated a random graph with MAX_DEG = ", MAX_DEG, ", VERT = ", VERT, "."

	# ####################  (OPTIONAL) sparsify!
	# # DROP == the fraction of edges deleted, as specified at the beginning of the script  
	# if args.sparsify:
	# 	new_rando = sparsify(rando, int(DROP*rando.number_of_edges()))
	# else:
	# 	new_rando = rando 

	# J = generate_J(new_rando, FIXED_COLOR)
	# h = generate_h(new_rando, FIXED_COLOR)

	# ####################  (OPTIONAL) print h 
	# if args.printh:
	# 	print h

	# ####################  (OPTIONAL) print and plot J
	# if args.printJ:
	# 	plt.imshow(J)
	# 	plt.colorbar()
	# 	plt.show()

	# 	np.set_printoptions(threshold=np.inf)
	# 	print J

	# ####################  (OPTIONAL) export to rudy form
	# if args.export:
	# 	export(new_rando, h, J)

	# ##### solve for the chromatic number and time it
	# chro_num, TTS, opt_coloring = chrome_num_random_algo(new_rando, VERT, MAX_ITER)
	# print ""
	# print "The time it takes to color the random graph is: ", TTS, " s."
	# print "The chromatic number gamma = ", chro_num
	# print "============================================================"
	
	# #### (OPTIONAL) print the coloring
	# print "The coloring = ", opt_coloring

	# # plot the graph
	# if args.draw:
	# 	nx.draw_shell(new_rando, with_labels=True, font_weight='bold')
	# 	plt.show()

	# ##### save to file "datafile.json"
	# data = {"Graph": {}, "Chrom":{}}
	# data["Graph"]["MAX_DEG"] = MAX_DEG
	# data["Graph"]["VERT"] = VERT
	# data["Graph"]["edges"] = list(new_rando.edges())
	# data["Chrom"]["chro_num"] = chro_num
	# data["Chrom"]["TTS"] = TTS
	# data["Chrom"]["MAX_ITER"] = MAX_ITER
	# data["Chrom"]["opt_coloring"] = opt_coloring


	# filename = 'vert'+str(VERT)+'_edge'+str(new_rando.number_of_edges())+'_maxdeg'+str(MAX_DEG)+ \
	# 			'_fixedColor'+str(FIXED_COLOR) + '_drop'+str(int(DROP*100))+'_sln'
	# with open(filename+'.json', 'w') as outfile:
	# 	json.dump(data, outfile)
	# 	outfile.write("\n")

main()