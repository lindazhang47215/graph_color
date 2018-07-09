
import networkx as nx 
import matplotlib.pyplot as plt 
from timeit import default_timer as timer 
import json
import numpy as np 
import random
import sys
import argparse

"""
This script generates a random graph with up to MAX_DEG 
number of edges connected to each vertex. The constants of the graph, such as 
the number of vertices and MAX_DEG, are specified at the beginning of this file 
below.

As a starter, try running in your terminal:

	python generate_graph.py

This will generate a file "[GRAPH_PARAMETERS]_sln.json", which contains the 
details of the graph and the coloring that the greedy algorithm found. 

The script has a few options:
  --export    optional export rudy file
  --sparsify  optional sparsify by dropping a fraction of the edges
  --printh    optional print h
  --printJ    optional print J
  --draw      draw the vertices and edges of the graph

For example, by running

	python generate_graph.py --export --sparsity --draw

we generate the rudy form file "[GRAPH_PARAMETERS].txt" in addition to the json file, 
randomly delete a fraction of the edges, and then draw the vertices and edges 
of the graph. 
"""


# features to add
# 1. load from .json file
# 2. visualize the colored graph

MAX_DEG = 4		# the maximum degree of any vertex
VERT = 7	# number of vertices of the graph
MAX_ITER = 5000		# the number of iterations the greedy algorithm solves for
FIXED_COLOR = 3		# the fixed (constant) number of colors we try to color the graph with
DROP = 0.15		# the fraction of edges randomly dropped in the optional sparsifying step


def chrome_num_random_algo(graph, VERT, MAX_ITER):
	# solve for the chromatic number MAX_ITER times
	chro_num = VERT
	print ""
	print "Running ", MAX_ITER, "iterations..."
	print "Chromatic number changes so far: "
	start = timer()

	##### initial chromatic number 
	opt_coloring = nx.coloring.greedy_color(graph, strategy='random_sequential')
	print chro_num, "->", max(opt_coloring.values())+1
	chro_num = min(chro_num, max(opt_coloring.values())+1)

	for i in range(MAX_ITER):
		colors = nx.coloring.greedy_color(graph, strategy='random_sequential')
		max_color = max(colors.values())+1
		if max_color < chro_num:
			print chro_num, "->", max_color
			chro_num = max_color
			opt_coloring = colors
	end = timer()
	print "Done."

	return chro_num, end-start, opt_coloring

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

def export(graph, h, J):
	num_vert = graph.number_of_nodes()
	num_edge = graph.number_of_edges()

	##### open the file to write
	filename = 'vert'+str(num_vert)+'_edge'+str(num_edge)+'_maxdeg'+str(MAX_DEG)+ \
				'_fixedColor'+str(FIXED_COLOR) + '_drop'+str(int(DROP*100))
	fh = open(filename+'.txt', 'w')

	##### write the number of vertices and number of edges
	fh.write(str(num_vert)+" "+str(num_edge)+'\n')

	##### write the h elements corresponding to each vertex
	for i in range(len(h)):
		fh.write(str(i+1)+" "+str(i+1)+" "+str(int(h[i]))+'\n')

	##### write the J elements corresponding to each edge
	(m, n) = np.shape(J)
	for i in range(m):
		for j in range(n):
			if J[i][j] != 0:
				fh.write(str(i+1)+" "+str(j+1)+" "+str(int(J[i][j]))+'\n')
	fh.close()


def sparsify(graph, num_to_delete):
	edge_list = list(graph.edges())
	for i in range(num_to_delete):
		##### pick an edge to delete
		r = random.randint(0, graph.number_of_edges()-1)
		edge_list.remove(edge_list[r])
	new_graph = nx.Graph()
	new_graph.add_edges_from(edge_list)
	return new_graph

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--export", help="optional export rudy file",action="store_true")
	parser.add_argument("--sparsify", help="optional sparsify by dropping a fraction of the edges", action = "store_true")
	parser.add_argument("--printh", help="optional print h",action="store_true")
	parser.add_argument("--printJ", help="optional print J",action="store_true")
	parser.add_argument("--draw", help="draw the vertices and edges of the graph",action="store_true")
	args = parser.parse_args()
	
	##### generate a random graph with EACH vertex connected to MAX_DEG number of edges
	rando = nx.random_regular_graph(MAX_DEG, VERT)
	deg_list = list(rando.degree())
	max_deg = max(deg_list[i][1] for i in xrange(rando.number_of_nodes()))

	print "============================================================"
	print "Generated a random graph with MAX_DEG = ", MAX_DEG, ", VERT = ", VERT, "."

	####################  (OPTIONAL) sparsify!
	# DROP == the fraction of edges deleted, as specified at the beginning of the script  
	if args.sparsify:
		new_rando = sparsify(rando, int(DROP*rando.number_of_edges()))
	else:
		new_rando = rando 

	J = generate_J(new_rando, FIXED_COLOR)
	h = generate_h(new_rando, FIXED_COLOR)

	####################  (OPTIONAL) print h 
	if args.printh:
		print h

	####################  (OPTIONAL) print and plot J
	if args.printJ:
		plt.imshow(J)
		plt.colorbar()
		plt.show()

		np.set_printoptions(threshold=np.inf)
		print J

	####################  (OPTIONAL) export to rudy form
	if args.export:
		export(new_rando, h, J)

	##### solve for the chromatic number and time it
	chro_num, TTS, opt_coloring = chrome_num_random_algo(new_rando, VERT, MAX_ITER)
	print ""
	print "The time it takes to color the random graph is: ", TTS, " s."
	print "The chromatic number gamma = ", chro_num
	print "============================================================"
	
	#### (OPTIONAL) print the coloring
	print "The coloring = ", opt_coloring

	# plot the graph
	if args.draw:
		nx.draw_shell(new_rando, with_labels=True, font_weight='bold')
		plt.show()

	##### save to file "datafile.json"
	data = {"Graph": {}, "Chrom":{}}
	data["Graph"]["MAX_DEG"] = MAX_DEG
	data["Graph"]["VERT"] = VERT
	data["Graph"]["edges"] = list(new_rando.edges())
	data["Chrom"]["chro_num"] = chro_num
	data["Chrom"]["TTS"] = TTS
	data["Chrom"]["MAX_ITER"] = MAX_ITER
	data["Chrom"]["opt_coloring"] = opt_coloring


	filename = 'vert'+str(VERT)+'_edge'+str(new_rando.number_of_edges())+'_maxdeg'+str(MAX_DEG)+ \
				'_fixedColor'+str(FIXED_COLOR) + '_drop'+str(int(DROP*100))+'_sln'
	with open(filename+'.json', 'w') as outfile:
		json.dump(data, outfile)
		outfile.write("\n")

main()