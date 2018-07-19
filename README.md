The script generate_graph.py generates a random graph with up to MAX_DEG 
number of edges connected to each vertex. The constants of the graph such as 
the number of vertices and MAX_DEG, are specified at the beginning of the 
file "generate_graph.py".

As a starter, try running in your terminal:

	python generate_graph.py

This will generate a file "[GRAPH_PARAMETERS]_sln.json", which contains the 
details of the graph and the coloring that the greedy algorithm found. 

The script has a few options:
  --import_json		optional import json file
  --export_json		optional export json file
  --export    		optional export rudy file
  --sparsify  		optional sparsify by dropping a fraction of the edges
  --printh    		optional print h
  --printJ    		optional print J
  --draw      		draw the vertices and edges of the graph

For example, by running

	python generate_graph.py --export --sparsify --draw --export_json

we generate the rudy form file "[GRAPH_PARAMETERS].txt", 
randomly delete a fraction of the edges, draw the vertices and edges 
of the graph, and then export the json file with info on the graph and 
solutions.

Solutions are found using 3 methods: 
1. brute-force coloring the graph using back-tracking: 
	if there is no 3-coloring solution, then the list for the solution 
	ends with -1

2. greedy algorithm: 
	colors the graph with the fewest number of colors possible, and 
	returns that solution. If there is no 3-coloring solution, then it 
	still returns a solution.

3. brute-force search for 3-coloring solution by minimizing H: 
	Try to color the graph with 3 colors. If there's no 3-coloring solution,
	return a solution with 3 colors and minimal number of conflicts.