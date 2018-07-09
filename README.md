The script generate_graph.py generates a random graph with up to MAX_DEG 
number of edges connected to each vertex. The constants of the graph such as 
the number of vertices and MAX_DEG, are specified at the beginning of the 
file "generate_graph.py".

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