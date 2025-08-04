import sys
import csv
from collections import deque
import numpy as np


def read_graph(filename):
    nodes_set = set()
    graph = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                u = int(row[0])
                v = int(row[1])
                nodes_set.add(u)
                nodes_set.add(v)
                if u not in graph:
                    graph[u] = []
                graph[u].append(v)
            except ValueError:
                continue
    return nodes_set, graph

def build_in_degree(graph, nodes):
    in_degree_dict = {node: 0 for node in nodes}
    for u, neighbors in graph.items():
        for v in neighbors:
            if v in in_degree_dict:
                in_degree_dict[v] += 1
    return in_degree_dict

def build_out_degree(graph, nodes):
    out_degree_dict = {}
    for node in nodes:
        if node in graph:
            out_degree_dict[node] = len(graph[node])
        else:
            out_degree_dict[node] = 0
    return out_degree_dict

def is_dag_func(graph, nodes, in_degree_dict):
    in_degree_work = in_degree_dict.copy()
    q = deque()
    for node in nodes:
        if in_degree_work[node] == 0:
            q.append(node)
    count = 0
    while q:
        node = q.popleft()
        count += 1
        if node in graph:
            for neighbor in graph[node]:
                in_degree_work[neighbor] -= 1
                if in_degree_work[neighbor] == 0:
                    q.append(neighbor)
    return count == len(nodes)

def pagerank(graph, nodes, out_degree_dict, n):
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    dangling_nodes = [node for node in nodes if out_degree_dict[node] == 0]
    d = 0.85
    num_iter = 20
    v = np.ones(n) / n

    for _ in range(num_iter):
        total_dangling = sum(v[node_to_index[node]] for node in dangling_nodes)
        base = (1 - d) / n + d * total_dangling / n
        new_v = np.full(n, base)

        for node in graph:
            j = node_to_index[node]
            out_deg = out_degree_dict[node]
            if out_deg == 0:
                continue
            contrib = d * v[j] / out_deg
            for neighbor in graph[node]:
                i = node_to_index[neighbor]
                new_v[i] += contrib

        v = new_v

    return min(v), max(v)

def main():
    if len(sys.argv) != 2:
        print("Usage: ./graph_solution <path_to_graph.txt>")
        sys.exit(1)
    filename = sys.argv[1]
    nodes_set, graph = read_graph(filename)
    n = len(nodes_set)
    if n == 0:
        print("is_dag: true")
        print("max_in_degree: 0")
        print("max_out_degree: 0")
        print("pr_max: 0.000000")
        print("pr_min: 0.000000")
        sys.exit(0)
    nodes = sorted(nodes_set)
    in_degree_dict = build_in_degree(graph, nodes)
    out_degree_dict = build_out_degree(graph, nodes)
    is_dag = is_dag_func(graph, nodes, in_degree_dict)
    max_in_deg = max(in_degree_dict.values())
    max_out_deg = max(out_degree_dict.values())
    pr_min, pr_max = pagerank(graph, nodes, out_degree_dict, n)
    print(f"is_dag: {'true' if is_dag else 'false'}")
    print(f"max_in_degree: {max_in_deg}")
    print(f"max_out_degree: {max_out_deg}")
    print(f"pr_max: {pr_max:.6f}")
    print(f"pr_min: {pr_min:.6f}")

if __name__ == '__main__':
    main()