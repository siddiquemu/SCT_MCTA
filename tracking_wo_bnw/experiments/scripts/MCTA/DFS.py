#T_a: 0,1,2,3,4
#T_p: 5,6,7,8
#generate graph g_mc

vertices = ['0','1','2','3','4','5','6','7','8']
g_mc  = {keys: [] for keys in vertices}
edges = {keys: [] for keys in vertices}
for i,vi in enumerate(vertices):
    for j,vj in enumerate(vertices):
        if i != j:
            g_mc[vi].append(vj)
            if (vi,vj) in [('0','6'),('0','8'),('2','7'),('2','6'),('3','7')]:
                edges[vi].append(vj)


def dfs(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n, visited)
    return visited

visited = dfs(edges,'0', [])
print(visited)
print(edges)