import array
import collections

class disjoint_set:
    def __init__(self, size=10):
        self.parent = array.array('L', [i for i in range(0, size)])
    
    def reset(self):
        for i in len(self.parent):
            self.parent[i] = i
    
    def find_parent(self, p):
        while self.parent[p] != p:
            p = self.parent[p]
        return p

    def union(self, lhs, rhs):
        lhs_parent = self.find_parent(lhs)
        rhs_parent = self.find_parent(rhs)

        if lhs_parent == rhs_parent:
            return False
        
        self.parent[lhs_parent] = rhs_parent
        return True

class graph:
    def __init__(self):
        self.edges = dict()

    def add_edge(self, lhs, rhs):
        if lhs not in self.edges:
            self.edges[lhs] = set()
        
        if rhs not in self.edges:
            self.edges[rhs] = set()
        
        self.edges[lhs].add(rhs)
        self.edges[rhs].add(lhs)
    
    def path_is_longer(self, lhs, rhs, length):
        return len(self.bfs(lhs, rhs)) > length
    
    def bfs(self, lhs, rhs):
        if lhs not in self.edges or rhs not in self.edges:
            return []

        if rhs in self.edges[lhs]:
            return [lhs, rhs]

        visited = {}
        deq = collections.deque([lhs])
        while len(deq) > 0:
            p = deq.pop()
            if p == rhs:
                path = [p]
                while p != lhs:
                    p = visited[p]
                    path.append(p)
                path.append(lhs)
                path.reverse()
                return path

            for q in self.edges[p]:
                if q not in visited:
                    visited[q] = p
                    deq.append(q)
        
        return []


            

