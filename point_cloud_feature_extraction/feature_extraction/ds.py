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
    def __init__(self, size = 10):
        self.edges = [None] * size

    def add_edge(self, lhs, rhs):
        if self.edges[lhs] == None:
            self.edges[lhs] = set()
        
        if self.edges[rhs] == None:
            self.edges[rhs] = set()
        
        self.edges[lhs].add(rhs)
        self.edges[rhs].add(lhs)
    
    def path_is_longer(self, lhs, rhs, length):
        if self.edges[lhs] == None or self.edges[rhs] == None:
            return False
        
        if rhs in self.edges[lhs]:
            return 2 > length

        visited = {}
        deq = collections.deque([lhs])
        while len(deq) > 0:
            p = deq.popleft()
            if p == rhs:
                path_length = 0
                while p != lhs:
                    p = visited[p]
                    path_length += 1
                return path_length > length
            
            for q in self.edges[p]:
                if q not in visited:
                    visited[q] = p
                    deq.append(q)
            
        return False
    
    def bfs(self, lhs, rhs):
        if self.edges[lhs] == None or self.edges[rhs] == None:
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


            

