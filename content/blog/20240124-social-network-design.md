---
title: 20240124-social-network-design
date: 2024-01-24
tags:
  - system_design
updated: 2024-01-24
---
## Question
 如何設計給Linkedin、Facebook等公司可以處理非常大量資料的data structure？請說明如何設計計算兩個人的最短距離的 (e.g. src->B-C->dst) 演算法
 
## Solutions
### Step 1 Simplify the problem - forget about the millions of users
首先使用簡單方法，先從BFS，為何不是DFS，因為DFS會非常沒有效率
另一種方式是bidirectional BFS，也就是從source跟target同時進行BFS，碰撞到即找到最短路徑
使用visited hash table避免走已經走過的路徑
#### 程式
people: A->B->C
```python
def findPathBid(people, src, dst):
	while True:
		# search from src
		collision = searchLevel(people, src, dst)
		if collision:
			return mergePaths(src, dst, collision)
		# search from dst
		collision = searchLevel(people, dst, src)
		if collision:
			return mergePaths(dst, src, collision)
	return None
```
#### 數學證明
假設每個人有k的朋友，路徑q
BFS: $k+k*k$ , O($k^q$)
Bi BFS: $2k$,  O($k^{\frac{q}{2}})$
e.g.
A->B->C->D->E
BFS: $100^4$
Bi BFS: $2*100^2$
### Step 2 Handle the Million of Users
太多資料可能不能只放在同一台機器，所以使用ID替代
1. For each friend ID: `machine_idx = getMachineIDForUser(personID)`
2. Go to machine `#machine_idx`
3. On that machine, do: `friend = getPersonWithID(personID)`
### Optimization: Reduce machine jumps
在不同機器檢索(jump)很消耗資源、可以使用batch jump e.g. A,C,F同時在機器1，進行同時進行A,C,F的query
### Optimization: Smart division of people and machines
人通常會和同一地區的人作為朋友，所以使用地區來做機器存取依據可以減少jump次數

### Breadth-first search usually requires "making" a node as visited. How do you do that in this case?
使用hash table 檢索node id 查看是否被便利

### Other Follow-Up 
- In the real world, servers fail. How does this affect to you?
- How could you take advantage of caching?
- Do you search until the end of the graph(infinite)? How do you decide when to give up?
- In real life, some people have more friends of friends than others, and are therefore more likely to make a path between you and someone else. How could you use use the data to pick where to start traversing?
