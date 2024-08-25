# 8 puzzle Problem 

## Problem Definition

Given a 3×3 board with 8 tiles (each numbered from 1 to 8) and one empty space, the objective is to place the numbers to match the final configuration using the empty space. We can slide four adjacent tiles (left, right, above, and below) into the empty space.

#### Random initial state

| 5 | 7 | 3 |
|---|---|---|
| 2 | 8 |   |
| 1 | 6 | 4 |

#### Final state

| 1 | 2 | 3 |
|---|---|---|
| 8 |   | 4 |
| 7 | 6 | 5 |

We define a problem state as the location of each part in a cell, including the empty space. If we apply an action to a state, we obtain a new state. Being an action the movement of the empty space to left, right, up or down.

## Solving the problem by search solving algorithms

#### Depth-first search (DFS) & Breadth-first search (BFS)

DFS explores as far as possible along each branch before backtracking. It is an exhaustive search technique used to traverse or search tree or graph data structures. It starts at the root node and explores as far as possible along each branch before backtracking.  BFS explores all neighbor nodes at the present depth before moving on to nodes at the next depth level. It is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root and explores the neighbor nodes first, before moving to the next level neighbors.

Limitations:

* DFS: Can get stuck exploring deep paths that do not lead to the goal, potentially leading to excessive memory usage and slow performance.
* BFS: Explores all nodes at the present depth before moving to the next depth level, which can be inefficient as it does not prioritize more promising paths.

#### A* algorithm

Unlike the previous ones, this algorithm implements an informed search. In this case, the algorithm uses heuristics. A heuristic value tells the algorithm which path will provide the solution as early as possible. The heuristic function is used to generate this heuristic value. Different heuristic functions can be designed depending on the searching problem. So we can conclude that Heuristic search is a technique that uses a heuristic value for optimizing the search. So, in this way, when A* is being applied, The next node chosen from the open list is based on its f score, the node with the least f score is picked up and explored.  The definition of f is as follows: **f(n) = g(n) + h(n)**.

Where f(n) = actual distance so far + estimated distance remaining (heuristic value)

Thinking about possible heuristics, here we have some examples:

1. The number of misplaced tiles by comparing the current state and the goal state
2. The sum of the distances of each tile from its goal position ("Manhattan distance")
3. As above but takes into account the number of direct adjacent tile inversions present. Having many adjacent inversions suggests that the current configuration is far from the solution, as you will have to make additional moves to correct each inversion, which makes it more accurate by reflecting the “effort” needed to solve the puzzle.


#### Current state

| 5 | 7 | 3 |
|---|---|---|
| 2 | 8 |   |
| 1 | 6 | 4 |

#### Final state

| 1 | 2 | 3 |
|---|---|---|
| 8 |   | 4 |
| 7 | 6 | 5 |

Heuristic 1. tiles 5, 1, 4, 2, 7 and 8 are out of place, so estimate = 6. <br>
Heuristic 2. estimate = 4 + 3 + 0 + 2 + 1 + 2 + 0 + 1 = 13. <br>
Heuristic 3. in this case we don't have any case of direct reversal, so the cost is 13.

## Bibliography

1. GeeksforGeeks. *8 Puzzle Problem using Branch and Bound*. GeeksforGeeks. Available in: [https://www.geeksforgeeks.org/8-puzzle-problem-using-branch-and-bound/](https://www.geeksforgeeks.org/8-puzzle-problem-using-branch-and-bound/)

2. Kumar, Prateek. *Solving 8-Puzzle using A* Algorithm*. Good Audience. Available in: [https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288](https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288)

3. Marshall, J. *Heuristic Search*. Sarah Lawrence College. Available in: [http://science.slc.edu/~jmarshall/courses/2005/fall/cs151/lectures/heuristic-search/](http://science.slc.edu/~jmarshall/courses/2005/fall/cs151/lectures/heuristic-search/)
