# Graph must be initialized with vertices amount. Next, edges has to be added.
# DFS visits one entire branch doing backtracking once arrived at the deepest node

# Sokoban board

#   #######            # -> wall
#   #     #            * -> player
#   # # # #            @ -> box
#   #. $@ #            X -> goal
#   #######

# player position is represented by a tuple. For example, (2, 1) means that the player is in the string at position 2
# and character at position 1 as shown in the picture above (bah is a matrix). The same for box and goal positions

# The state is defined by the player and boxes current position. States are unique! I think so xd

# How we can represent a movement? Our player only can move in four directions: up, down, left and right
# We can think a movement like the increment or decrement of a position :
# (-1, 0) going up; (1, 0) down; (0, -1) left; (0, 1) right

# How we can represent the action of moving a box? If we do a movement and our new position is coincident with
# a box position... well it means that the box has to be moved in the direction that the player moved. Anyway
# we must check if it is possible to move the box


class DfsSolver:
    def __init__(self, board, player_pos, box_positions, goal_positions):
        self.board = board
        self.player_pos = player_pos
        self.box_positions = set(box_positions)
        self.goal_positions = set(goal_positions)
        self.visited = set()

    def is_solved(self):
        return self.box_positions == self.goal_positions

    def get_possible_moves(self, player_pos):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        possible_moves = []

        for move in directions:
            new_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
            if self.is_valid_move(player_pos, new_pos):
                possible_moves.append(new_pos)

        return possible_moves

    def is_valid_move(self, player_pos, new_pos):
        x, y = new_pos
        if self.board[x][y] == '#':
            # Me la di en la pera
            return False
        if new_pos in self.box_positions:
            # Can we push the box? Imagine right now that we are the box
            next_pos = (x + (x - player_pos[0]), y + (y - player_pos[1]))
            if next_pos in self.box_positions or self.board[next_pos[0]][next_pos[1]] == '#':
                # There is another box or the wall
                return False

    def move(self, player_pos, new_pos):
        # We make the movement effective
        new_box_positions = set(self.box_positions)
        if new_pos in new_box_positions:
            # Moving the box
            next_pos = (new_pos[0] + (new_pos[0] - player_pos[0]), new_pos[1] + (new_pos[1] - player_pos[1]))
            new_box_positions.remove(new_pos)
            new_box_positions.add(next_pos)

        return DfsSolver(self.board, new_pos, new_box_positions, self.goal_positions)

    def dfs(self):
        # The stack is going to persist our frontier states
        stack = [(self.player_pos, self.box_positions)]
        self.visited.add((self.player_pos, frozenset(self.box_positions)))

        while stack:
            # Next movement
            player_pos, box_positions = stack.pop()
            if box_positions == self.goal_positions:
                return True

            possible_moves = self.get_possible_moves(player_pos)
            for move in possible_moves:
                new_state = self.move(player_pos, move)
                if (new_state.player_pos, frozenset(new_state.box_positions)) not in self.visited:
                    self.visited.add((new_state.player_pos, frozenset(new_state.box_positions)))
                    stack.append((new_state.player_pos, new_state.box_positions))

        return False
