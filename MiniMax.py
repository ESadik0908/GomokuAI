'''
We tried just a greedy search initially which wighted the board based off a similar heuristic to the one in update_weights,
however, after implimenting a minimax search, we found minimax to outpreform a greedy search appraoch.

Our greedy search implementation on its own does not perform as well because it doesn't take into account future play.

Because of the limited time we have experimented with prioritising depth vs breadth
and found that exploring less initial moves deeper produces better results.

With more time it would perform better, as we could check more game lines. We would also then be 
able to implement some openings to speed up the early game where moves would be evaluated to a deep depth.

To further develop the algorithm, we could add more shapes to look for when evaluating the board so the ai can better
understand complex game states.
'''


import numpy as np

import sys
from misc import legalMove
from gomokuAgent import GomokuAgent

# represents the 8 cardinal directions we can move in. These are paired up with their oposite direction value
moveSet = np.array([
    [[-1,0], [1, 0]], #up, down
    [[0, -1], [0, 1]], #left, right
    [[-1,-1], [1,1]], #up left, down right
    [[-1, 1], [1, -1]] #down left, up right
    ])

# refers to a node or single spot the board
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = 0
        self.owner = 0
        self.threat = [0] 


# initialising localBoard of nodes
localBoard = [[Node(x, y) for y in range(11)] for x in range(11)]


def update_board(BOARD_SIZE, board):
  """
  Read the board and update the nodes in the local board to represent the board passed in.
  """
  for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
          localBoard[x][y].owner = board[x, y]
          localBoard[x][y].threats = [0]
          if localBoard[x][y].owner != 0:
            localBoard[x][y].value = -99

def update_weights(player):
  """
    Loops through every node in the board and check each pair of directions (up down, left right, diag down, diag up), call the 
    count_connected_cells function to check for the playes stones as well as the enemies. Update the nodes value to the value output
    by count_connected_cells. Also track how many stones are in each connected direction (e.g up and down) if there are stones in 
    both directions, sum them, if this is 3 or more for enemy stones, play this node (prevent loss). If this is 4 or more for friendly 
    stones, play this node (win). 

  Arguments:
    player: the agents player id

  """
  for row in localBoard:
    for node in row:  #loop thorugh whole board
      if node.owner == 0: # if node has no token
        for move in moveSet: #check all move directions
          friendlyStones = []
          enemyStones = []
          for dir in move:

            enemyValue, enemyCount = count_connected_cells_weights(node, dir[0], dir [1], -player, player)
            friendlyValue, freindlyCount = count_connected_cells_weights(node, dir[0], dir [1], player, player)
            
            if freindlyCount == 4:
              reset_weights()
              node.value = sys.maxsize / 2
              return
            
            if enemyCount == 4:
              reset_weights()
              node.value = sys.maxsize / 2
              return
            
            # append counts for each pair of directions to an array
            enemyStones.append(enemyCount)
            friendlyStones.append(freindlyCount)

            node.value += enemyValue        
            node.value += friendlyValue
            
            
            if enemyCount >= 2:
              node.value += enemyCount * enemyCount #s quared to incentivise building lines
            if freindlyCount >= 2:
              node.value += freindlyCount * freindlyCount # squared to incentivise building lines

          
          if 0 not in friendlyStones: # if node is surounded by tokens
            total = sum(friendlyStones)
            if total >= 4:
              reset_weights()
              node.value = sys.maxsize / 2
              return
          
          if 0 not in enemyStones:
            total = sum(enemyStones)
            if total >= 4:
              reset_weights()
              node.value = sys.maxsize / 2
              return
        
          if 0 not in enemyStones:
            total = sum(enemyStones)
            if total >= 3:
              node.value += 5000
          
          if 0 not in friendlyStones:
            total = sum(friendlyStones)
            if total >= 3:
              node.value += 10000

                  
def count_connected_cells_weights(node, dx, dy, owner, player):
    """
      Generates the weights for a given node on the board by counting how many stones there are in a direction.
      Weights are modified by if there are 3 or 4 stones in a row and if they are blocked on one end or at the 
      edge of the bord. Value tracks the weight and count tracks the number of stones in a row.
    
    Arguments:
      node: the node we are checking from
      dx: the x direction to check in
      dy: the y direction to check in
      owner: the the type of stone we are checking for (1 or -1)
      player: the agents player id
    
    Returns:
      Value: the weight for the node
      Count: number of connected stones of type owner in the given direction
    """
    count= 0
    inc = 1
    if owner == player: # weight inc higher for own tokens
      inc = 2

    value = 0
    x, y = node.x, node.y
    while True:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= len(localBoard) or y >= len(localBoard):
            #if we reach the edge of the board and the count is 4, set the nodes value to half the max int,
            #if we are counting freindly stones and find a 4 then reset the weights set the value to the maximum
            if count >= 4:
              value = sys.maxsize / 3 #we weight as very high but lower than if the tokens are ours

              if player == owner:
                reset_weights()
                value = sys.maxsize / 2
                break
              
            break

        if localBoard[x][y].owner != owner:
            #if the owner doesn't match and the next stone is ours and the count is 4 or more then reset the weights and
            #set the weight of this node to the max int value
            if localBoard[x][y].owner == player: 
              if count >= 4:
                reset_weights()
                value = sys.maxsize / 2
                break
            
            #if the node after the end of the connected stones is blank and the count is more than 2 then multiply the value by
            #1000 if we are counting enemy stones (to make sure we block open 3's) and by 100 if we are counting friendly stones
            if localBoard[x][y].owner == 0:
              if count > 2:
                if owner != player:
                  value = 5000
                if owner == player:
                  if player == 1:
                    value = 50000
                  if player == -1:
                    value = 500
            break
        value += inc
        count+= 1
    return value, count


def count_connected_cells(board, node, dx, dy, owner, player):
    '''
    count the number of connected cells for a given location 
    we make sure to count in line and either side of the cell

    Arguments:
    board : current board state
    node : given location on board
    dx : x direction to move in
    dy : y direction to move in
    owner : type of stone we are checking for
    player : deprecated

    Returns:
    count : count of number of cells connected
    stoneShape : string coresponding to the shape of the connected cells identified
    '''
    count= 0
    stoneShape = None
    x, y = node.x, node.y
    while True:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= len(board) or y >= len(board):
          if count == 2:
            stoneShape = "dead_two"
          if count == 3:
            stoneShape = "dead_three"
          if count == 4:
            stoneShape = "dead_four"
          break

        if board[x][y].owner != owner:
          if board[x][y].owner == 0:
            if count == 2:
              stoneShape = "live_two"
            if count == 3:
              stoneShape = "live_three"
            if count == 4:
              stoneShape = "live_four"
          
          if board[x][y].owner == -owner:
            if count == 2:
              stoneShape = "dead_two"
            if count == 3:
              stoneShape = "dead_three"
            if count == 4:
              stoneShape = "dead_four"
          break

        count += 1
    return count, stoneShape



def eval_board(board, player):
  '''
  Calls eval_player for self and enemy player

  Arguments:
  board : current board state
  player : id for self

  Returns:
    single cost combining the weights of self and enemy
  '''

  friendlyCost = eval_player(board, player)
  enemyCost = eval_player(board, -player)
            
  cost = friendlyCost - (enemyCost * 0.9)  # we take away enemy cost here
  
  return cost



def eval_player(board, player):
  '''
  evaluates the quality of the board for agiven player

  we only eval the given players tokens, and then call the function again with -player to get enemy token evals.

  Arguments:
  board : current board state
  player : id for self

  Returns:
  single value weighting for a given board

  '''
  five_in_a_row = 0
  live_four = 0
  dead_four = 0
  live_three = 0
  dead_three = 0
  live_two = 0
  dead_two = 0

  cost = 0
  for row in board:
    for node in row:
      if node.owner == 0:
        for move in moveSet:
          friendlyStones = []
          for dir in move:
            
            count_friendly, stoneShape = count_connected_cells(board, node, dir[0], dir[1], player, player)
            if count_friendly == 0:
              continue
            
            if count_friendly >= 5:
              five_in_a_row += 1
            

            # dead refers to if it is blocked or not either by enemy token or wall
            # live refers to shapes that are open

            # broken and unbroken are weighted the same

            if stoneShape == "dead_two":
              dead_two += 1
              
            if stoneShape == "live_two":
              live_two += 1
            
            if stoneShape == "dead_three":
              dead_three += 1
            
            if stoneShape == "live_three":
              live_three += 1
            
            if stoneShape == "dead_four":
              dead_four += 1
            
            if stoneShape == "live_four":
              live_four += 1
          
          if 0 not in friendlyStones:
            total = sum(friendlyStones)
            if total == 4:
              dead_four += 1
            
            if total == 3:
              dead_three += 1
            
            if total == 2:
              dead_two += 1    
  
  if five_in_a_row != 0: #if it is not 0 then there is a 5 in a row present
    cost = sys.maxsize / 2
    return cost
  
  if live_four >= 1:
    cost += 15000
  
  # these are combinations that are unloseable
  if (live_three >= 2) or (dead_four >= 2) or (dead_four == 1 and live_three == 1):
    cost += 10000
  
  if live_three != 0:
    cost += 5000
  
  if dead_four != 0:
    cost += 1000
    
 
  return cost

def reset_weights():
  """
  Reset the board weights to be manhatten distance
  """
  for row in localBoard:
      for node in row:
        if node.owner == 0:
          x_dist = abs(node.x - 5)
          y_dist = abs(node.y - 5)
          dist = x_dist + y_dist
          node.value = 5 - dist


def game_over(player, board):
  '''
  Return true if gameover has been reached by either player.
  used for stoping tree exploration.
  '''
  for row in board:
    for node in row:
        for move in moveSet:
          for dir in move:

            count_enemy, threat_type_enemy = count_connected_cells(board, node, dir[0], dir[1], -player, player)

            count_friendly, threat_type_friendly = count_connected_cells(board, node, dir[0], dir[1], player, player)

            if count_friendly >= 5 or count_enemy >= 5:
              return True


def best_move(player):
  '''
  For all moves created by get_top_x we create a tree using minimax and work out the best move given the resulting trees
  Arguments:
  player: id for self
  Returns:
  tuple containing the best x,y of a node
  '''
  bestScore = -sys.maxsize
  bestMove = None
  best_nodes = get_top_x(2, player)
  for node in best_nodes:
    print("p: ", player, ' Node', node.x, node.y, 'Value', node.value)
    if node.value == sys.maxsize / 2:
      return (node.x, node.y)
    node.owner = player
    score = minimax(localBoard, 9, False, -sys.maxsize, sys.maxsize, player)
    print("\n", player, " eval ", score)
    node.owner = 0
    if score > bestScore:
      bestScore = score
      bestMove = (node.x, node.y)

  if bestMove == None:
    for row in localBoard:
      for node in row:
        if node.owner == 0:
          bestMove = (node.x, node.y)
          break

  return bestMove


def get_top_x(x, player):
  '''
  Generate the best x moves for a player given a board.


  Arguments : 
  x : number of moves to return
  player : id for self

  Returns:
  returns an array of the best nodes with x length created via updating the weights of the board.
  '''
  reset_weights()
  update_weights(player)
  nodes = []
  for row in localBoard:
    for node in row:
      if node.owner == 0:
        nodes.append(node)
  
  bestX = sorted(nodes, key=lambda x: x.value, reverse=True)
  bestX = bestX[:x]
  return bestX



def minimax(board, depth, isMaximizing, a, b, player):
    '''
    steps through tree of boards maximising then minimising cost function
    to get best self move then best enemy move, eventually returning the
    best initial choice that has the best forced outcome
    D depth down in the tree

    Arguments:

    Board : starting board is passed into the function
    depth : depth we will cap exploration to
    isMaximising : boolean value representing player1 or player 2
    a : alpha value for pruning tree paths that get worse
    b : beta value for above
    player : represents our ai id -player represents enemy id


    Returns:
    returns the best score found from tree traversal
    '''
    if game_over(player, board) == True or depth == 0:
        return eval_board(board, player)
  
    # player 1
    if isMaximizing == True:
        bestScore = -sys.maxsize
        best_nodes = get_top_x(2, player)
        for node in best_nodes:
          localBoard[node.x][node.y].owner = player
          score = minimax(board, depth-1, False, a, b, player)
          localBoard[node.x][node.y].owner = 0
          bestScore = max(score, bestScore)
          a = max(a, score)
          if b <= a:
              break
        return bestScore
    
    # player 2
    if isMaximizing == False:
        bestScore = sys.maxsize
        best_nodes = get_top_x(2, -player)
        for node in best_nodes:
          localBoard[node.x][node.y].owner = -player
          score = minimax(board, depth-1, True, a, b, player)
          localBoard[node.x][node.y].owner = 0
          bestScore = min(score, bestScore)
          b = min(b, score)
        
        return bestScore
        
class Player(GomokuAgent):
  def move(self, board):
    while True:
      update_board(self.BOARD_SIZE, board)
      return best_move(self.ID)