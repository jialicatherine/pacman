# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Find the distance between the new position the closest food.
        newFoodPos = newFood.asList()
        closestFoodDis = float('inf')
        # If all food are eaten, win :>
        if len(newFoodPos) == 0:
            return closestFoodDis
        closestFoodDis = min([closestFoodDis]+[util.manhattanDistance(newPos, foodPos) for foodPos in newFoodPos])

        # Find the distance between the new position the closest ghost which is not scared.
        closestGhostDis = float('inf')
        newGhostsPos = [newghostState.getPosition() for newghostState in newGhostStates if newghostState.scaredTimer == 0]
        # If new position will meet a ghost that is not scared, lose :<
        if newPos in newGhostsPos:
            return -float('inf')
        if len(newGhostsPos) != 0:
            closestGhostDis = min([closestGhostDis]+[util.manhattanDistance(newPos, newGhostPos) for newGhostPos in newGhostsPos])

        # It's better if food is close so we add the reciprocal of closest food distance to our score.
        # It's better if ghost is far away so we substract the reciprocal of closest ghost from our score.
        return (successorGameState.getScore() + 1.0/closestFoodDis - 1.0/closestGhostDis)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getMax(self, state, depth):
        # Pacman is always agent index 0
        legalActions = state.getLegalActions(0)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)
        maxV, action = -float('inf'), None
        for a in legalActions:
            successor = state.generateSuccessor(0, a)
            successorValue = self.getMin(successor, 1, depth)
            if successorValue[0] > maxV:
                maxV, action = successorValue[0], a
        return (maxV, action)

    def getMin(self, state, agentIndex, depth):
        legalActions = state.getLegalActions(agentIndex)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)

        minV, action = float('inf'), None
        for a in legalActions:
            successor = state.generateSuccessor(agentIndex, a)
            # If all ghost have moved, pacman move -> take max value
            if agentIndex == state.getNumAgents() - 1:
                successorValue = self.getMax(successor, depth + 1)
            else:
                # ghost will choose the min value for pacman
                successorValue = self.getMin(successor, agentIndex+1, depth)
            if successorValue[0] < minV:
                minV, action = successorValue[0], a
        return (minV, action)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getMax(self, state, depth, a, b):
        # Pacman is always agent index 0
        legalActions = state.getLegalActions(0)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)
        maxV, action = -float('inf'), None
        for la in legalActions:
            successor = state.generateSuccessor(0, la)
            successorValue = self.getMin(successor, 1, depth, a, b)
            if successorValue[0] > maxV:
                maxV, action = successorValue[0], la
            # Pruning
            if maxV > b:
                return (maxV, action)
            # Update the lower bound
            a = max(a, maxV)
        return (maxV, action)

    def getMin(self, state, agentIndex, depth, a, b):
        legalActions = state.getLegalActions(agentIndex)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)

        minV, action = float('inf'), None
        for la in legalActions:
            successor = state.generateSuccessor(agentIndex, la)
            # If all ghost have moved, pacman move -> take max value
            if agentIndex == state.getNumAgents() - 1:
                successorValue = self.getMax(successor, depth + 1, a, b)
            else:
                successorValue = self.getMin(successor, agentIndex+1, depth, a, b)
            if successorValue[0] < minV:
                minV, action = successorValue[0], la
            # Pruning
            if minV < a:
                return (minV, action)
            # Update the upper bound
            b = min(b, minV)
        return (minV, action)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(gameState, 0, -float('inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getMax(self, state, depth):
        # Pacman is always agent index 0
        legalActions = state.getLegalActions(0)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)
        maxV, action = -float('inf'), None
        for a in legalActions:
            successor = state.generateSuccessor(0, a)
            successorValue = self.getExp(successor, 1, depth)
            if successorValue[0] > maxV:
                maxV, action = successorValue[0], a
        return (maxV, action)

    def getExp(self, state, agentIndex, depth):
        legalActions = state.getLegalActions(agentIndex)
        # If the game is finished
        if not legalActions or depth == self.depth:
            return (self.evaluationFunction(state), None)
        expV, action = 0, None
        for a in legalActions:
            successor = state.generateSuccessor(agentIndex, a)
            if agentIndex == state.getNumAgents() - 1:
                successorValue = self.getMax(successor, depth + 1)
            else:
                successorValue = self.getExp(successor, agentIndex+1, depth)
            # Expected value
            expV += successorValue[0]/len(legalActions)
        return (expV, action)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    # Score according to the distance to food
    foodlist = currentGameState.getFood().asList()
    if not foodlist: # all food have been eaten -> win
        closestFoodDis = float('inf')
    else:
        closestFoodDis = 1.0/min([util.manhattanDistance(currPos, foodPos) for foodPos in foodlist])

    # Score according to the distance to scared ghosts
    foodGhostPos = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer > 0]
    if not foodGhostPos: # No scared ghost
        closestFoodGhost = 0
    else:
        DistToFGP = [(ghostState.scaredTimer - util.manhattanDistance(currPos, ghostState.getPosition())) for ghostState in ghostStates if ghostState.scaredTimer > 0]
        closestFoodGhost = max(DistToFGP)

    # Score according to the distance to unscared ghosts
    ghostsPos = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer == 0]
    if currPos in ghostsPos: # Game lost
        closestGhost = -float('inf')
    elif not ghostsPos: # No unscared ghost
        closestGhost = 0
    else:
        closestGhost = -1.0/min([util.manhattanDistance(currPos, ghostPos) for ghostPos in ghostsPos])

    return (currentGameState.getScore() + closestFoodDis + closestFoodGhost + closestGhost)

# Abbreviation
better = betterEvaluationFunction
