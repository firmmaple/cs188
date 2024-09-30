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


from typing import Union
from util import manhattanDistance
import random
import util

from game import Agent, Directions
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(f"Chose action: { legalMoves[chosenIndex]} with score: {bestScore}")

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # score = successorGameState.getScore()
        # getScore() returns the total score earned by Pacman.
        # It seems that initializing the score to 0 is slightly better
        # than using the score from the game state.
        score = 0

        # Calculate manhattan distance to the closest ghost
        distanceToGhosts = [
            manhattanDistance(newPos, ghostState.getPosition())
            for ghostState in newGhostStates
        ]
        distanceToClosestGhost = max(min(distanceToGhosts), 1)  # Avoid division by zero

        # Penalize being too close to a non-scared ghost
        if min(newScaredTimes) < 3:
            if distanceToClosestGhost < 4:
                score -= 20 / distanceToClosestGhost
                # Check if Pacman is surrounded by walls on three sides (seems not necessary)
                walls = successorGameState.getWalls()
                x, y = newPos
                wallCount = sum(
                    [
                        walls[x + dx][y + dy]
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    ]
                )
                if wallCount >= 3:
                    score -= 100

        # Calculate manhattan distance to the closest food
        distanceToClosestFood = [
            manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()
        ]
        if len(distanceToClosestFood) > 0:
            # Avoid division by zero
            distanceToFood = max(min(distanceToClosestFood), 1)
            score += 10 / distanceToFood

        # Penalize for remaining food
        score -= successorGameState.getNumFood() * 10

        return score


def scoreEvaluationFunction(currentGameState: GameState) -> float:
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depthLimit = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agentIndex: int, gameState: GameState, curDepth: int):
            if gameState.isWin() or gameState.isLose() or curDepth == self.depthLimit:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman is always agent 0
                return maxValue(agentIndex, gameState, curDepth)
            else:
                return minValue(agentIndex, gameState, curDepth)

        def maxValue(agentIndex: int, gameState: GameState, curDepth: int):
            bestValue = float("-inf")
            bestAction = None
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                successorValue = minimax(nextAgentIndex, successorState, nextDepth)
                if successorValue > bestValue:
                    bestValue = successorValue
                    bestAction = action
            if curDepth == 0:
                return bestAction  # return action instead of value
            return bestValue

        def minValue(agentIndex: int, gameState: GameState, curDepth: int):
            bestValue = float("+inf")
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                bestValue = min(
                    bestValue, minimax(nextAgentIndex, successorState, nextDepth)
                )
            return bestValue

        return minimax(self.index, gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(
            agentIndex: int, gameState: GameState, curDepth: int, alpha, beta
        ):
            if gameState.isWin() or gameState.isLose() or curDepth == self.depthLimit:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman is always agent 0
                return maxValue(agentIndex, gameState, curDepth, alpha, beta)
            else:
                return minValue(agentIndex, gameState, curDepth, alpha, beta)

        def maxValue(agentIndex: int, gameState: GameState, curDepth: int, alpha, beta):
            bestValue = float("-inf")
            bestAction = None
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = alphaBeta(
                    nextAgentIndex, successorState, nextDepth, alpha, beta
                )
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                if bestValue > beta:
                    return bestValue
                alpha = max(alpha, bestValue)
            if curDepth == 0:
                return bestAction  # return action instead of value
            return bestValue

        def minValue(agentIndex: int, gameState: GameState, curDepth: int, alpha, beta):
            bestValue = float("+inf")
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                bestValue = min(
                    bestValue,
                    alphaBeta(nextAgentIndex, successorState, nextDepth, alpha, beta),
                )
                if bestValue < alpha:
                    return bestValue
                beta = min(beta, bestValue)
            return bestValue

        return alphaBeta(self.index, gameState, 0, float("-inf"), float("+inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agentIndex: int, gameState: GameState, curDepth: int):
            if gameState.isWin() or gameState.isLose() or curDepth == self.depthLimit:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman is always agent 0
                return maxValue(agentIndex, gameState, curDepth)
            else:
                return expValue(agentIndex, gameState, curDepth)

        def maxValue(agentIndex: int, gameState: GameState, curDepth: int):
            bestValue = float("-inf")
            bestAction = None
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                successorValue = expectimax(nextAgentIndex, successorState, nextDepth)
                if successorValue > bestValue:
                    bestValue = successorValue
                    bestAction = action
            if curDepth == 0:
                return bestAction  # return action instead of value
            return bestValue

        def expValue(agentIndex: int, gameState: GameState, curDepth: int):
            value = 0
            # the agents move in order of increasing agent index
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = curDepth if nextAgentIndex != 0 else curDepth + 1
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Ghost chooses actions from legal actions uniformly at random
                value += expectimax(nextAgentIndex, successorState, nextDepth) / len(
                    legalActions
                )
            return value

        return expectimax(self.index, gameState, 0)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
