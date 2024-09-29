# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from typing import Union
import util
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def genericSearchGFS(
    problem: SearchProblem,
    fringe: Union[util.Stack, util.Queue, util.PriorityQueueWithFunction],
) -> list[str]:
    # All of your search functions need to return a list of actions
    # that will lead the agent from the start to the goal.
    # These actions all have to be legal moves (valid directions, no moving through walls).
    startState = problem.getStartState()
    fringe.push((startState, [], 0))
    exploredSet = set()
    while not fringe.isEmpty():
        current_state, actions, current_cost = fringe.pop()
        logging.debug(f"current_state: {current_state}, actions: {actions}")
        if current_state in exploredSet:
            continue
        elif problem.isGoalState(current_state):
            # logging.info(f"Find a solution with actions: {actions}")
            return actions
        exploredSet.add(current_state)
        for successor, action, stepCost in problem.getSuccessors(current_state):
            newActions = actions + [action]
            newCost = current_cost + stepCost
            logging.debug(f"Get successor: {successor}, newActions: {newActions}")
            fringe.push((successor, newActions, newCost))
    logging.warn("Search failed. Cannot find a solution!")
    return []


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return genericSearchGFS(problem, util.Stack())


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return genericSearchGFS(problem, util.Queue())


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueueWithFunction(
        # Priority queue item structure: (state, actions, cost)
        # state: current state (Any type based on the problem)
        # actions: List of actions leading to this state (List[str])
        # cost: Cumulative cost to reach this state (int or float)
        lambda item: item[2]
    )
    return genericSearchGFS(problem, fringe)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueueWithFunction(
        # Priority queue item structure: (state, actions, cost)
        # state: current state (Any type based on the problem)
        # actions: List of actions leading to this state (List[str])
        # cost: Cumulative cost to reach this state (int or float)
        lambda item: item[2] + heuristic(item[0], problem)
    )
    return genericSearchGFS(problem, fringe)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
