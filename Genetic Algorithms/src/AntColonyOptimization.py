import time
from Maze import Maze
from PathSpecification import PathSpecification
import random
from Route import Route
from Ant import Ant

# Class representing the first assignment. Finds the shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

     # Loop that starts the shortest path process
     # @param path_specification of the route we wish to optimize
     # @return ACO optimized route

    def find_shortest_route(self, path_specification):
        # Initialize pheromone levels
        self.maze.reset()

        # Loop over generations
        for i in range(self.generations):
            # Create a new set of ants for this generation
            ants = [Ant(self.maze, path_specification) for _ in range(self.ants_per_gen)]

            # Move all ants through the maze
            for ant in ants:
                route = ant.find_route()
                self.maze.add_pheromone_routes(route, self.q)

        # Find best route from all generations
        best_route = None
        best_length = float('inf')
        for i in range(self.generations):
            ants = [Ant(self.maze, path_specification) for _ in range(self.ants_per_gen)]
            for ant in ants:
                route = ant.find_route()
                length = route.length()
                if length < best_length:
                    best_route = route
                    best_length = length

        return best_route