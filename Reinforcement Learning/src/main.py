from Maze import Maze
from Agent import Agent
from MyQLearning import MyQLearning
from MyEGreedy import MyEGreedy

# Load the maze
# Toy maze:
# file = "../data/toy_maze.txt"

# Easy maze:
file = "../data/easy_maze.txt"
maze = Maze(file)

# Set the reward at the bottom right to 10
terminal_goal_reward = 10
# Toy maze

# terminal_state = maze.get_state(9, 9)
terminal_state = maze.get_state(maze.w-1, maze.h-1)
maze.set_reward(terminal_state, terminal_goal_reward)

# Create a robot at starting and reset location (0,0) (top left)
robot = Agent(0, 0)

# Make a selection object (you need to implement the methods in this class)
selection = MyEGreedy()

# Make a Qlearning object (you need to implement the methods in this class)
learn = MyQLearning()

# Start: My additions
epsilon = 0.1
alfa = 0.7
gamma = 0.9
episode = 1
number_of_episodes = 150
# End: My additions

# keep learning until you decide to stop
# For number of episodes or convergence:
while episode <= number_of_episodes:
    # TODO implement the action selection and learning cycle
    # TODO figure out a stopping criterion
    
    # Start of the episode
    step = 1
    # number_of_steps = 30000
    number_of_steps = 60000

    while robot.get_state(maze) is not terminal_state and step <= number_of_steps:
        state = robot.get_state(maze)
        action = selection.get_egreedy_action(robot, maze, learn, epsilon)
        next_state = robot.do_action(action, maze)
        r = maze.get_reward(next_state)
        
        best_action = selection.get_best_action(robot, maze, learn)
        learn.update_q(state, action, r, next_state, best_action, alfa, gamma)
        
        step += 1
    
    robot.reset()
    episode += 1

# print(robot.num_of_steps_per_episode)
robot.plot_learning()

    