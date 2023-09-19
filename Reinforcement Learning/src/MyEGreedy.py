import numpy as np

class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        # TODO to select an action at random in State s
        valid_actions = maze.get_valid_actions(agent)
        return np.random.choice(valid_actions)

    def get_best_action(self, agent, maze, q_learning):
        # TODO to select the best possible action currently known in State s.
        valid_actions = maze.get_valid_actions(agent)
        state = agent.get_state(maze)
        action_values = q_learning.get_action_values(state, valid_actions)
        action_value_pairs = zip(valid_actions, action_values)
        
        # Preventing the agent from being biased to select the same action in get_best_action() over
        # and over before it has learned something about the environment:
        # If there are multiple actions with same max select uniformally randomly
        max_value = np.max(action_values)
        max_actions = []
        for action, value in action_value_pairs:
            if value == max_value:
                max_actions.append(action)

        return np.random.choice(max_actions)

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        # TODO to select between random or best action selection based on epsilon.
        best_action = self.get_best_action(agent, maze, q_learning)
        random_action = self.get_random_action(agent, maze)

        return np.random.choice([random_action, best_action], p=[epsilon, (1-epsilon)])
