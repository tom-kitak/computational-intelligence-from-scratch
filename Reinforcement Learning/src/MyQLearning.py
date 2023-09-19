from QLearning import QLearning

class MyQLearning(QLearning):
    def update_q(self, state, action, r, state_next, best_action, alfa, gamma):
        value = self.get_q(state, action) + alfa * (r + gamma * self.get_q(state_next, best_action) - self.get_q(state, action))
        self.set_q(state, action, value)
