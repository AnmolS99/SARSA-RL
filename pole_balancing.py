import numpy as np
from matplotlib import pyplot as plt


class PoleBalancingSimWorld:
    """
    Pole Balancing simulation world
    """

    def __init__(self, l=0.5, m_p=0.1, g=-9.8, timestep=0.02) -> None:

        self.l = l  # Length of the pole
        self.m_p = m_p  # Mass of the pole
        self.g = g  # Gravity

        self.m_c = 1  # Mass of the cart
        self.theta = None
        self.theta_first_der = None
        self.theta_second_der = None
        self.x = None
        self.x_vel = 0
        self.x_acc = 0
        self.f = 10
        self.b = self.f
        self.theta_m = 0.21
        self.x_minus = -2.4
        self.x_plus = 2.4
        self.timestep = timestep
        self.episode_len = 300
        self.steps_taken = 0
        self.history = []
        self.best_episode_history = []

    def begin_episode(self):
        """
        Starting an episode
        """
        # Centering the cart at the horizontal position
        self.x = (self.x_minus + self.x_plus) / 2

        # Setting horizontal cart velocity to 0
        self.x_vel = 0

        # Randomly choosing theta (the pole angle)
        self.theta = np.random.uniform(-self.theta_m, self.theta_m)

        # Setting theta first temporal derivative to 0
        self.theta_first_der = 0

        # Resetting the number of steps taken in the current episode
        self.steps_taken = 0

        # Resetting the history, and adding the initial state to the history
        self.history = [(0, self.theta)]

        return self.get_current_state()

    def next_state(self, action):
        """
        Performing an action and going to the next state
        """

        if action == "left":
            self.b = -self.f
        elif action == "right":
            self.b = self.f
        else:
            print("Invalid action")

        # Calculating all the relationships
        theta_second_numerator = self.g * np.sin(self.theta) + np.cos(
            self.theta) * ((-self.b - self.m_p * self.l *
                            (self.theta_first_der**2) * np.sin(self.theta)) /
                           (self.m_c + self.m_p))

        theta_second_denominator = self.l * ((4 / 3) -
                                             ((self.m_p *
                                               (np.cos(self.theta)**2)) /
                                              (self.m_c + self.m_p)))
        self.theta_second_der = theta_second_numerator / theta_second_denominator

        self.x_acc = (self.b + self.m_p * self.l *
                      ((self.theta_first_der**2) * np.sin(self.theta) -
                       self.theta_second_der * np.cos(self.theta))) / (
                           self.m_p + self.m_c)

        self.x = self.x + self.timestep * self.x_vel
        self.x_vel = self.x_vel + self.timestep * self.x_acc

        self.theta = self.theta + self.timestep * self.theta_first_der
        self.theta_first_der = self.theta_first_der + self.timestep * self.theta_second_der

        # Calculate the reward
        if self.theta_in_range() and self.x_in_range():
            reward = 1 + (self.theta_m - abs(self.theta)) * 10 + (self.x_plus -
                                                                  abs(self.x))
        else:
            reward = -1000000

        # Increment number of steps taken
        self.steps_taken += 1

        # Adding the current step to the history
        self.history.append((self.steps_taken, self.theta))

        return self.get_current_state(), reward

    def end_episode(self):
        """
        Ending the episode by saving the history if it is the best one yet
        """
        if len(self.history) > len(self.best_episode_history):
            self.best_episode_history = self.history

    def show_best_history(self, delay):
        """
        Showing the history of the best episode
        """
        # Plotting the history (angle of the pole) of the best episode
        timesteps = [i[0] for i in self.best_episode_history]
        thetas = [i[1] for i in self.best_episode_history]
        plt.plot(timesteps, thetas)
        plt.xlabel("Timestep")
        plt.ylabel("Angle (Radians)")
        plt.show()

    def get_current_state(self):
        """
        Returns current state, but with rounded values so that the number of possible states stays relatively small
        """
        return self.one_hot_encode(
            (np.sign(self.x), np.round(self.x_vel), np.sign(self.theta),
             np.round(self.theta_first_der)))

    def get_valid_actions(self, state):
        """
        Returns list of actions that can be performed in a certain state, which will always be moving the cart left or right
        """
        return ["left", "right"]

    def theta_in_range(self):
        """
        Checking if theta (the angle) is in the valid range
        """
        return abs(self.theta) <= self.theta_m

    def x_in_range(self):
        """
        Checking if the cart is in the horizontal range
        """
        return self.x > self.x_minus and self.x < self.x_plus

    def is_end_state(self):
        """
        Checks whether s is an end state or not
        """
        return (not (self.theta_in_range() and self.x_in_range())
                ) or self.steps_taken >= self.episode_len

    def one_hot_encode(self, state):
        """
        One hot encoding
        """
        one_hot_x = self.one_hot_encode_sign(state[0])
        one_hot_x_vel = self.one_hot_encode_number(state[1])
        one_hot_theta = self.one_hot_encode_sign(state[2])
        one_hot_theta_first_der = self.one_hot_encode_number(state[3])
        return np.concatenate(
            (one_hot_x, one_hot_x_vel, one_hot_theta, one_hot_theta_first_der))

    def one_hot_encode_sign(self, number):
        """
        One hot encoding sign numbers (and 0)
        """
        one_hot = np.zeros(3)
        one_hot[int(number) + 1] = 1
        return one_hot

    def one_hot_encode_number(self, number):
        """
        One hot encoding numbers
        """
        n = 3
        one_hot = np.zeros((2 * n) + 1)
        if number >= -(n - 1) and number <= (n - 1):
            one_hot[int(number) + n] = 1
        elif number < -(n - 1):
            one_hot[0] = 1
        else:
            one_hot[-1] = 1
        return one_hot
