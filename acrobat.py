"""haakon8855, anmols99, mnottveit"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


class AcrobatSimWorld:
    """
    Pole Balancing simulation world.
    State representation: [theta1 (float), theta1' (float), theta2 (float), theta2' (float)]
    Action representation: action ∈ {-1, 0, 1}
    """

    def __init__(self,
                 max_steps: int = 3200,
                 gravity: float = 9.8,
                 timestep: float = 0.05) -> None:
        self.max_steps = max_steps
        self.gravity = gravity
        self.timestep = timestep

        self.l_1 = 1  # Length of upper segment
        self.l_2 = 1  # Length of lower segment
        self.lc_1 = 0.5  # Length from endpoint to center of mass for upper segment
        self.lc_2 = 0.5  # Length from endpoint to center of mass for lower segment
        self.m_1 = 1  # Mass of upper segment
        self.m_2 = 1  # Mass of lower segment

        self.theta_1 = None
        self.theta_1_der = None
        self.theta_2 = None
        self.theta_2_der = None

        self.force = 1  # Magnitude of the force applied

        self.episode_len = 400
        self.steps_taken = 0
        self.center_plot = (3, 3)
        self.history = []
        self.theta_hist = []
        self.theta1d_hist = []
        self.theta2d_hist = []

    def begin_episode(self):
        """
        Starting an episode by resetting state variables
        and returning the initial state.
        """
        # Setting both angles to zero (both segments hanging straight down)
        self.theta_1 = 0
        self.theta_2 = 0

        # Setting derivatives of both angles to 0 (no rotational velocity)
        self.theta_1_der = 0
        self.theta_2_der = 0

        # Resetting the number of steps taken in the current episode
        self.steps_taken = 0

        # Resetting the history, and adding the initial state to the history
        xp_1, yp_1 = self.center_plot
        xp_2, yp_2, xtip, ytip = self.calculate_segment_positions(xp_1, yp_1)
        x_coords = [self.center_plot[0], xp_2, xtip]
        y_coords = [self.center_plot[1], yp_2, ytip]
        self.history = [(x_coords, y_coords)]
        self.theta1d_hist = [self.theta_1_der]
        self.theta2d_hist = [self.theta_2_der]

        return self.get_current_state()

    def next_state(self, action: np.array):
        """
        Performing an action and going to the next state.
        """
        action = self.rev_one_hot_action(action)

        if action < 0:  # Force to left
            applied_force = -self.force
        elif action > 0:  # Force to right
            applied_force = self.force
        else:  # No force
            applied_force = 0

        reward = self.perform_actions(applied_force)

        return self.get_current_state(), reward

    def perform_actions(self, applied_force: int):
        """
        Performs one action given the applied force from that action.
        """
        reward = 0
        for _ in range(4):
            # Calculate the second derivatives of the angles
            theta_1_second_der, theta_2_second_der = self.calculate_angular_acceleration(
                applied_force)

            # Update the state variables with the calculated second derivatives
            self.theta_2_der = self.theta_2_der + self.timestep * theta_2_second_der
            self.theta_1_der = self.theta_1_der + self.timestep * theta_1_second_der
            self.theta_2 = self.theta_2 + self.timestep * self.theta_2_der
            self.theta_1 = self.theta_1 + self.timestep * self.theta_1_der
            if self.theta_2 > 2 * np.pi:
                self.theta_2 -= 2 * np.pi
            elif self.theta_2 < -2 * np.pi:
                self.theta_2 += 2 * np.pi
            if self.theta_1 > 2 * np.pi:
                self.theta_1 -= 2 * np.pi
            elif self.theta_1 < -2 * np.pi:
                self.theta_1 += 2 * np.pi

            # Increment number of steps taken
            self.steps_taken += 1

            # Adding the current step to the history
            xp_1, yp_1 = self.center_plot
            xp_2, yp_2, xtip, ytip = self.calculate_segment_positions(
                xp_1, yp_1)
            x_coords = [self.center_plot[0], xp_2, xtip]
            y_coords = [self.center_plot[1], yp_2, ytip]
            self.history.append((x_coords, y_coords))
            self.theta1d_hist.append(self.theta_1_der)
            self.theta2d_hist.append(self.theta_2_der)

            # Calculate the reward
            reward += self.calc_reward()

        return reward

    def calculate_angular_acceleration(self, applied_force):
        """
        Calculates the second derivatives of both thetas (angular acceleration)
        using the applied force given (left, right or no force).
        """
        phi_2 = self.m_2 * self.lc_2 * self.gravity * np.cos(self.theta_1 +
                                                             self.theta_2 -
                                                             np.pi / 2)
        phi_1 = (-self.m_2 * self.l_1 * self.lc_2 *
                 (self.theta_2_der**2) * np.sin(self.theta_2)) - (
                     2 * self.m_2 * self.l_1 * self.lc_2 * self.theta_2_der *
                     self.theta_1_der * np.sin(self.theta_2)) + (
                         (self.m_1 * self.lc_1 + self.m_2 * self.l_1) *
                         self.gravity * np.cos(self.theta_1 - np.pi / 2) +
                         phi_2)
        d_2 = self.m_2 * (
            (self.lc_2**2) + self.l_1 * self.lc_2 * np.cos(self.theta_2)) + 1
        d_1 = self.m_1 * (self.lc_1**2) + self.m_2 * (
            (self.l_1**2) + (self.lc_2**2) +
            2 * self.l_1 * self.lc_2 * np.cos(self.theta_2)) + 2
        theta_2_second_der = (pow(
            (self.m_2 * (self.lc_2**2) + 1 - (d_2**2) / d_1),
            -1)) * (applied_force +
                    (d_2 / d_1) * phi_1 - self.m_2 * self.l_1 * self.lc_2 *
                    (self.theta_1_der**2) * np.sin(self.theta_2) - phi_2)
        theta_1_second_der = -(d_2 * theta_2_second_der + phi_1) / d_1
        return theta_1_second_der, theta_2_second_der

    def calculate_segment_positions(self, xp_1=0, yp_1=0):
        """
        Calculates the positions of the pivot point p2
        and the tip of the second segment.
        """
        xp_2 = xp_1 + self.l_1 * np.sin(self.theta_1)
        yp_2 = yp_1 - self.l_1 * np.cos(self.theta_1)
        xtip = xp_2 + self.l_2 * np.sin(self.theta_2)
        ytip = yp_2 - self.l_2 * np.cos(self.theta_2)
        return xp_2, yp_2, xtip, ytip

    def end_episode(self):
        """
        Ending the episode.
        """

    def get_current_state(self):
        """
        Returns current state, but with rounded values so that the number
        of possible states stays relatively small
        """
        state = self.coarse_code_state(
            (self.theta_1, self.theta_1_der, self.theta_2, self.theta_2_der))
        flattened_state = state.flatten()
        return flattened_state

    def get_valid_actions(self, _):
        """
        Returns list of actions that can be performed in a certain state,
        which will always be moving the cart left or right
        """
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def is_end_state(self):
        """
        Checks whether current state is an end state or not.
        Current state is end state if tip is above goal height.
        """
        _, _, _, ytip = self.calculate_segment_positions()
        if ytip >= self.l_2:
            return True
        return False

    def calc_reward(self):
        """
        Calculating the reward based on height of tip
        """
        reward = 0
        if self.is_end_state():
            reward += 1000 / self.steps_taken
            return reward
        else:
            reward += -1

        # Tip lower segment distance from bottom
        _, _, _, ytip = self.calculate_segment_positions()
        bottom = -(self.l_1 + self.l_2)
        reward += (ytip - bottom)

        return reward

    def one_hot_encode(self, state):
        """
        One hot encoding.
        """
        one_hot_theta_1 = self.one_hot_encode_sign(state[0])
        one_hot_theta_1_der = self.one_hot_encode_number(state[1])
        one_hot_theta_2 = self.one_hot_encode_sign(state[2])
        one_hot_theta_2_der = self.one_hot_encode_number(state[3])
        return np.concatenate((one_hot_theta_1, one_hot_theta_1_der,
                               one_hot_theta_2, one_hot_theta_2_der))

    def one_hot_encode_sign(self, number):
        """
        One hot encoding sign numbers (and 0)
        """
        sign = np.sign(number)
        one_hot = np.zeros(3)
        one_hot[int(sign) + 1] = 1
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

    def coarse_code_state(self, state):
        """
        Returns a coarse coded representation of the given state.
        """
        theta_range = [-2 * np.pi, 2 * np.pi]
        theta_1_der_range = [-5, 5]
        theta_2_der_range = [-7, 7]
        theta_1 = state[0]
        theta_1_der = state[1]
        theta_2 = state[2]
        theta_2_der = state[3]
        one_hot_state = []
        one_hot_state += list(
            self.coarse_code_pair(theta_1, theta_1_der,
                                  [theta_range, theta_1_der_range], 6, [4, 4]))
        one_hot_state += list(
            self.coarse_code_pair(theta_2, theta_2_der,
                                  [theta_range, theta_2_der_range], 6, [4, 4]))
        one_hot_state += list(
            self.coarse_code_pair(theta_1, theta_2, [theta_range, theta_range],
                                  6, [4, 4]))
        one_hot_state += list(
            self.coarse_code_pair(theta_1_der, theta_2_der,
                                  [theta_1_der_range, theta_2_der_range], 6,
                                  [4, 4]))
        one_hot_state += list(
            self.coarse_code_pair(theta_1, theta_2_der,
                                  [theta_range, theta_2_der_range], 6, [4, 4]))
        one_hot_state += list(
            self.coarse_code_pair(theta_2, theta_1_der,
                                  [theta_range, theta_1_der_range], 6, [4, 4]))
        return np.array(one_hot_state).flatten()

    def coarse_code_pair(self, var1, var2, value_ranges, num_tilings,
                         num_tiles):
        """
        Returns the coarse coding for a pair of variables.
        value_ranges = [[var1.min, var1.max], [var2.min, var2.max]]
        """
        tilings = self.get_tilings(num_tilings, value_ranges, num_tiles)
        one_hot_tilings = []
        for tiling in tilings:
            one_hot_tiling = np.zeros(num_tiles[0] * num_tiles[1])
            x_coord = np.digitize(var1, tiling[0])
            y_coord = np.digitize(var2, tiling[1])
            one_hot_index = x_coord + y_coord * num_tiles[1]
            one_hot_tiling[one_hot_index] = 1
            one_hot_tilings.append(one_hot_tiling)
        return np.array(one_hot_tilings).flatten()

    def get_tilings(self, num_tilings: int, value_ranges: list,
                    num_tiles: list):
        """
        Return a list of tilings
        num_tilings = 4
        value_ranges = [[-6.28, 6.28], [-5, 5]]
        num_tiles = [10, 10] ten tiles in x and ten tiles in y
        """
        offsets = []
        for j, value_range in enumerate(value_ranges):
            value_range_size = value_range[1] - value_range[0]
            offset_magnitude = value_range_size / (2 * num_tiles[j])
            offsets.append(
                np.linspace(-offset_magnitude, offset_magnitude, num_tilings))
        tilings = []
        for i in range(num_tilings):
            tiling_i = []
            for j, value_range in enumerate(value_ranges):
                tiling = self.get_tiling(value_range, num_tiles[j],
                                         offsets[j][i])
                tiling_i.append(tiling)
            tilings.append(tiling_i)
        return np.array(np.round(tilings, 3))

    def get_tiling(self, value_range, num_tiles, offset):
        """
        Returns a list of numbers indicating the border between two bins.
        """
        tiling = np.linspace(value_range[0], value_range[1],
                             num_tiles + 1) + offset
        return tiling[1:-1]

    def rev_one_hot_action(self, one_hot_a):
        """
        Reversing one-hot encoding for action
        """
        if one_hot_a[0] == 1:
            return -1
        elif one_hot_a[1] == 1:
            return 0
        elif one_hot_a[2] == 1:
            return 1
        else:
            raise Exception("Invalid one-hot action")

    def show_episode(self, info: str = "", interval: int = 10):
        """
        Shows the given state in pyplot.
        """
        plt.plot(self.theta1d_hist)
        plt.show()
        plt.plot(self.theta2d_hist)
        plt.show()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(xlim=(0, 6), ylim=(0, 4.5))
        segments = ax.plot(self.history[0][0],
                           self.history[0][1],
                           color='red',
                           lw=5,
                           marker='o',
                           markerfacecolor='grey',
                           markersize=10,
                           markeredgecolor='black')[0]

        def animate(i):
            if i == (len(self.history) - 1):
                plt.close()
            x_coords = self.history[i][0]
            y_coords = self.history[i][1]
            segments.set_xdata(x_coords)
            segments.set_ydata(y_coords)
            ax.set_title(f"Step: {i}, r: {self.calc_reward()}, {info}")

        _ = FuncAnimation(fig,
                          animate,
                          interval=interval,
                          frames=len(self.history))
        plt.draw()
        plt.show()
        return segments
