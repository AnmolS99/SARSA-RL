"""haakon8855, anmols99, mnottveit"""

from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import pyplot as plt


class AcrobatSimWorld:
    """
    Pole Balancing simulation world.
    State representation: [theta1 (float), theta1' (float), theta2 (float), theta2 (float)']
    Action representation: action ∈ {-1, 0, 1}
    """

    def __init__(self, gravity=9.8, timestep=0.05) -> None:
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

        return self.get_current_state()

    def next_state(self, action: int):
        """
        Performing an action and going to the next state.
        """

        if action < 0:  # Force to left
            applied_force = -self.force
        elif action > 0:  # Force to right
            applied_force = self.force
        else:  # No force
            applied_force = 0

        # Calculate the second derivatives of the angles
        theta_1_second_der, theta_2_second_der = self.calculate_angular_acceleration(
            applied_force)

        # Update the state variables with the calculated second derivatives
        self.theta_2_der = self.theta_2_der + self.timestep * theta_2_second_der
        self.theta_1_der = self.theta_1_der + self.timestep * theta_1_second_der
        self.theta_2 = self.theta_2 + self.timestep * self.theta_2_der
        self.theta_1 = self.theta_1 + self.timestep * self.theta_1_der

        # Calculate the reward
        if self.is_end_state():
            reward = 1
        else:
            reward = 0

        # Increment number of steps taken
        self.steps_taken += 1

        # Adding the current step to the history
        xp_1, yp_1 = self.center_plot
        xp_2, yp_2, xtip, ytip = self.calculate_segment_positions(xp_1, yp_1)
        x_coords = [self.center_plot[0], xp_2, xtip]
        y_coords = [self.center_plot[1], yp_2, ytip]
        self.history.append((x_coords, y_coords))

        return self.get_current_state(), reward

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

    def calculate_segment_positions(self, xp_1, yp_1):
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
        self.show_episode()

    def get_current_state(self):
        """
        Returns current state, but with rounded values so that the number
        of possible states stays relatively small
        """
        return self.one_hot_encode(
            (self.theta_1, self.theta_1_der, self.theta_2, self.theta_2_der))

    def get_valid_actions(self):
        """
        Returns list of actions that can be performed in a certain state,
        which will always be moving the cart left or right
        """
        return [-1, 0, 1]

    def is_end_state(self):
        """
        Checks whether current state is an end state or not.
        Current state is end state if tip is above goal height.
        """
        _, _, _, ytip = self.calculate_segment_positions(0, 0)
        if ytip >= self.l_2:
            return True
        return False

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

    def show_episode(self, interval: int = 10):
        """
        Shows the given state in pyplot.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(xlim=(0, 6), ylim=(0, 4.5))
        segments = ax.plot(self.history[0][0],
                           self.history[0][1],
                           color='red',
                           lw=2)[0]

        def animate(i):
            x_coords = self.history[i][0]
            y_coords = self.history[i][1]
            segments.set_xdata(x_coords)
            segments.set_ydata(y_coords)
            ax.set_title(str(i))

        _ = FuncAnimation(fig, animate, interval=interval)
        plt.draw()
        plt.show()
        return segments


def main():
    """
    Main function for testing acrobat simworld
    """
    simworld = AcrobatSimWorld()
    simworld.begin_episode()
    for _ in range(200):
        simworld.next_state(1)
    # for _ in range(50):
    #     simworld.next_state(-1)
    for _ in range(1000):
        simworld.next_state(0)
    simworld.show_episode(interval=20)


if __name__ == '__main__':
    main()
