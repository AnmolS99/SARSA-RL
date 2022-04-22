"""haakon8855, anmols99, mnottveit"""

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
        self.history = []
        self.best_episode_history = []

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
        # self.history = [(0, self.theta)]
        self.history = [(0, (self.theta_1, self.theta_1_der, self.theta_2,
                             self.theta_2_der))]

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
        self.history.append(
            (self.steps_taken, (self.theta_1, self.theta_1_der, self.theta_2,
                                self.theta_2_der)))

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
        Ending the episode by saving the history if it is the best one yet
        """
        if len(self.history) < len(self.best_episode_history):
            self.best_episode_history = self.history

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
        One hot encoding
        """
        # TODO: Must find out reasonable one hot encoding of state
        one_hot_theta_1 = self.one_hot_encode_sign(state[0])
        one_hot_theta_1_der = self.one_hot_encode_sign(state[1])
        one_hot_theta_2 = self.one_hot_encode_sign(state[2])
        one_hot_theta_2_der = self.one_hot_encode_sign(state[3])
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
        # TODO: If we decide to use this, must probably change idk
        # n = 3
        # one_hot = np.zeros((2 * n) + 1)
        # if number >= -(n - 1) and number <= (n - 1):
        #     one_hot[int(number) + n] = 1
        # elif number < -(n - 1):
        #     one_hot[0] = 1
        # else:
        #     one_hot[-1] = 1
        # return one_hot

    def show_best_history(self):
        """
        Showing the history of the best episode
        """
        # TODO: Do we really need history? We are probably just showing games live.
        # # Plotting the history (angle of the pole) of the best episode
        # timesteps = [i[0] for i in self.best_episode_history]
        # thetas = [i[1] for i in self.best_episode_history]
        # plt.plot(timesteps, thetas)
        # plt.xlabel("Timestep")
        # plt.ylabel("Angle (Radians)")
        # plt.show()

    def show_state(self):
        """
        Shows the given state in pyplot.
        """
        xp_1, yp_1, xtip, ytip = self.calculate_segment_positions(2, 2)
        plt.plot([2, xp_1, xtip], [2, yp_1, ytip])
        plt.show()


def main():
    """
    Main function for testing acrobat simworld
    """
    simworld = AcrobatSimWorld()
    simworld.begin_episode()
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()
    simworld.next_state(1)
    simworld.show_state()


if __name__ == '__main__':
    main()
