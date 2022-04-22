from critic import Critic
from matplotlib import pyplot as plt
import numpy as np


class RLSystem():
    """
    The reinforcement learning system, consisting of an actor and a critic
    """

    def __init__(self, sim_world, num_episodes, max_steps, critic_lr, critic_disc_factor,
                 display, delay, nn_specs) -> None:
        self.sim_world = sim_world
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.display = display
        self.delay = delay

        self.critic = Critic(critic_lr, nn_specs, critic_disc_factor, 0, 0)

    def sarsa(self):
        """
        The SARSA algorithm
        """
        result_list = []

        # Repeating for each episode
        for i in range(self.num_episodes):

            # Decreasing epsilon for actor, since we want less exploration as number of episodes goes up
            # Every time we have been through a percent of num_episodes, epsilon is decreased
            if self.num_episodes >= 100:
                if i % (self.num_episodes // 100) == 0:
                    self.critic.epsilon *= (1 - self.critic.epsilon_decay_rate)
            else:
                self.critic.epsilon *= (1 - self.critic.epsilon_decay_rate)

            # Initializing state and action
            s = self.sim_world.begin_episode()
            a = self.critic.get_action(s, self.sim_world.get_valid_actions(s))


            # Initializing list of target values to the states in critic on the form [(s_0, V_star(s_0)), (s_1, V_star(s_1)), ...]
            targets = []

            # Repeating for each step of the episode
            for _ in range(self.max_steps):

                # Performing the action a in state s and ending up in state s_next and recieving reward r
                s_next, r = self.sim_world.next_state(a)

                # Getting the action (a_next) to do in state s_next
                a_next = self.critic.policy(
                    s_next, self.sim_world.get_valid_actions(s_next))
            
                # Calculating V_star for s
                target = r + self.critic.disc_factor * self.critic.Q(s_next, a_next)

                # Adding V_star to the list
                targets.append(((s, a), target))

                
                # Setting the current state to s_next and current action to a_next
                s = s_next
                a = a_next

                # If we are in an end state, we end the episode
                if self.sim_world.is_end_state():
                    self.sim_world.end_episode()
                    break

            # Training V_theta on each case
            s_a = [i[0] for i in targets]
            y = [i[1] for i in targets]
            self.critic.nn.fit(np.array(s_a),
                                np.array(y),
                                verbose=0)

            # Storing the number of steps taken in the current (finished) episode
            result_list.append(self.sim_world.steps_taken)

        # Plotting the result list
        plt.plot(result_list)
        plt.xlabel("Episode")
        plt.ylabel("Timestep")
        plt.show()

        if self.display:
            # Showing the history of the best episode
            self.sim_world.show_best_history(self.delay)
