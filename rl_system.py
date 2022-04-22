from actor import Actor
from critic import Critic
from matplotlib import pyplot as plt
import numpy as np


class RLSystem():
    """
    The reinforcement learning system, consisting of an actor and a critic
    """

    def __init__(self, sim_world, num_episodes, max_steps, critic_use_nn,
                 critic_nn_specs, actor_lr, critic_lr, actor_elig_decay,
                 critic_elig_decay, actor_disc_factor, critic_disc_factor,
                 epsilon, epsilon_decay_rate, display, delay) -> None:
        self.sim_world = sim_world
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.display = display
        self.delay = delay

        self.critic = Critic(critic_use_nn, critic_nn_specs, critic_lr,
                             critic_elig_decay, critic_disc_factor)
        self.actor = Actor(actor_lr, actor_elig_decay, actor_disc_factor,
                           epsilon, epsilon_decay_rate)

    def generic_actor_critic_algorithm(self):
        """
        The generic actor-critic algorithm
        """
        result_list = []

        # Repeating for each episode
        for i in range(self.num_episodes):

            # Resetting eligibilities in actor and critic
            self.actor.reset_elig()
            self.critic.reset_elig()

            # Decreasing epsilon for actor, since we want less exploration as number of episodes goes up
            # Every time we have been through a percent of num_episodes, epsilon is decreased
            if self.num_episodes >= 100:
                if i % (self.num_episodes // 100) == 0:
                    self.actor.epsilon *= (1 - self.actor.epsilon_decay_rate)
            else:
                self.actor.epsilon *= (1 - self.actor.epsilon_decay_rate)

            # Initializing state and action
            s = self.sim_world.begin_episode()
            a = self.actor.get_action(s, self.sim_world.get_valid_actions(s))

            # Initializing a list of steps taken in the current episode on the form [(s_0, a_0), (s_1, a_1), ...]
            episode_list = []

            # Initializing list of target values to the states in critic on the form [(s_0, V_star(s_0)), (s_1, V_star(s_1)), ...]
            V_star_list = []

            # Repeating for each step of the episode
            for _ in range(self.max_steps):
                # Adding the state s and action a we are in to the episode_list before taking the next step
                episode_list.append((s, a))

                # Performing the action a in state s and ending up in state s_next and recieving reward r
                s_next, r = self.sim_world.next_state(a)

                # Getting the action (a_next) to do in state s_next
                a_next = self.actor.get_action(
                    s_next, self.sim_world.get_valid_actions(s_next))

                # Setting the eligibility trace in the actor to 1
                self.actor.elig[(tuple(s), a)] = 1

                if self.critic.use_nn:
                    # Calculating V_star for s
                    V_star_s = self.critic.calculate_v_star(r, s_next)

                    # Adding V_star to the list
                    V_star_list.append((s, V_star_s))

                # Critic calculates TD-error
                td_error = self.critic.calculate_td_error(r, s, s_next)

                # Setting the eligibility trace in the critic to 1
                if not self.critic.use_nn:
                    self.critic.elig[tuple(s)] = 1

                # Going through each SAP in the current episode and updating values and policies
                for s_curr, a_curr in episode_list:

                    if not self.critic.use_nn:
                        # Critic updates current states value
                        self.critic.state_values[tuple(s_curr)] = (
                            self.critic.get_state_value(s_curr) +
                            self.critic.lr * td_error *
                            self.critic.get_elig_value(s_curr))

                        # Critic updates eligibility trace
                        self.critic.elig[tuple(s_curr)] = (
                            self.critic.disc_factor * self.critic.elig_decay *
                            self.critic.get_elig_value(s_curr))

                    # Actor updates policy
                    self.actor.policy[(tuple(s_curr), a_curr)] = (
                        self.actor.get_policy(s_curr, a_curr) + self.actor.lr *
                        td_error * self.actor.get_elig_value(s_curr, a_curr))

                    # Actor updates eligibility trace
                    self.actor.elig[(
                        tuple(s_curr), a_curr
                    )] = self.actor.disc_factor * self.actor.elig_decay * self.actor.get_elig_value(
                        s_curr, a_curr)

                # Setting the current state to s_next and current action to a_next
                s = s_next
                a = a_next

                # If we are in an end state, we end the episode
                if self.sim_world.is_end_state():
                    self.sim_world.end_episode()
                    break

            if self.critic.use_nn:
                # Training V_theta on each case
                states = [i[0] for i in V_star_list]
                V_star_states = [i[1] for i in V_star_list]
                s = np.array(V_star_states)
                self.critic.nn.fit(np.array(states),
                                   np.array(V_star_states),
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
