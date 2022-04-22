from matplotlib import pyplot as plt
from gambler import GamblerSimWorld
from pole_balancing import PoleBalancingSimWorld
from rl_system import RLSystem
from towers_of_hanoi import TowersOfHanoiSimWorld
import random
import numpy as np
import tensorflow as tf


def show_optimal_state_action_gambler(gsw, rls):
    wager = []
    for s in range(1, 101):
        one_hot_s = gsw.one_hot_encode(s)
        wager.append(
            rls.actor.get_optimal_action(one_hot_s,
                                         gsw.get_valid_actions(one_hot_s)))
    plt.plot(wager)
    plt.xlabel("State")
    plt.ylabel("Wager")
    plt.vlines(x=[12.5, 25, 37.5, 50, 62.5, 75, 87.5],
               ymin=[0, 0, 0, 0, 0, 0, 0],
               ymax=[12.5, 25, 37.5, 50, 37.5, 25, 12.5],
               colors="red",
               linestyles="dotted")
    plt.show()


def run():
    # Setting random seeds to be able to reproduce results
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Creating the different sim worlds
    pbsw = PoleBalancingSimWorld(l=1, m_p=0.3, g=5, timestep=0.5)
    num_pegs = 5
    num_discs = 6
    tohsw = TowersOfHanoiSimWorld(num_pegs, num_discs)
    gsw = GamblerSimWorld(0.6)

    # UNCOMMENT THE RLSYSTEM (rls) YOU WANT TO RUN

    # Pole balancing with table-based critic
    rls = RLSystem(pbsw, 2, 300, False, None, 0.3, 0.3, 0.5, 0.5, 0.99, 0.99,
                   0.5, 0.04, True, 1)

    # Pole balancing with ANN-based critic
    # rls = RLSystem(pbsw, 3, 300, True, (20, 20, 20, 40, 5, 1), 0.3, 0.3, 0.5,
    #                0.5, 0.5, 0.5, 0.5, 0.1, True, 1)

    # Towers of Hanoi with table-based critic
    # rls = RLSystem(tohsw, 10, 300, False, None, 0.3, 0.03, 0.5, 0.5, 0.99,
    #                0.99, 0.9, 0.03, True, 1)

    # Towers of Hanoi with ANN-based critic
    # rls = RLSystem(tohsw, 10, 300, True, (num_pegs * num_discs, 30, 30, 30, 1),
    #                2, 0.1, 0.5, 0.5, 0.99, 0.99, 0.9, 0.03, True, 1)

    # The Gambler with table-based critic
    # rls = RLSystem(gsw, 100, 300, False, None, 0.5, 0.05, 0.5, 0.5, 1, 1, 0.9,
    #                0.075, True, 1)

    # The Gambler with ANN-based critic
    # rls = RLSystem(gsw, 10, 300, True, (101, 5, 1), 0.08, 0.01, 0.5, 0.5, 1, 1,
    #                0.9, 0.075, True, 1)

    rls.generic_actor_critic_algorithm()

    if rls.sim_world == gsw:
        show_optimal_state_action_gambler(gsw, rls)


if __name__ == "__main__":
    run()
