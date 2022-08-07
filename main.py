"""haakon8855, anmols99, mnottveit"""

from acrobat import AcrobatSimWorld
from rl_system import RLSystem


def run():
    """
    Creating the SimWorld, Critic and RL-system and running the SARSA algorithm
    """
    # Creating the different sim worlds
    asw = AcrobatSimWorld()
    asw.begin_episode()
    input_len = len(asw.get_current_state()) + len(
        asw.get_valid_actions(None)[0])

    # Pole balancing with ANN-based critic
    rls = RLSystem(asw,
                   num_episodes=100,
                   max_steps=800,
                   critic_lr=0.001,
                   critic_disc_factor=0.99,
                   nn_specs=(input_len, 100, 1))
    rls.sarsa()


if __name__ == "__main__":
    run()
