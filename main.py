from acrobat import AcrobatSimWorld
from rl_system import RLSystem


def run():
    # Setting random seeds to be able to reproduce results
    # seed = 1001
    # random.seed(seed)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    # Creating the different sim worlds
    asw = AcrobatSimWorld()
    asw.begin_episode()
    input_len = len(asw.get_current_state()) + len(
        asw.get_valid_actions(None)[0])

    # Pole balancing with ANN-based critic
    rls = RLSystem(asw, 200, 500, 0.5, 0.99, (input_len, 30, 1))
    rls.sarsa()


if __name__ == "__main__":
    run()
