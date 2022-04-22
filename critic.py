import random
import numpy as np
import tensorflow as tf

class Critic:

    """
    The Critic class
    """

    def __init__(self, lr, nn_specs, disc_factor, epsilon,
                 epsilon_decay_rate) -> None:
        self.lr = lr
        self.nn = self.create_nn(nn_specs)
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate


    def create_nn(self, nn_specs):
        """
        Creating a neural network according to the specifications
        """
        # Converting specs from tuple to list
        nn_specs_list = list(nn_specs)

        # Creating a list of layers
        layers = []

        # Adding input layer
        input_neurons = nn_specs_list[0]
        layers.append(tf.keras.layers.Input((input_neurons, )))

        # Adding all hidden layers
        for layer_neurons in nn_specs_list[1:-1]:
            layers.append(
                tf.keras.layers.Dense(layer_neurons, activation="tanh"))

        # Adding output layer
        output_neurons = nn_specs_list[-1]
        layers.append(tf.keras.layers.Dense(output_neurons))

        # Creating the neural network model
        model = tf.keras.Sequential(layers)

        # Selecting the optimizer (Adam seems to be the best with adaptive learning rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model with MSE loss function
        model.compile(optimizer, "mean_squared_error")
        return model

    def Q(self, s, a):
        """
        Returns the policy
        """
        # If there is no policy for the pair (s, a), return 0
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        s_a = np.concatenate(s, a)
        return self.nn(s_a[None])

    def get_optimal_action(self, s, valid_actions):
        """
        Returns the action with the highest value given the state (and the current policy)
        """
        optimal_action = None
        optimal_score = None

        for action in valid_actions:

            policy_score = self.Q(s, action)

            # If this action has a higher score than the current optimal one
            if optimal_score == None or policy_score > optimal_score:
                optimal_action = action
                optimal_score = policy_score

        return optimal_action

    def policy(self, s, valid_actions):
        """
        Returns an action, with a probability of choosing a random action instead of the optimal one
        """
        # Having a probability of epsilon of choosing random action
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            return self.get_optimal_action(s, valid_actions)
        
