import random
import tensorflow as tf


class Criticc():
    """
    The Critic class
    """

    def __init__(self, use_nn, nn_specs, lr, elig_decay, disc_factor) -> None:

        self.lr = lr  # Learning rate (needs to be declared first as it is used in creation of NN)

        self.use_nn = use_nn

        if self.use_nn:
            self.nn = self.create_nn(nn_specs)
        else:
            self.nn = None

        self.elig_decay = elig_decay  # Eligibility decay
        self.disc_factor = disc_factor  # Discount factor

        # Initializing V(s) as empty dictionary
        self.state_values = {}

        # Initializing e(s) as empty dictionary
        self.elig = {}

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

    def get_state_value(self, s):
        """
        Returns the value of a state
        """
        # If no value is found for the state, return a small random number
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.state_values.get(tuple(s), random.random() * 0.5)

    def get_elig_value(self, s):
        """
        Returns the eligibility trace value for a state
        """
        # If no value is found for the elibility trace of the state, return 0
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.elig.get(tuple(s), 0)

    def reset_elig(self):
        """
        Resetting eligibilities by setting it to an empty dictionary, only if critic is table-based
        """
        if not self.use_nn:
            self.elig = {}

    def calculate_td_error(self, r, s, s_next):
        """
        Calculating the TD-error
        """
        if self.use_nn:
            return self.calculate_v_star(r, s_next) - self.calculate_v_theta(s)
        else:
            return r + self.disc_factor * self.get_state_value(
                s_next) - self.get_state_value(s)

    def calculate_v_star(self, r, s_next):
        """
        Calculating V_star of state s (which is not included in calculations), using s_next
        """
        return r + self.disc_factor * self.calculate_v_theta(s_next)

    def calculate_v_theta(self, s):
        """
        Using NN model to predict value of state s
        """
        return self.nn(s[None])
