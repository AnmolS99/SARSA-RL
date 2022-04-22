import random


class Actor():
    """
    The Actor class
    """

    def __init__(self, lr, elig_decay, disc_factor, epsilon,
                 epsilon_decay_rate) -> None:
        self.lr = lr
        self.elig_decay = elig_decay
        self.disc_factor = disc_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        # Initializing the policy dictionary
        self.policy = {}

        # Initializing e(s, a) as empty dictionary
        self.elig = {}

    def get_policy(self, s, a):
        """
        Returns the policy
        """
        # If there is no policy for the pair (s, a), return 0
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.policy.get((tuple(s), a), 0)

    def get_optimal_action(self, s, valid_actions):
        """
        Returns the action with the highest value given the state (and the current policy)
        """
        optimal_action = None
        optimal_score = None

        for action in valid_actions:

            policy_score = self.get_policy(s, action)

            # If this action has a higher score than the current optimal one
            if optimal_score == None or policy_score > optimal_score:
                optimal_action = action
                optimal_score = policy_score

        return optimal_action

    def get_action(self, s, valid_actions):
        """
        Returns an action, with a probability of choosing a random action instead of the optimal one
        """
        # Having a probability of epsilon of choosing random action
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            return self.get_optimal_action(s, valid_actions)

    def get_elig_value(self, s, a):
        """
        Returns the eligibility trace value for a state and an action
        """
        # If no value is found for the elibility trace of the state-action pair, return 0
        # IMPORTANT: s is assumed to be a list and is therefore converted to tuple
        return self.elig.get((tuple(s), a), 0)

    def reset_elig(self):
        """
        Resetting eligibilities by setting it to an empty dictionary
        """
        self.elig = {}