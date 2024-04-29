from gymnasium import ActionWrapper
from transform import *


class FlattenedActionWrapper(ActionWrapper):
    """ Flattens the action space of an `env` using
        `transform.flatten()`. This means that multiple
        discrete actions are joined to a single discrete
        action, and continuous (Box) spaces to a single
        vector valued action.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        trafo = flatten(env.action_space)
        self.action_space = trafo.target
        self.action = trafo.convert_from
