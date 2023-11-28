import numpy as np

def normalize(a, axis=-1,):
    l2 = np.atleast_1d(np.linalg.norm(a, 2, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class expression_metric:
    def __init__(self, user_actions, action_expressions):
        self.user_actions = user_actions
        self.action_expressions = normalize(action_expressions, axis=0)

    # Both expressions are numpy arrays of same size
    def expression_distance(expression_a, expression_b):
        return np.linalg.norm(expression_a - expression_b)
    
    def get_closest_user_action(self, expression):
        expression = np.array(expression)
        distances = np.array([expression_metric.expression_distance(action_expression, expression) for action_expression in self.action_expressions])
        return self.user_actions[np.argmin(distances)]