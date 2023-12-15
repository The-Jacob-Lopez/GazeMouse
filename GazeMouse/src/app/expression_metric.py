import numpy as np
from scipy import stats
from scipy.special import softmax

def normalize(a, axis=-1,):
    l2 = np.atleast_1d(np.linalg.norm(a, 2, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)

class expression_metric:
    def __init__(self, user_actions, action_expressions):
        self.user_actions = user_actions
        self.action_expressions = action_expressions
        
        feature_std = np.std(action_expressions, axis=0)
        self.feature_weights = softmax(feature_std)

    # Both expressions are numpy arrays of same size
    def expression_distance(self, expression_a, expression_b):
        expression_a = expression_a * self.feature_weights
        expression_b = expression_b * self.feature_weights
        return np.linalg.norm(expression_a - expression_b)
    
    def get_index_closest_user_action(self, expression):
        expression = np.array(expression)
        distances = np.array([expression_metric.expression_distance(self, action_expression, expression) for action_expression in self.action_expressions])
        return np.argmin(distances)
    
    def get_closest_user_action(self, expression):
       return self.user_actions[self.get_index_closest_user_action(expression)]
    
    def get_most_common_user_action(self, history):
        recent_user_action_indeces = [self.get_index_closest_user_action(expression) for expression in history]
        modes = stats.mode(recent_user_action_indeces)
        most_common_user_action_index = modes.mode
        return self.user_actions[most_common_user_action_index]