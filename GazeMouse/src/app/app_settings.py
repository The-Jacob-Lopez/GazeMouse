import mouse

class app_settings():
    def __init__(self, screen_resolution = [2560,1440]):
        self.screen_resolution = screen_resolution
        self.actions = ['neutral', 'left-click', 'right-click', 'shutdown']
        self.mouse_actions = [lambda : None, lambda : mouse.click(button='left'), lambda : self.set_mouse_tracking(True), lambda : self.set_mouse_tracking(False)]
        self.user_to_mouse_actions = {user:mouse for user, mouse in zip(self.actions, self.mouse_actions)}
        self.mouse_tracking_is_active = False
    
    def set_mouse_tracking(self, bool):
        self.mouse_tracking_is_active = bool
    
    def get_mouse_action(self, user_action):
        return self.user_to_mouse_actions[user_action]