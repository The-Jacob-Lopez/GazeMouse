import mouse

class app_settings():
    def __init__(self, screen_resolution = [2560,1440]):
        self.screen_resolution = screen_resolution
        self.actions = ['neutral', 'left-click', 'right-click', 'shutdown']
        self.mouse_actions = [lambda : None, lambda : mouse.click(button='left'), mouse.click(button='right'), lambda : None]
        self.user_to_mouse_actions = {user:mouse for user, mouse in zip(self.actions, self.mouse_actions)}
    
    def get_mouse_action(self, user_action):
        return self.user_to_mouse_actions[user_action]