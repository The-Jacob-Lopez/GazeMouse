import tkinter as tk
from PIL import Image, ImageTk
import time
import math
from dataclasses import dataclass

@dataclass
class display_attributes:
    screen_width: int
    screen_height: int

@dataclass
class root_attributes:
    alpha: float
    fullscreen: bool

@dataclass
class cursor_attributes:
    fill: str
    outline: str

class app:
    def __init__(self, root_attributes: root_attributes, cursor_attributes: cursor_attributes):
        self.root_attributes = root_attributes
        self.cursor_attributes = cursor_attributes

        self.root = tk.Tk()
        self.display_attributes = display_attributes(self.root.winfo_screenwidth(),
                                           self.root.winfo_screenheight())

        # Init root
        self.root.attributes('-alpha', self.root_attributes.alpha)
        self.root.attributes('-fullscreen', self.root_attributes.fullscreen)

        # Init canvas
        self.canvas = tk.Canvas(self.root, width=self.display_attributes.screen_width, height=self.display_attributes.screen_height)
        self.canvas.pack()

        # Init Cursor
        self.cursor = self.canvas.create_oval(0, 0, 0, 0, fill="red", outline="red")  

        # Init Workers and product queues
        # TODO: make workers for background, cursor position and radius, and mouse action
        self.background_worker = ...
        self.cursor_worker = ...
        self.action_worker = ...

        # Init keybindings
        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def run(self):
        #TODO: erase this
        self.root.after(100, lambda: self.red_circle(500, 200, 10))
        #TODO: finish this
        self.start_update_background()
        self.root.mainloop()
    
    def start_update_background(self):
        pass

    def start_update_cursor(self):
        pass

    def start_update_action(self):
        pass

    def set_background(self, image):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        resized_image = image.resize((screen_width, screen_height), resample=Image.LANCZOS)
        bg_image = ImageTk.PhotoImage(resized_image)
        background_label = tk.Label(self.root, image=bg_image)
        background_label.photo = bg_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        
    def red_circle(self, x, y, radius):
        self.cursor_size = radius
        
        self.canvas.coords(self.cursor, x - self.cursor_size, y - self.cursor_size,
                            x + self.cursor_size, y + self.cursor_size)

if __name__ == "__main__":
    # Parameter setup
    root_attrib = root_attributes(alpha=0.6, fullscreen=True)
    cursor_attrib = cursor_attributes(fill='red', outline='red')

    # Run app
    app = app(root_attrib, cursor_attrib)
    app.run()

