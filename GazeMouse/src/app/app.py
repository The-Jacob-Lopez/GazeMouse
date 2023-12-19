import tkinter as tk
from PIL import Image, ImageTk
import time
import math
from dataclasses import dataclass
import numpy as np
import random
from PIL import Image
from PIL import ImageDraw

@dataclass
class root_attributes:
    alpha: float
    fullscreen: bool

@dataclass
class cursor_attributes:
    fill: str
    outline: str

class cursor_provider:
    def __init__(self):
        pass
    
    def get_cursor(self):
        x = random.uniform(500,750)
        y = random.uniform(500,750)
        stddev = random.uniform(50, 150)
        return [x,y], int(stddev)

class background_provider:
    def __init__(self):
        pass
    
    def get_background(self):
        imarray = np.random.rand(500,500,3) * 128
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        return im

class mouse_provider:
    pass

class app:
    def __init__(self, root_attributes: root_attributes, cursor_attributes: cursor_attributes):
        self.root_attributes = root_attributes
        self.cursor_attributes = cursor_attributes

        # Init root
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', self.root_attributes.fullscreen)

        # Init canvas
        self.canvas = tk.Canvas(self.root, background='green')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.imgsize = (self.canvas.winfo_screenwidth(), self.canvas.winfo_screenheight())
        empty_image = ImageTk.PhotoImage(Image.new('RGB', self.imgsize), size=self.imgsize)
        self.canvas.imgref = empty_image
        self.image_on_canvas = self.canvas.create_image(0, 0, image=empty_image, anchor='nw')

        # Init Cursor
        self.cursor = self.canvas.create_oval(0, 0, 0, 0, fill="red", outline="red")  

        # Init keybindings
        self.root.bind("<Escape>", lambda event: self.root.destroy())

        # Init providers
        self.cursor_provider = cursor_provider()
        self.background_provider = background_provider()
        self.mouse_provider = mouse_provider()

        # Internal State
        self.curr_cursor_pos = [0,0]
        self.curr_cursor_radius = 0
        self.curr_background = Image.new('RGB', (self.canvas.winfo_screenwidth(), self.canvas.winfo_screenwidth()))

    def run(self):
        self.start_update_background()
        self.start_update_cursor()
        self.start_update_window()
        self.root.mainloop()
    
    def start_update_background(self):
        background = self.background_provider.get_background()
        if background is None:
            return
        self.curr_background = background.resize(self.imgsize, resample=Image.LANCZOS)
        self.root.after(100, self.start_update_background)  

    def start_update_cursor(self):
        data = self.cursor_provider.get_cursor()
        if data is None:
            return
        pos, stddev = data
        self.curr_cursor_pos = pos
        self.curr_cursor_radius = stddev
        self.root.after(100, self.start_update_cursor)  

    def start_update_action(self):
        pass
    
    def start_update_window(self):
        curr_background = self.curr_background.copy().convert('RGBA')
        transp = Image.new('RGBA', curr_background.size, (0,0,0,0))  # Temp drawing image.
        draw = ImageDraw.Draw(transp, "RGBA")
        curr_x, curr_y  = self.curr_cursor_pos
        r = self.curr_cursor_radius
        transparent_red = (255, 128, 10, 50)
        draw.ellipse((curr_x-r, curr_y-r, curr_x+r, curr_y+r), fill = transparent_red, outline=transparent_red)
        curr_background.paste(Image.alpha_composite(curr_background, transp))
        curr_background = ImageTk.PhotoImage(curr_background, size=self.imgsize)    
        self.canvas.imgref = curr_background
        self.canvas.itemconfig(self.image_on_canvas, image=curr_background)
        self.root.after(100, self.start_update_window)

if __name__ == "__main__":
    # Parameter setup
    root_attrib = root_attributes(alpha=0.6, fullscreen=True)
    cursor_attrib = cursor_attributes(fill='red', outline='red')

    # Run app
    app = app(root_attrib, cursor_attrib)
    app.run()

