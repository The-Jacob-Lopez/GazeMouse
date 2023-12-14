import tkinter as tk
from PIL import Image, ImageTk
import time
import math



class Circle:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.6)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
        self.canvas.pack()

        # self.num_refreshes = 0
        # self.start_time = time.time()

        # self.label = tk.Label(self.root, text="Hello, Tkinter!")
        # self.label.pack()
        # self.root.after(10, self.update)

        self.cursor_size = 5

        self.cursor = self.canvas.create_oval(395, 295, 405, 305, fill="red", outline="red")
        self.root.mainloop()

    # def update(self):
    #     self.num_refreshes += 1

    #     current_time = time.time()
    #     elapsed_time = current_time - self.start_time
    #     fps = int(self.num_refreshes / elapsed_time)

    #     self.label.config(text=f"FPS: {fps}")

    #     self.num_refreshes = 0
    #     self.start_time = current_time

    #     self.root.after(10, self.update)
    
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
    cursor = Circle()

