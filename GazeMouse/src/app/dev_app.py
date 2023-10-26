from tkinter import Tk, Label, Button
import cv2
from PIL import Image, ImageTk
import mss
import numpy as np


# Define a video capture object
vid = cv2.VideoCapture(0)

# Declare the width and height in variables
width, height = 800, 600

# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a GUI app
app = Tk()

# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind("<Escape>", lambda e: app.quit())

# Create a label and display it on app
label_widget = Label(app)
label_widget.pack()

# Create a function to open camera and
# display it in the label_widget on app


def open_camera():

    # Capture the video frame by frame
    _, frame = vid.read()

    # Convert image from one color space to other
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Capture the latest frame and transform to image
    captured_image = Image.fromarray(opencv_image)

    # Convert captured image to photoimage
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photoimage in the label
    label_widget.photo_image = photo_image

    # Configure image in the label
    label_widget.configure(image=photo_image)

    # Repeat the same process after every 10 milliseconds
    label_widget.after(10, open_camera)


def capture_screen():
    """
    Callback to capture and display screen content from monitor 1.
    """
    with mss.mss() as sct:
        # Select the default monitor
        monitor = sct.monitors[1]

        # Grab the screen content as bytes and convert to RGB
        image_bytes = np.asarray(sct.grab(monitor))
        image_bytes = cv2.cvtColor(image_bytes, cv2.COLOR_BGR2RGB)

        # Convert to PIL and resize to the label_widget
        captured_image = Image.fromarray(image_bytes)
        captured_image = _resize_image_to_window(captured_image)

        # Display in the label
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)
    
    # Repeat the same process after every 10 milliseconds
    label_widget.after(10, capture_screen)


def _resize_image_to_window(image):
    """
    Scales an image to fit within the height and width 
    """
    scaling_factor = min(width / image.width, height / image.height)
    new_width = int(image.width * scaling_factor)
    new_height = int(image.height * scaling_factor)
    return image.resize((new_width, new_height))



# Create a button to open the camera in GUI app
button1 = Button(app, text="Open Camera", command=open_camera)
button1.pack()

# Create a button to capture screen content
button2 = Button(app, text="Capture Screen", command=capture_screen)
button2.pack()

# Create an infinite loop for displaying app on screen
app.mainloop()
