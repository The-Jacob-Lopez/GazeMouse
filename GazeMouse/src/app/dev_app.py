from tkinter import Tk, Label, Button
from PIL import ImageTk
from multiprocessing import Queue, freeze_support
from src.app.screen_recorder import screen_recorder
from src.app.webcam_recorder import webcam_recorder
from src.app.predictive_webcam_recorder import predictive_webcam_recorder
from src.app.mediapipe_webcam_recorder import mediapipa_webcam_recorder
import multiprocessing
from src.app.mediapipe_webcam_recorder import draw_landmarks_on_image
from PIL import Image

# Multiprocessed variables instantiated as global
screen_capture_queue = Queue(maxsize=2)
webcam_capture_queue = Queue(maxsize=2)
tracker_pred_queue = Queue(maxsize=2)
detector_pred_queue = Queue(maxsize=2)

screen_capture = screen_recorder(screen_capture_queue, width = 800, height = 600)
#webcam_capture = webcam_recorder(webcam_capture_queue, width = 800, height = 600)
webcam_capture = predictive_webcam_recorder(webcam_capture_queue, tracker_pred_queue, detector_pred_queue)

def run_app():
 
    app = Tk()

    # Shut down all processes on app close
    def shutdown():
        screen_capture.stop_recording()
        webcam_capture.stop_processing()
        app.quit()
 
    # Key bindings
    app.bind("<Escape>", lambda e: shutdown())

    # Window Geometry
    app.geometry('350x200')

    # Create a label and display it on app
    webcam_capture_widget = Label(app)
    screen_capture_widget = Label(app)

    # Organize the widgets
    webcam_capture_widget.pack()
    screen_capture_widget.pack()

    # Displays the webcam capture on the given widget
    def open_camera():

        captured_image = webcam_capture_queue.get()
        
        # TODO: connect this prediction to mouse input
        tracker_pred = tracker_pred_queue.get()
        
        # TODO: connect this for facial expression detection
        detector_pred = detector_pred_queue.get()
        
        #overlay detector prediction on face 
        captured_image = Image.fromarray(draw_landmarks_on_image(captured_image, detector_pred))
        
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Displaying photoimage in the label
        webcam_capture_widget.photo_image = photo_image

        # Configure image in the label
        webcam_capture_widget.configure(image=photo_image)

        # Repeat the same process after every 10 milliseconds
        webcam_capture_widget.after(10, open_camera)

    # Displays the screen capture on the given widget
    def capture_screen():
        """
        Callback to capture and display screen content from monitor 1.
        """
        captured_image = screen_capture_queue.get()
        
        # Display in the label
        photo_image = ImageTk.PhotoImage(image=captured_image)
        screen_capture_widget.photo_image = photo_image
        screen_capture_widget.configure(image=photo_image)

        # Repeat the same process after every 10 milliseconds
        screen_capture_widget.after(10, capture_screen)

    # Create a button to open the camera in GUI app
    button1 = Button(app, text="Open Camera", command=open_camera)
    button1.pack()

    # Create a button to capture screen content
    button2 = Button(app, text="Capture Screen", command=capture_screen)
    button2.pack()

    # Enables capturing
    multiprocessing.Process(target=screen_capture.start_recording).start()
    multiprocessing.Process(target=webcam_capture.start_processing).start()

    # Create an infinite loop for displaying app on screen
    app.mainloop()

if __name__ == "__main__":
    freeze_support()
    run_app()

