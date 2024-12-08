import pygame
from ultralytics import YOLO
import keyboard
import threading
import bettercam
from pynput.mouse import Button, Controller
import math
import numpy as np
import time
import cv2
import torch


import pyautogui

# initialize the mouse controller
mouse = Controller()


def initialize_bomb_bbox(x1, y1, x2, y2):
    """
    initializes a numpy array representing a bomb's bounding box coordinates.
    """
    bomb_coords = np.array([x1, y1, x2, y2])
    return bomb_coords


def is_within_bomb(fruit_box, bomb_list):
    """
    checks if any corner of a fruit's bounding box is within any bomb's bounding box.
    """
    f_x1, f_y1, f_x2, f_y2 = fruit_box

    for bomb in bomb_list:
        b_x1, b_y1, b_x2, b_y2 = bomb

        # Check each corner of the fruit box
        if (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y1 <= b_y2) or \
                (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y2 <= b_y2) or \
                (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y1 <= b_y2) or \
                (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y2 <= b_y2):
            return True
    return False


def determine_safe_fruits(fruits, bombs, bottom_y):
    """
    determines which fruits are safe to slice by checking if they overlap with any bombs.
    """
    safe_fruits = []
    for fruit in fruits:
        center_x, center_y, width, height = fruit

        # skip fruits that are too low on the screen
        if center_y > bottom_y - 100:
            continue

        # calculate the bounding box of the fruit
        fruit_x1 = center_x - width / 2
        fruit_y1 = center_y - height / 2
        fruit_x2 = center_x + width / 2
        fruit_y2 = center_y + height / 2
        fruit_box = (fruit_x1, fruit_y1, fruit_x2, fruit_y2)



        # check if the fruit is not inside a bomb
        if not is_within_bomb(fruit_box, bombs):
            safe_fruits.append((center_x, center_y))  # add fruit to safe list if not inside a bomb

    return safe_fruits


def sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass


cached_cos_sin = {}


def move_mouse(radius, num_steps):
    """
    moves the mouse in a circular pattern around its current position.
    """
    # cache cosine and sine values for the circle to optimize performance
    if radius not in cached_cos_sin:
        angles = np.linspace(0, 2 * np.pi, num_steps)
        cos_vals = np.cos(angles) * radius
        sin_vals = np.sin(angles) * radius
        cached_cos_sin[radius] = (cos_vals, sin_vals)
    else:
        cos_vals, sin_vals = cached_cos_sin[radius]

    # calculate new mouse positions for the circular motion
    x_positions = mouse.position[0] + cos_vals
    y_positions = mouse.position[1] + sin_vals

    # move the mouse in a circle


    for i in range(len(cos_vals)):
        new_x = x_positions[i]
        new_y = y_positions[i]
        mouse.position = (new_x, new_y)
        sleep(0.000001)  # very short sleep to control mouse movement speed


def run_bot(safe_fruits):
    """
    simulates mouse actions to slice safe fruits.
    """
    for fruit_location in safe_fruits:
        fruit_x, fruit_y = fruit_location
        radius = 15  # radius for circular mouse motion
        num_steps = 20  # number of steps in the circular motion

        mouse.position = (fruit_x, fruit_y)  # move mouse to fruit location
        mouse.press(Button.left)  # press mouse button to start slicing
        move_mouse(radius, num_steps)  # perform circular slicing motion
        mouse.release(Button.left)  # release mouse button after slicing
        sleep(0.05) # short delay to simulate the slicing action


def take_screenshot(stop_event, model, device, region):

    """
    continuously takes screenshots and processes them using the object detection model.
    """

    # create the camera object after user-defined region is known
    camera = bettercam.create(output_color="BGR", region=region)

    while not stop_event.is_set():

        screenshot = camera.grab(region=region)  # capture a screenshot
        print("Screenshot taken")

        if screenshot is None:
            continue

        # run the object detection model on the screenshot
        # Setting `int8=True` if supported by your YOLO version, else remove it.
        results = model(source=screenshot, device=device, verbose=True, iou=0.25, int8=True, conf=0.5,
                        imgsz=(640, 640))

        detected_fruits, detected_bombs = [], []

        # skip if no objects are detected
        if detected_fruits is None:
            continue

        # parse detection results
        for box, cls in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist()):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # separate detected objects into fruits and bombs
            if results[0].names[int(cls)].lower() == "bomb":
                detected_bombs.append((x1 + region[0], y1 + region[1], x2, y2))
                cv2.rectangle(screenshot, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            elif results[0].names[int(cls)].lower() == "fruit":
                detected_fruits.append((center_x + region[0], center_y + region[1], width, height))
                cv2.rectangle(screenshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        cv2.imshow("Preview", screenshot)

        # determine which fruits are safe to slice
        safe_fruits = determine_safe_fruits(detected_fruits, detected_bombs, region[3])

        run_bot(safe_fruits)  # perform slicing action on safe fruits

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()  # close all OpenCV windows


def load_model(device):
    """
    loads the YOLO model for object detection.
    """
    model = YOLO('runs/detect/train2/weights/best.pt', task='detect')  # load the pre-trained model
    dummy_data = torch.zeros((1, 3, 640, 640))  # dummy input for model warm-up
    model(dummy_data, device=device, half=True, imgsz=(640, 640))  # warm up the model
    return model


def main():
    """
    main function to initialize and start the bot.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Let the user define the gameplay area


    print("Move your mouse to the TOP-LEFT corner of the gameplay window and press Enter.")
    keyboard.wait('enter')
    top_left_x, top_left_y = pyautogui.position()
    print(f"Top-left corner recorded at: ({top_left_x}, {top_left_y})")

    print("Now move your mouse to the BOTTOM-RIGHT corner of the gameplay window and press Enter.")
    keyboard.wait('enter')
    bottom_right_x, bottom_right_y = pyautogui.position()
    print(f"Bottom-right corner recorded at: ({bottom_right_x}, {bottom_right_y})")

    # Define the region from the selected corners
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    region = (top_left_x, top_left_y, top_left_x+width, top_left_y+height)

    model = load_model(device)  # load the detection model
    stop_event = threading.Event()
    screenshot_thread = threading.Thread(target=take_screenshot, args=(stop_event, model, device, region))
    screenshot_thread.start()

    print("Press 'q' to stop the bot.")
    keyboard.wait("q")  # wait for 'q' key to stop the bot
    stop_event.set()
    screenshot_thread.join()


if __name__ == "__main__":
    main()


