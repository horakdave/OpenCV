import pyautogui

def find_image_and_move_mouse(image_path):
    while True:
        try:
            position = pyautogui.locateOnScreen(image_path)
            if position is not None:
                center = pyautogui.center(position)
                pyautogui.moveTo(center)
                print("the image was found")
                break
            else:
                print("the Image was not found")
        except pyautogui.FailSafeException:
            print("error")
            break

#                                                                                               kde
image_path = 'your/own/path'

#                                                                                               proces
find_image_and_move_mouse(image_path)