import pyautogui
class Agent:
    def shoot_first_gun(self, shoot):
        try:
            if shoot:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()
        except KeyboardInterrupt:
            print('\n')    

    def shoot_second_gun(self, shoot):
        try:
            if shoot:
                pyautogui.mouseDown(button="right")
            else:
                pyautogui.mouseUp(button="right")
        except KeyboardInterrupt:
            print('\n')

    def move_mouse(self, x_pos, y_pos):
        try:
            pyautogui.moveTo(x_pos, y_pos)
        except KeyboardInterrupt:
            print('\n')
