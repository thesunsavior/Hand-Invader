from app.models.hand_detector import hand_detection_model
from app.util.train_util import train_hand_detection

if __name__ == "__main__":
    train_hand_detection(hand_detection_model, 2)