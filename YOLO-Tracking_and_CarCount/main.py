import cv2
from ultralytics import YOLO

# Video link: https://drive.google.com/file/d/1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-/view

# Mouse callback function to get image coordinate
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert scaled coordinates to original
        orig_x = int(x / ratio)
        orig_y = int(y / ratio)
        print(f"Mouse clicked at: X={orig_x}, Y={orig_y}")

def get_lane_divider_x(y):
    """Calculate x coordinate of lane divider at given y coordinate"""
    return int(lane_divider_slope * y + lane_divider_intercept)

def draw_sloped_lane_divider(frame, y_start, y_end):
    """Draw sloped lane divider based on curb"""
    x_start = get_lane_divider_x(y_start)
    x_end = get_lane_divider_x(y_end)
    cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 3)


# ตั้งค่าตัวแปรต่างๆ
ratio = 0.5  # สเกลของรูปภาพที่แสดง (กำหนดเอง)
line_y_in = 1300      # ค่าแกน y (ฝั่งขาเข้า, ขวา) (กำหนดเอง)
line_y_out = line_y_in   # ค่าแกน y (ฝั่งขาออก, ซ้าย) (กำหนดเอง)
lane_divider_slope = 0.409    # คำนวนจากสูตรเส้นตรง
lane_divider_intercept = 1459.6  # คำนวนจากสูตรเส้นตรง
divider_x_at_out = get_lane_divider_x(line_y_out)
divider_x_at_in = get_lane_divider_x(line_y_in)

# โหลด YOLO model
model = YOLO('yolo12l.pt')
class_list = model.names

# อ่าน video 
cap = cv2.VideoCapture('vehicle-counting.mp4')

while cap.isOpened():
    ret, frame= cap.read()
    new_width = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)
    if not ret:
        break
    scaled_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow("frame", scaled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Button q pressed")
        break

cap.release()
cv2.destroyAllWindows()