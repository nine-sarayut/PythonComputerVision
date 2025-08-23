import cv2
from ultralytics import YOLO

# Video link: https://drive.google.com/file/d/1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-/view

lane_points = []
# Mouse callback function เพื่อรับพิกัดและคำนวณ slope
def mouse_callback(event, x, y, flags, param):
    global lane_points
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / ratio)
        orig_y = int(y / ratio)
        print(f"Mouse clicked at: X={orig_x}, Y={orig_y}")
        lane_points.append((orig_x, orig_y))

        if len(lane_points) == 2:
            # คำนวณ slope และ intercept จากสองจุด
            x1, y1 = lane_points[0]
            x2, y2 = lane_points[1]

            if y2 != y1:  # ป้องกันการหารด้วยศูนย์
                slope = (x2 - x1) / (y2 - y1)
                intercept = x2 - slope * y2
                print(f"\nLane divider calculated:")
                print(f"slope = {slope:.3f}")
                print(f"intercept = {intercept:.1f}")
                print(f"Equation: x = {slope:.3f} * y + {intercept:.1f}")

            lane_points.clear()  # รีเซ็ตสำหรับการคำนวณครั้งถัดไป

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
lane_divider_slope = 0.376    # คำนวนจากสูตรเส้นตรง
lane_divider_intercept = 1485.7  # คำนวนจากสูตรเส้นตรง
divider_x_at_out = get_lane_divider_x(line_y_out)
divider_x_at_in = get_lane_divider_x(line_y_in)
name = "YOLO car count"

class_count_in = {}
class_count_out = {}
crossed_in_ids  = set()
crossed_out_ids = set()

# โหลด YOLO model
model = YOLO('yolo12l.pt')
class_list = model.names
# อ่าน video 
cap = cv2.VideoCapture('vehicle-counting.mp4')
cv2.namedWindow(name)
cv2.setMouseCallback(name, mouse_callback)

while cap.isOpened():
    ret, frame= cap.read()
    new_width = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)
    if not ret:
        break

    draw_sloped_lane_divider(frame, y_start=700, y_end=frame.shape[0])
    # Left Lane (OUT)
    cv2.line(frame, (0, line_y_out), (divider_x_at_out, line_y_out), (0, 0, 255), 3)
    cv2.putText(frame, "OUT (left)", (700, line_y_out - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    # Right Lane (IN)
    cv2.line(frame, (divider_x_at_in, line_y_in), (frame.shape[1], line_y_in), (0, 255, 0), 3)
    cv2.putText(frame, "IN (right)", (divider_x_at_in, line_y_in - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    results = model.track(frame, persist=True, classes=[2,7], device=0, verbose=False)
    if results[0].boxes.data is not None:
        # เก็บค่าที่ต้องการใช้
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()
    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2 
        class_name = class_list[class_idx]

        divider_x_at_vehicle = get_lane_divider_x(cy)

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

        if cx > divider_x_at_vehicle and cy < line_y_in and track_id not in crossed_in_ids:
            crossed_in_ids.add(track_id)
            class_count_in[class_name] = class_count_in.get(class_name, 0) + 1
        if cx < divider_x_at_vehicle and cy > line_y_out and track_id not in crossed_out_ids:
            crossed_out_ids.add(track_id)
            class_count_out[class_name] = class_count_out.get(class_name, 0) + 1
    # แสดงจำนวนบนรูป
    y_offset = 30
    cv2.putText(frame, "VEHICLES IN (Right Lane):", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 25
    
    for class_name, count in class_count_in.items():
        cv2.putText(frame, f"{class_name}: {count}", (70, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
    y_offset += 10
    cv2.putText(frame, "VEHICLES OUT (Left Lane):", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_offset += 25
    
    for class_name, count in class_count_out.items():
        cv2.putText(frame, f"{class_name}: {count}", (70, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
    scaled_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow(name, scaled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Button q pressed")
        break

cap.release()
cv2.destroyAllWindows()