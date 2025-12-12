import cv2
import numpy as np
import cv2.aruco as aruco

# -----------------------------
# 1. ตั้งค่า ArUco detection
# -----------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# -----------------------------
# 2. โหลดรูปภาพที่จะ overlay
# -----------------------------
overlay_image = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED) # เปลี่ยนชื่อไฟล์ภาพที่ต้องการ overlay
if overlay_image is None:
    raise FileNotFoundError("ภาพ overlay.png ไม่พบ")

# -----------------------------
# 3. Function เพื่อวางภาพบน marker
# -----------------------------
def overlay_image_on_marker(frame, overlay_img, corners):
    pts_dst = np.array(corners[0], dtype=np.float32)
    h, w = overlay_img.shape[:2]
    
    # ตั้งค่าตำแหน่ง (มุมของภาพ overlay)
    pts_src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # สร้าง mask จากพื้นที่ที่ถูก warp
    matrix, _ = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(overlay_img, matrix, (frame.shape[1], frame.shape[0]))
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # ลบพื้นที่ใน frame ที่จะวางภาพ overlay
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    # เอาเฉพาะส่วนของภาพที่ถูก warp
    warped_fg = cv2.bitwise_and(warped, warped, mask=mask)
    # รวมภาพทั้งสอง
    frame[:] = cv2.add(frame_bg, warped_fg)
    
    return frame


# -----------------------------
# 4. เริ่มจับภาพจากกล้องและตรวจจับ ArUco markers
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Place an ArUco marker in front of the camera to see the overlay")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:        
        for i in range(len(ids)):
            frame = overlay_image_on_marker(frame, overlay_image, corners[i])

    cv2.imshow("ArUco Image Overlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
