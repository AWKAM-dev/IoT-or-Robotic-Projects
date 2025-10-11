#http://10.13.216.153/
#http://10.156.216.153/
#http://10.36.158.153

"""
import os
import time
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# ==== CONFIG ====
ESP32_IP = "http://10.156.216.153"
SAVE_DIR = "C:\\Users\\awkam\\OneDrive\\Desktop\\Code\\Hard\\Robbie\\Captures"
os.makedirs(SAVE_DIR, exist_ok=True)

CAPTURE_URL = f"{ESP32_IP}/capture"  # ESP32-CAM single image endpoint
CAPTURE_INTERVAL = 2  # Seconds between captures

# Load YOLOv8 model
model = YOLO("C:\\Users\\awkam\\OneDrive\\Desktop\\Code\\Hard\\Robbie\\YOLO training set\\A4Lit\\Results\\best.pt")

print("[INFO] Press SPACE to capture, process, and save an image.")
print("[INFO] Press Q to quit the program.")
counter = 10

# Open a blank window so cv2.waitKey works
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)

while True:
    try:
        key = cv2.waitKey(1) & 0xFF  # Check for keypress

        if key == ord(' '):  # SPACE pressed → capture image
            print("[INFO] Capturing image from ESP32...")
            response = requests.get(CAPTURE_URL, timeout=10)

            if response.status_code != 200:
                print("[ERROR] Could not fetch image from ESP32")
                time.sleep(CAPTURE_INTERVAL)
                continue

            # === CONVERT TO NUMPY IMAGE ===
            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Save captured image
            filename = os.path.join(SAVE_DIR, f"capture_{counter:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved image: {filename}")

            # === RUN YOLO DETECTION ===
            results = model(frame, verbose=False)

            # === PROCESS DETECTIONS ===
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Print detection results
                    print(f"Class: {model.names[int(cls)]}, Center: ({cx}, {cy}), Confidence: {conf:.2f}")

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

            # Show processed image
            cv2.imshow("YOLOv8 Detection", frame)

            counter += 1
            time.sleep(CAPTURE_INTERVAL)

        elif key == ord('q'):  # Q pressed → exit
            print("[INFO] Exiting program...")
            break

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(CAPTURE_INTERVAL)
        continue

cv2.destroyAllWindows()
"""

# import os
# import time
# import requests

# ESP32_IP = "http://10.156.18.153"
# SAVE_DIR = "C:\\Users\\awkam\\OneDrive\\Desktop\\Code\\Hard\\Robbie\\YOLO training set\\A4Lit\\LCD"
# os.makedirs(SAVE_DIR, exist_ok=True)

# interval = 2   # seconds between shots
# num_images = 2

# for i in range(num_images):
#     filename = os.path.join(SAVE_DIR, f"image_{i+1:04d}.jpg")
#     try:
#         r = requests.get(f"{ESP32_IP}/capture", timeout=5)
#         with open(filename, "wb") as f:
#             f.write(r.content)
#         print(f"Saved {filename}")
#     except Exception as e:
#         print(f"Error {e}")
#     time.sleep(interval)

"""
import os
import time
import requests
import cv2
import numpy as np

# ESP32-CAM IP address
ESP32_IP = "http://10.156.216.153"

# Directory to save images
SAVE_DIR = r"C:\\Users\\awkam\\OneDrive\\Desktop\\Code\\Hard\\Robbie\\YOLO training set\\A4Lit\\IRRec"
os.makedirs(SAVE_DIR, exist_ok=True)

# Interval between captures (if you want delays)
interval = 0  # seconds
counter = 0

print("[INFO] Press SPACE to capture an image, 'q' to quit.")

# Create a dummy window so cv2.waitKey works properly
cv2.namedWindow("Capture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Capture Control", 300, 50)
cv2.imshow("Capture Control", 255 * np.ones((50, 300, 3), dtype=np.uint8))  # white window

while True:
    key = cv2.waitKey(1) & 0xFF  # check keypress each frame

    if key == ord(' '):  # SPACE pressed
        counter += 1
        filename = os.path.join(SAVE_DIR, f"Under{counter:04d}.jpg")
        try:
            r = requests.get(f"{ESP32_IP}/capture", timeout=5)
            with open(filename, "wb") as f:
                f.write(r.content)
            print(f"[SAVED] {filename}")
        except Exception as e:
            counter -= 1
            print(f"[ERROR] {e}")
        time.sleep(interval)

    elif key == ord('q'):  # Quit
        print("[INFO] Exiting...")
        break

cv2.destroyAllWindows()
"""

"""
esp32_yolo_grid.py

Flow:
1) GET image from ESP32 (HTTP)
2) Run YOLOv8 model on the image
3) Compute object centers
4) Overlay a grid (configurable)
5) Convert pixel centers -> grid cell indices and real-world XY (cm)
6) Send coordinates over serial to Arduino

Configure in CONFIG section below.
"""

# esp32_yolo_grid_fixed.py
import cv2
import numpy as np
import requests
import serial
import time
from ultralytics import YOLO  # pip install ultralytics

# -------------------------
# CONFIG (edit these)
# -------------------------
ESP32_IP = "10.177.141.153"                # change to your ESP32 IP
CAPTURE_PATH = "/capture"               # change if your ESP32 uses a different endpoint
ESP_CAPTURE_URL = f"http://{ESP32_IP}{CAPTURE_PATH}"

MODEL_PATH = "/home/aditya/Desktop/IoT-or-Robotic-Projects/Robbie/YOLO/best.pt" #path to your YOLO model or "yolov8n.pt" for small pretrained
CONF_THRESHOLD = 0.3                    # detection confidence threshold

# Grid configuration:
GRID_COLS = 30                           # number of columns of the grid overlay
GRID_ROWS = 20                           # number of rows of the grid overlay

# Workspace real size mapping (optional)
WORKSPACE_WIDTH_CM = 30.0               # real width in cm corresponding to the image width
WORKSPACE_HEIGHT_CM = 20.0              # real height in cm corresponding to the image height

# Serial port to Arduino
SERIAL_PORT = "/dev/ttyUSB0"                    # e.g. "COM5" on Windows or "/dev/ttyUSB0" on Linux
SERIAL_BAUD = 9600

# Behavior
SAVE_DEBUG_IMAGE = True                 # save annotated image locally
DEBUG_IMAGE_PATH = "/home/aditya/Pictures/Robbie/annotated.jpg"
SEND_EACH_DETECTION = False             # if True, send coordinates for each detection as separate lines
LOOP_INTERVAL = 1.0                     # seconds between cycles if run in loop
RUN_ONCE = False                        # set True to run a single cycle then exit

# -------------------------
# End CONFIG
# -------------------------

# Initialize YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Initialize serial (lazy open)
ser = None
def open_serial():
    global ser
    if not SERIAL_PORT:
        return None
    if ser is None:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            time.sleep(2)  # allow Arduino to reset if needed. 20 if running tests else 2
            print(f"[INFO] Opened serial port {SERIAL_PORT} @ {SERIAL_BAUD}")
        except Exception as e:
            print(f"[WARN] Could not open serial port {SERIAL_PORT}: {e}")
            ser = None
    return ser

# Helper: request image from ESP32
def get_esp32_image(url):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image from ESP32")
        return img
    except Exception as e:
        raise RuntimeError(f"Error fetching image from ESP32: {e}")

# Helper: run YOLO and return list of detections (x1,y1,x2,y2,conf,class)
def run_yolo_on_image(img):
    # ultralytics returns a Results object; pass numpy array directly
    results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)
    if len(results) == 0:
        return []
    r = results[0]
    dets = []
    # r.boxes may be empty; handle gracefully
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        # boxes.xyxy is a tensor Nx4; boxes.conf Nx1; boxes.cls Nx1
        try:
            xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
            confs = boxes.conf.cpu().numpy().flatten()
            clss = boxes.cls.cpu().numpy().astype(int).flatten()
            for (box, conf, cls) in zip(xyxy, confs, clss): 
                x1, y1, x2, y2 = box
                dets.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
        except Exception:
            # fallback if access method differs
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy().flatten()[0])
                cls = int(box.cls.cpu().numpy().flatten()[0])
                x1, y1, x2, y2 = b
                dets.append((float(x1), float(y1), float(x2), float(y2), conf, cls))
    return dets

# Helper: overlay grid and detections
def annotate_image(img, detections, cols=GRID_COLS, rows=GRID_ROWS):
    out = img.copy()
    h, w = out.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    # draw grid lines
    for i in range(1, cols):
        x = int((i * cell_w))
        cv2.line(out, (x, 0), (x, h), (200, 200, 200), 1)
    for j in range(1, rows):
        y = int(    (j * cell_h))
        cv2.line(out, (0, y), (w, y), (200, 200, 200), 1)

    # draw detections
    for idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        # box and center
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        label = f"{cls}:{conf:.2f}"
        cv2.putText(out, label, (x1i, max(0, y1i-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # grid index
        col_idx = (cx // cell_w)
        row_idx = (cy // cell_h)
        cv2.putText(out, f"g=({col_idx},{row_idx})", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    return out

# Helper: convert center pixel -> (grid_col, grid_row) and real XY in cm (if workspace dims provided)
def pixel_to_grid_and_xy(cx, cy, img_w, img_h, cols, rows, workspace_w_cm=None, workspace_h_cm=None):
    cell_w = img_w / cols
    cell_h = img_h / rows
    col_idx = int(cx // cell_w)
    row_idx = int(cy // cell_h)
    real_x_cm = None
    real_y_cm = None
    if workspace_w_cm is not None and workspace_h_cm is not None:
        # map pixel center to real-world XY assuming origin at image left-top
        real_x_cm = (cx / img_w) * workspace_w_cm
        real_y_cm = (cy / img_h) * workspace_h_cm
    return col_idx, row_idx, real_x_cm, real_y_cm

# Helper: send coordinates via serial in a simple csv format
def send_coords_over_serial(port_obj, coords):
    """
    coords: list of dicts
    Format: "<x_cm>\n<y_cm>\n"
    Waits until Arduino sends back "DONE"
    """
    for obj in coords:
        xy = obj.get("xy", (None, None))
        if xy[0] is None or xy[1] is None:
            continue

        line = f"{xy[0]:.2f}\n{xy[1]:.2f}\n"
        try:
            port_obj.write(line.encode('utf-8'))
            port_obj.flush()
            print(f"[SERIAL] Sent: {line.strip()}")

            # ✅ wait for Arduino to signal completion
            while True:
                reply = port_obj.readline().decode(errors="ignore").strip()
                if reply:
                    print(f"[SERIAL] Arduino: {reply}")
                if reply == "DONE":
                    break

        except Exception as e:
            print(f"[ERROR] Failed to write to serial: {e}")


# Run one cycle: capture -> detect -> annotate -> send
def run_cycle():
    try:
        img = get_esp32_image(ESP_CAPTURE_URL)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    img_h, img_w = img.shape[:2]
    detections = run_yolo_on_image(img)
    print(f"[INFO] Detections: {len(detections)}")

    annotated = annotate_image(img, detections, cols=GRID_COLS, rows=GRID_ROWS)

    # Build coords list
    coords_list = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gcol, grow, rx, ry = pixel_to_grid_and_xy(
            cx, cy, img_w, img_h,
            GRID_COLS, GRID_ROWS,
            WORKSPACE_WIDTH_CM, WORKSPACE_HEIGHT_CM
        )

        coords_list.append({
            "class": cls,
            "conf": conf,
            "grid": (gcol, grow),
            "xy": (rx, ry)
        })
        
    # Also display image in a window (optional)
    try:
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
    except Exception:
        pass

    # Save / show annotated image if desired
    if SAVE_DEBUG_IMAGE:
        try:
            cv2.imwrite(DEBUG_IMAGE_PATH, annotated)
            print(f"[INFO] Saved annotated image to {DEBUG_IMAGE_PATH}")
        except Exception as e:
            print(f"[WARN] Could not save annotated image: {e}")

    # Open serial and send coords
    port = open_serial()
    if port is not None and len(coords_list) > 0:
        send_coords_over_serial(port, coords_list)


    # Print a short report
    for obj in coords_list:
        print(f"OBJ class={obj['class']} conf={obj['conf']:.2f} grid={obj['grid']} xy={obj['xy']}")

# Entry point
if __name__ == "__main__":
    print("[INFO] Starting ESP32 -> YOLO -> Serial pipeline")
    if SERIAL_PORT:
        open_serial()

    try:
        while True:
            run_cycle()
            if RUN_ONCE:
                break
            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
#NoteToSelf: grid unit is ~1.25 cm and trans_x = 3.75 and trans_y=10 (cm both)
